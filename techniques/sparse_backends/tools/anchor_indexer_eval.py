"""Offline routing-quality study: mean pooling vs. anchor (first-order Taylor) indexer.

For every (query block i, KV block j) the ground truth is the exact block
log-partition ``log2 sum_t exp(s q.k_t)`` averaged over the queries of block i.
Each indexer produces a proxy score for the same pairs; we report

* recall of the true top-rho blocks at a matched per-row budget,
* Spearman rank correlation between proxy and truth per row,
* relative L2 error of the Sol-Attn output (routing by proxy top-k at the same
  budget, approximate branch using that indexer's own summaries) vs. dense.

Inputs are synthetic by default (RoPE-like locality + optional needles) or a
``.pt`` file holding ``{"q": [B,T,H,D], "k": ..., "v": ...}`` dumped from a model.

    python tools/anchor_indexer_eval.py --tokens 4096 --anchors 1,4,8 --temps inf,2,1
    python tools/anchor_indexer_eval.py --source dump.pt --heads 0,1 --density 0.15
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sol_attn.anchor import block_mean_queries, prepare_anchor  # noqa: E402
from sol_attn.reference import (  # noqa: E402
    block_log2_partition,
    dense_reference,
    sol_attn_reference,
)

D = 128
LOG2E = math.log2(math.e)


def synthetic_qkv(tokens, heads, *, seed=0, regions=8, needle_frac=0.25, device="cpu"):
    """Content clusters per region + rotary position mixing + optional needles."""

    gen = torch.Generator().manual_seed(seed)
    blocks = (tokens + 63) // 64
    region_of_token = torch.arange(tokens) * regions // tokens
    content = torch.nn.functional.normalize(torch.randn(regions, heads, D, generator=gen), dim=-1) * math.sqrt(D)
    base = content[region_of_token]  # [T, H, D]
    q = base * 0.8 + torch.randn(tokens, heads, D, generator=gen) * 0.5
    k = base * 0.8 + torch.randn(tokens, heads, D, generator=gen) * 0.5
    v = torch.randn(tokens, heads, D, generator=gen)

    # RoPE over half the dims with a spread of frequencies -> locality.
    half = D // 4
    freqs = 1.0 / (10000 ** (torch.arange(half) / half))
    angle = torch.arange(tokens)[:, None] * freqs[None, :]  # [T, half]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]

    def rope(x):
        x1, x2, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)

    q, k = rope(q), rope(k)

    # Needles: some blocks hide one key copied from a random far-away query.
    needle_blocks = torch.rand(blocks, generator=gen) < needle_frac
    for j in torch.nonzero(needle_blocks).flatten().tolist():
        src = int(torch.randint(0, tokens, (1,), generator=gen))
        k[j * 64] = q[src] * 1.1
    q = q[None].to(torch.bfloat16).contiguous()
    k = k[None].to(torch.bfloat16).contiguous()
    v = v[None].to(torch.bfloat16).contiguous()
    return q.to(device), k.to(device), v.to(device)


def proxy_scores(q, out):
    """[B, H, N, N] proxy routing scores (log2 domain) from prepared summaries."""

    batch, tokens, heads, _ = q.shape
    blocks = (tokens + 63) // 64
    num_anchors = out["num_anchors"]
    q_bar = block_mean_queries(q)  # [B, N, H, D]
    kc = out["kc"][:, :blocks].float().view(batch, blocks, heads, num_anchors, D)
    bias = out["bias"][:, :blocks].view(batch, blocks, heads, num_anchors)
    scores = torch.einsum("bihd,bjhad->bihaj", q_bar, kc) * (D**-0.5) * LOG2E
    scores = scores + bias.permute(0, 2, 3, 1)[:, None]  # [B, N(i), H, A, N(j)]
    idx = out["assign"].long()[..., None, None].expand(-1, -1, -1, 1, blocks)
    return torch.gather(scores, 3, idx).squeeze(3).permute(0, 2, 1, 3)  # [B, H, N, N]


def spearman(a, b):
    ra = a.argsort(dim=-1).argsort(dim=-1).float()
    rb = b.argsort(dim=-1).argsort(dim=-1).float()
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    return ((ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-9)).mean().item()


def recall_at_budget(proxy, truth, budget, exclude_neighbors=True):
    blocks = truth.shape[-1]
    if exclude_neighbors:
        pos = torch.arange(blocks, device=truth.device)
        near = (pos[:, None] - pos[None, :]).abs() <= 1
        truth = truth.masked_fill(near, float("-inf"))
        proxy = proxy.masked_fill(near, float("-inf"))
    top_true = truth.topk(budget, dim=-1).indices
    top_proxy = proxy.topk(budget, dim=-1).indices
    hits = (top_true[..., :, None] == top_proxy[..., None, :]).any(-1).float().sum(-1)
    return (hits / budget).mean().item()


def topk_route(proxy, budget):
    blocks = proxy.shape[-1]
    pos = torch.arange(blocks, device=proxy.device)
    near = (pos[:, None] - pos[None, :]).abs() <= 1
    route = torch.zeros_like(proxy, dtype=torch.bool) | near[None, None]
    idx = proxy.masked_fill(near, float("-inf")).topk(budget, dim=-1).indices
    route.scatter_(-1, idx, True)
    return route


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="synthetic", help="'synthetic' or a .pt with q/k/v")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", default="2", help="synthetic: head count; .pt: comma list of head ids")
    parser.add_argument("--anchors", default="1,4,8")
    parser.add_argument("--temps", default="inf,4,2,1")
    parser.add_argument("--modes", default="position")
    parser.add_argument("--density", type=float, default=0.15, help="exact-block budget per row")
    parser.add_argument("--needles", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-output-error", action="store_true", help="skip the (slow) attention output error")
    args = parser.parse_args()

    if args.source == "synthetic":
        q, k, v = synthetic_qkv(args.tokens, int(args.heads), seed=args.seed, needle_frac=args.needles, device=args.device)
    else:
        blob = torch.load(args.source, map_location=args.device)
        q, k, v = (blob[name].to(torch.bfloat16).contiguous() for name in ("q", "k", "v"))
        if args.heads:
            ids = [int(x) for x in args.heads.split(",")]
            q, k, v = (x[:, :, ids].contiguous() for x in (q, k, v))
        if args.tokens and q.shape[1] > args.tokens:
            q, k, v = (x[:, : args.tokens].contiguous() for x in (q, k, v))

    batch, tokens, heads, _ = q.shape
    blocks = (tokens + 63) // 64
    budget = max(1, int(round(args.density * blocks)))
    print(f"tokens={tokens} blocks={blocks} heads={heads} budget={budget}/{blocks} ({budget / blocks:.1%}) device={args.device}")

    truth = block_log2_partition(q, k, scale=D**-0.5)  # [B, H, T, N]
    pad = blocks * 64 - tokens
    if pad:
        truth = torch.nn.functional.pad(truth, (0, 0, 0, pad))
    truth = truth.view(batch, heads, blocks, 64, blocks)
    counts = torch.full((blocks,), 64.0, device=truth.device)
    counts[-1] = tokens - (blocks - 1) * 64
    truth = truth.sum(dim=3) / counts[None, None, :, None]  # mean over valid queries

    dense = None if args.no_output_error else dense_reference(q, k, v, scale=D**-0.5).float()

    header = f"{'indexer':<28}{'recall@budget':>14}{'spearman':>10}{'rel-L2 out':>12}"
    print(header)
    print("-" * len(header))

    configs = [(1, math.inf, "position")]  # mean pooling baseline
    for mode in args.modes.split(","):
        for a in (int(x) for x in args.anchors.split(",")):
            for t in (float(x) for x in args.temps.split(",")):
                if math.isinf(t):
                    continue  # T=inf is mean pooling regardless of anchors/mode
                configs.append((a, t, mode))

    for num_anchors, temperature, mode in configs:
        out = prepare_anchor(
            q, k, v, scale=D**-0.5, tau=1.0, num_anchors=num_anchors, temperature=temperature, anchor_mode=mode
        )
        proxy = proxy_scores(q, out)
        rec = recall_at_budget(proxy, truth, budget)
        rho = spearman(proxy, truth)
        if dense is not None:
            route = topk_route(proxy, budget)
            approx = sol_attn_reference(
                q, k, v, scale=D**-0.5, kc=out["kc"], vc=out["vc"], bias=out["bias"], assign=out["assign"],
                threshold=out["threshold"], route=route,
            ).float()
            err = ((approx - dense).norm() / dense.norm()).item()
            err_s = f"{err:12.4f}"
        else:
            err_s = f"{'-':>12}"
        label = "mean pooling" if math.isinf(temperature) else f"anchor A={num_anchors} T={temperature:g} {mode}"
        print(f"{label:<28}{rec:14.3f}{rho:10.3f}{err_s}")


if __name__ == "__main__":
    main()
