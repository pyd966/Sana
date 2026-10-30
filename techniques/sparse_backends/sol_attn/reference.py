"""Device-agnostic PyTorch reference of the Sol-Attn forward pass.

The reference mirrors the Triton kernel's semantics in the log2 domain:

* for query block ``i`` the per-token score tile against the block summaries
  is ``s2 * Q_i . kc_j`` (plus the anchor bias when present);
* a KV block is exact when the column mean of that tile exceeds the query
  block's threshold, when ``|i - j| <= 1``, or when it overlaps the sink range;
* unselected blocks contribute ``exp2(score) * vc_j`` to the numerator and
  ``exp2(score) * len_j`` (mean-pooling) or ``exp2(score)`` (anchor, the block
  count lives in the bias) to the denominator;
* selected blocks are computed exactly.

It is written for correctness and offline studies, not speed.
"""

from __future__ import annotations

import math

import torch

BLOCK_SIZE = 64


def _num_blocks(tokens: int) -> int:
    return (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def route_mask_reference(
    q: torch.Tensor,
    kc: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale: float,
    bias: torch.Tensor | None = None,
    assign: torch.Tensor | None = None,
    sink_blocks: tuple[int, int] | None = None,
    neighbor_exact: bool = True,
) -> torch.Tensor:
    """Boolean exact-block mask ``[B, H, N, N]`` (query block, KV block)."""

    batch, tokens, heads, head_dim = q.shape
    blocks = _num_blocks(tokens)
    num_anchors = kc.shape[2] // heads
    log2_scale = float(scale) * math.log2(math.e)
    device = q.device

    from .anchor import block_mean_queries

    q_bar = block_mean_queries(q)  # [B, N, H, D]
    if assign is None:
        assign = torch.zeros((batch, blocks, heads), dtype=torch.long, device=device)
    pseudo_head = (
        torch.arange(heads, device=device)[None, None, :] * num_anchors + assign.long()
    )  # [B, N, H]

    kc_f = kc[:, :blocks].float()  # [B, N, HA, D]
    # scores[b, i, h, j] = s2 * q_bar[b,i,h] . kc[b, j, pseudo_head[b,i,h]]
    kc_sel = kc_f[
        torch.arange(batch, device=device)[:, None, None, None],
        torch.arange(blocks, device=device)[None, None, None, :],
        pseudo_head[..., None],
    ]  # [B, N, H, N, D]
    scores = torch.einsum("bihd,bihjd->bihj", q_bar, kc_sel) * log2_scale
    if bias is not None:
        bias_sel = bias[:, :blocks].float()[
            torch.arange(batch, device=device)[:, None, None, None],
            torch.arange(blocks, device=device)[None, None, None, :],
            pseudo_head[..., None],
        ]  # [B, N, H, N]
        scores = scores + bias_sel
    exact = scores > threshold[..., None]  # [B, N(i), H, N(j)]
    exact = exact.permute(0, 2, 1, 3)  # [B, H, N, N]

    if neighbor_exact:
        pos = torch.arange(blocks, device=device)
        exact = exact | ((pos[:, None] - pos[None, :]).abs() <= 1)[None, None]
    if sink_blocks is not None:
        start, end = sink_blocks
        if end > start:
            sink = torch.zeros(blocks, dtype=torch.bool, device=device)
            sink[start:end] = True
            exact = exact | sink[None, None, None, :]
    return exact


def sol_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kc: torch.Tensor,
    vc: torch.Tensor,
    threshold: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
    assign: torch.Tensor | None = None,
    sink_blocks: tuple[int, int] | None = None,
    route: torch.Tensor | None = None,
    return_route: bool = False,
):
    """Sol-Attn forward in FP32 from precomputed summaries.

    ``bias is None`` selects mean-pooling semantics (``vc`` is the block value
    sum and the denominator is weighted by block length); otherwise ``vc`` is
    the anchor-weighted mean and ``bias`` already carries the block count.
    ``route`` may override the routing mask (``[B, H, N, N]`` bool).
    """

    batch, tokens, heads, head_dim = q.shape
    blocks = _num_blocks(tokens)
    num_anchors = kc.shape[2] // heads
    log2_scale = float(scale) * math.log2(math.e)
    device = q.device

    if route is None:
        route = route_mask_reference(
            q,
            kc,
            threshold,
            scale=scale,
            bias=bias,
            assign=assign,
            sink_blocks=sink_blocks,
        )
    if assign is None:
        assign = torch.zeros((batch, blocks, heads), dtype=torch.long, device=device)

    lengths = torch.full((blocks,), float(BLOCK_SIZE), device=device)
    lengths[-1] = float(tokens - (blocks - 1) * BLOCK_SIZE)

    out = torch.empty_like(q)
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    kc_f, vc_f = kc[:, :blocks].float(), vc[:, :blocks].float()
    bias_f = None if bias is None else bias[:, :blocks].float()

    for b in range(batch):
        for h in range(heads):
            for i in range(blocks):
                q_start, q_end = i * BLOCK_SIZE, min(tokens, (i + 1) * BLOCK_SIZE)
                q_i = q_f[b, q_start:q_end, h]  # [m, D]
                ph = h * num_anchors + int(assign[b, i, h])
                exact = route[b, h, i]  # [N]

                approx_scores = (q_i @ kc_f[b, :, ph].T) * log2_scale  # [m, N]
                if bias_f is not None:
                    approx_scores = approx_scores + bias_f[b, :, ph][None, :]
                approx_scores = approx_scores.masked_fill(exact[None, :], float("-inf"))

                exact_idx = torch.nonzero(exact, as_tuple=False).flatten()
                if exact_idx.numel():
                    tok = (exact_idx[:, None] * BLOCK_SIZE + torch.arange(BLOCK_SIZE, device=device)[None, :]).flatten()
                    tok_valid = tok < tokens
                    tok = tok[tok_valid]
                    exact_scores = (q_i @ k_f[b, tok, h].T) * log2_scale  # [m, n_exact]
                    exact_values = v_f[b, tok, h]
                else:
                    exact_scores = q_i.new_full((q_i.shape[0], 0), float("-inf"))
                    exact_values = v_f.new_zeros((0, head_dim))

                row_max = torch.maximum(
                    approx_scores.max(dim=1).values,
                    exact_scores.max(dim=1).values if exact_scores.shape[1] else approx_scores.new_full((q_i.shape[0],), float("-inf")),
                )
                p_approx = torch.exp2(approx_scores - row_max[:, None])
                p_exact = torch.exp2(exact_scores - row_max[:, None])

                numerator = p_approx @ vc_f[b, :, ph] + p_exact @ exact_values
                if bias_f is None:
                    denominator = (p_approx * lengths[None, :]).sum(dim=1) + p_exact.sum(dim=1)
                else:
                    denominator = p_approx.sum(dim=1) + p_exact.sum(dim=1)
                out[b, q_start:q_end, h] = (numerator / denominator[:, None]).to(q.dtype)

    if return_route:
        return out, route
    return out


def block_log2_partition(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Exact per-block ``log2 sum_t exp(s q.k_t)`` for every query token.

    Returns ``[B, H, T, N]`` FP32.  This is the quantity an indexer tries to
    approximate; ground truth for routing-quality studies.
    """

    batch, tokens, heads, head_dim = q.shape
    blocks = _num_blocks(tokens)
    log2_scale = float(scale) * math.log2(math.e)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * log2_scale
    pad = blocks * BLOCK_SIZE - tokens
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(batch, heads, tokens, blocks, BLOCK_SIZE)
    return torch.logsumexp(scores * math.log(2.0), dim=-1) / math.log(2.0)


def dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float) -> torch.Tensor:
    """FP32 dense attention on BTHD tensors."""

    out = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        scale=float(scale),
    ).transpose(1, 2)
    return out.to(q.dtype)


__all__ = [
    "block_log2_partition",
    "dense_reference",
    "route_mask_reference",
    "sol_attn_reference",
]
