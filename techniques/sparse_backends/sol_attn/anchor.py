"""First-order Taylor (anchor) indexer for Sol-Attn.

Sol-Attn summarises every 64-token KV block by its mean key and its summed
value.  That summary is the first-order Taylor expansion of the block
log-partition function ``f_j(q) = log sum_{k in block j} exp(s q.k)`` around the
origin ``q' = 0``.  This module moves the expansion point to one or more
per-head *anchor* queries:

    p_t   = softmax_t( s (q' / T) . k_t )        (softmax over the block)
    k'_j  = sum_t p_t k_t                         (anchor-weighted key)
    v'_j  = sum_t p_t v_t                         (anchor-weighted value)
    b'_j  = -sum_t p_t log p_t                    (entropy, "log effective keys")
    f_j(q) ~= s q . k'_j + b'_j                   (tangent at q'/T)

``T`` is the anchor temperature.  Because ``softmax(s (q'/T).k)`` equals the
softmax at anchor ``q'/T``, temperature slides the expansion point along the
line from the anchor to the origin; ``T -> inf`` recovers mean pooling.

The tensors returned by :func:`prepare_anchor` follow the layout the Triton
reference kernel consumes:

* ``kc``/``vc``: ``[batch, padded_blocks, heads * anchors, head_dim]`` BF16,
  pseudo-head index ``head * anchors + anchor``;
* ``bias``: ``[batch, padded_blocks, heads * anchors]`` FP32, ``b'`` in log2
  units so it adds directly to the log2-domain score tile;
* ``assign``: ``[batch, blocks, heads]`` INT32, anchor id per query block;
* ``threshold``: ``[batch, blocks, heads]`` FP32 routing thresholds in log2
  units, computed in closed form from the moments of ``[s k'_j ; b'_j]``.

Everything here is plain PyTorch and device agnostic; it is a research path
and keeps full FP32 copies of K/V, so it is not the fast preprocessing.
"""

from __future__ import annotations

import math

import torch

BLOCK_SIZE = 64
SUMMARY_PAD = 64
ANCHOR_MODES = ("position", "kmeans")


def _num_blocks(tokens: int) -> int:
    return (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def _padded_blocks(blocks: int) -> int:
    return (blocks + SUMMARY_PAD - 1) // SUMMARY_PAD * SUMMARY_PAD


def _pad_tokens(x: torch.Tensor, blocks: int) -> torch.Tensor:
    """Zero-pad the token axis of a ``[B, T, H, D]`` tensor to ``blocks * 64``."""

    tokens = x.shape[1]
    padded = blocks * BLOCK_SIZE
    if padded == tokens:
        return x
    pad = x.new_zeros((x.shape[0], padded - tokens) + tuple(x.shape[2:]))
    return torch.cat([x, pad], dim=1)


def _token_valid(tokens: int, blocks: int, device) -> torch.Tensor:
    """``[blocks, 64]`` mask of real (non-padding) tokens."""

    index = torch.arange(blocks * BLOCK_SIZE, device=device).view(blocks, BLOCK_SIZE)
    return index < tokens


def block_mean_queries(q: torch.Tensor) -> torch.Tensor:
    """Masked per-block mean of ``q``: ``[B, T, H, D] -> [B, N, H, D]`` FP32."""

    batch, tokens, heads, head_dim = q.shape
    blocks = _num_blocks(tokens)
    valid = _token_valid(tokens, blocks, q.device)
    q_blocks = _pad_tokens(q.float(), blocks).view(
        batch, blocks, BLOCK_SIZE, heads, head_dim
    )
    q_sum = (q_blocks * valid[None, :, :, None, None]).sum(dim=2)
    count = valid.sum(dim=1).to(q_sum.dtype)
    return q_sum / count[None, :, None, None]


def compute_anchors(
    q: torch.Tensor,
    num_anchors: int,
    *,
    mode: str = "position",
    kmeans_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose per-head anchor queries and assign every query block to one.

    Returns ``anchors`` ``[B, H, A, D]`` FP32 and ``assign`` ``[B, N, H]`` INT32.

    * ``position``: split the ``N`` query blocks into ``A`` contiguous ranges
      along the token axis and use each range's mean query as its anchor.  With
      Morton or raster token order a range is a spatio-temporal region, so the
      anchor shares the RoPE phase of the queries it serves.
    * ``kmeans``: start from the positional anchors and refine them with a few
      k-means iterations on the block-mean queries.
    """

    if num_anchors < 1:
        raise ValueError("num_anchors must be >= 1")
    if mode not in ANCHOR_MODES:
        raise ValueError(f"anchor mode must be one of {ANCHOR_MODES}")

    q_bar = block_mean_queries(q)  # [B, N, H, D]
    batch, blocks, heads, head_dim = q_bar.shape
    anchors_n = min(num_anchors, blocks)

    # Positional initialisation: contiguous, near-equal ranges of query blocks.
    range_id = (
        torch.arange(blocks, device=q.device) * anchors_n // blocks
    )  # [N] in [0, anchors_n)
    assign = range_id[None, :, None].expand(batch, blocks, heads).contiguous()
    anchors = _centroids_from_assign(q_bar, assign, anchors_n)

    if mode == "kmeans":
        for _ in range(int(kmeans_iters)):
            # Squared distance from every block-mean query to every anchor.
            dist = torch.cdist(
                q_bar.permute(0, 2, 1, 3).reshape(batch * heads, blocks, head_dim),
                anchors.reshape(batch * heads, anchors_n, head_dim),
            )  # [B*H, N, A]
            new_assign = dist.argmin(dim=-1).view(batch, heads, blocks).permute(0, 2, 1)
            new_assign = new_assign.contiguous()
            if torch.equal(new_assign, assign):
                break
            assign = new_assign
            anchors = _centroids_from_assign(q_bar, assign, anchors_n, previous=anchors)

    if anchors_n < num_anchors:
        # Fewer blocks than requested anchors: pad with copies so the layout
        # stays [H * num_anchors]; the extra anchors are never assigned.
        pad = anchors[:, :, -1:].expand(batch, heads, num_anchors - anchors_n, head_dim)
        anchors = torch.cat([anchors, pad], dim=2)
    return anchors.contiguous(), assign.to(torch.int32)


def _centroids_from_assign(
    q_bar: torch.Tensor,
    assign: torch.Tensor,
    num_anchors: int,
    *,
    previous: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of the block-mean queries in each cluster: ``[B, H, A, D]``."""

    batch, blocks, heads, head_dim = q_bar.shape
    one_hot = torch.nn.functional.one_hot(assign.long(), num_anchors).to(q_bar.dtype)
    # [B, N, H, A] x [B, N, H, D] -> [B, H, A, D]
    sums = torch.einsum("bnha,bnhd->bhad", one_hot, q_bar)
    counts = one_hot.sum(dim=1)  # [B, H, A]
    centroids = sums / counts.clamp_min(1.0)[..., None]
    if previous is not None:
        empty = counts == 0
        centroids = torch.where(empty[..., None], previous, centroids)
    return centroids


def anchor_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    anchors: torch.Tensor,
    *,
    scale: float,
    temperature: float = math.inf,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anchor-weighted block summaries.

    ``k``/``v`` are ``[B, T, H, D]``; ``anchors`` is ``[B, H, A, D]``.  Returns
    ``k_prime`` and ``v_prime`` as ``[B, N, H, A, D]`` FP32 and ``entropy`` as
    ``[B, N, H, A]`` FP32 in natural-log units.
    """

    if not temperature > 0:
        raise ValueError("anchor temperature must be positive")
    batch, tokens, heads, head_dim = k.shape
    blocks = _num_blocks(tokens)
    num_anchors = anchors.shape[2]
    valid = _token_valid(tokens, blocks, k.device)  # [N, 64]

    k_blocks = _pad_tokens(k.float(), blocks).view(
        batch, blocks, BLOCK_SIZE, heads, head_dim
    )
    v_blocks = _pad_tokens(v.float(), blocks).view(
        batch, blocks, BLOCK_SIZE, heads, head_dim
    )

    k_prime = k.new_empty((batch, blocks, heads, num_anchors, head_dim), dtype=torch.float32)
    v_prime = torch.empty_like(k_prime)
    entropy = k.new_empty((batch, blocks, heads, num_anchors), dtype=torch.float32)

    for a in range(num_anchors):
        if math.isinf(temperature):
            logits = k_blocks.new_zeros((batch, blocks, BLOCK_SIZE, heads))
        else:
            anchor = anchors[:, :, a, :].float() * (float(scale) / float(temperature))
            logits = torch.einsum("bnthd,bhd->bnth", k_blocks, anchor)
        logits = logits.masked_fill(~valid[None, :, :, None], float("-inf"))
        p = torch.softmax(logits, dim=2)  # [B, N, 64, H]
        k_prime[:, :, :, a] = torch.einsum("bnthd,bnth->bnhd", k_blocks, p)
        v_prime[:, :, :, a] = torch.einsum("bnthd,bnth->bnhd", v_blocks, p)
        log_p = torch.where(p > 0, torch.log(p.clamp_min(torch.finfo(p.dtype).tiny)), 0.0)
        entropy[:, :, :, a] = -(p * log_p).sum(dim=2)
    return k_prime, v_prime, entropy


def anchor_thresholds(
    q_bar: torch.Tensor,
    k_prime: torch.Tensor,
    bias: torch.Tensor,
    assign: torch.Tensor,
    *,
    scale: float,
    tau: float,
    thresh_type: str = "diag",
) -> torch.Tensor:
    """Closed-form ``mu + tau * sigma`` thresholds for anchor routing scores.

    The routing score of query block ``i`` against KV block ``j`` is
    ``s2 * q_bar_i . k'_j + b2_j`` with ``s2 = scale * log2(e)`` and ``b2`` the
    entropy in log2 units.  It is linear in the block feature
    ``phi_j = [s2 k'_j ; b2_j]`` with query feature ``psi_i = [q_bar_i ; 1]``, so
    the row mean and variance over ``j`` follow from the first and second
    moments of ``phi`` exactly as in the mean-pooling kernel.

    ``q_bar`` ``[B, N, H, D]``; ``k_prime`` ``[B, N, H, A, D]`` (use the BF16-
    rounded values the kernel will see); ``bias`` ``[B, N, H, A]`` log2 units;
    ``assign`` ``[B, N, H]``.  Returns ``[B, N, H]`` FP32.
    """

    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    log2_scale = float(scale) * math.log2(math.e)
    phi_k = k_prime.float() * log2_scale  # [B, N, H, A, D]
    bias = bias.float()
    index = assign.long()[..., None]  # [B, N, H, 1]

    phi_mean = phi_k.mean(dim=1)  # [B, H, A, D]
    bias_mean = bias.mean(dim=1)  # [B, H, A]
    phi_mean_sel = torch.gather(
        phi_mean.unsqueeze(1).expand(-1, q_bar.shape[1], -1, -1, -1),
        3,
        index[..., None].expand(-1, -1, -1, 1, phi_k.shape[-1]),
    ).squeeze(3)  # [B, N, H, D]
    bias_mean_sel = torch.gather(
        bias_mean.unsqueeze(1).expand(-1, q_bar.shape[1], -1, -1), 3, index
    ).squeeze(3)  # [B, N, H]
    mean = (q_bar * phi_mean_sel).sum(dim=-1) + bias_mean_sel

    if thresh_type == "exact":
        feat = torch.cat([phi_k, bias[..., None]], dim=-1)  # [B, N, H, A, D+1]
        second = torch.einsum("bnhai,bnhaj->bhaij", feat, feat) / feat.shape[1]
        psi = torch.cat([q_bar, q_bar.new_ones(q_bar.shape[:-1] + (1,))], dim=-1)
        raw_second = torch.einsum("bnhi,bhaij,bnhj->bnha", psi, second, psi)
        raw_second = torch.gather(raw_second, 3, index).squeeze(3)
        variance = raw_second - mean.square()
    else:
        var_k = phi_k.var(dim=1, unbiased=False)  # [B, H, A, D]
        var_b = bias.var(dim=1, unbiased=False)  # [B, H, A]
        var_k_sel = torch.gather(
            var_k.unsqueeze(1).expand(-1, q_bar.shape[1], -1, -1, -1),
            3,
            index[..., None].expand(-1, -1, -1, 1, phi_k.shape[-1]),
        ).squeeze(3)
        var_b_sel = torch.gather(
            var_b.unsqueeze(1).expand(-1, q_bar.shape[1], -1, -1), 3, index
        ).squeeze(3)
        variance = (q_bar.square() * var_k_sel).sum(dim=-1) + var_b_sel

    return mean + float(tau) * torch.sqrt(torch.clamp_min(variance, 0.0) + 1.0e-6)


def prepare_anchor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    tau: float,
    thresh_type: str = "diag",
    num_anchors: int = 1,
    temperature: float = math.inf,
    anchor_mode: str = "position",
    kmeans_iters: int = 5,
    anchors: torch.Tensor | None = None,
    assign: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build anchor-indexer tensors for the Triton reference kernel.

    Pass explicit ``anchors`` ``[B, H, A, D]`` and ``assign`` ``[B, N, H]`` to
    bypass anchor selection (used by tests and offline studies).
    """

    batch, tokens, heads, head_dim = q.shape
    blocks = _num_blocks(tokens)
    padded = _padded_blocks(blocks)

    if anchors is None or assign is None:
        anchors, assign = compute_anchors(
            q, num_anchors, mode=anchor_mode, kmeans_iters=kmeans_iters
        )
    num_anchors = anchors.shape[2]
    assign = assign.to(torch.int32)

    k_prime, v_prime, entropy = anchor_summaries(
        k, v, anchors, scale=scale, temperature=temperature
    )
    kc = k_prime.to(torch.bfloat16)
    vc = v_prime.to(torch.bfloat16)
    bias = entropy / math.log(2.0)  # log2 units

    threshold = anchor_thresholds(
        block_mean_queries(q),
        kc.float(),
        bias,
        assign,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
    )

    def _pad_blocks(x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((batch, padded) + tuple(x.shape[2:]))
        out[:, :blocks] = x
        return out

    return {
        "kc": _pad_blocks(kc.reshape(batch, blocks, heads * num_anchors, head_dim)),
        "vc": _pad_blocks(vc.reshape(batch, blocks, heads * num_anchors, head_dim)),
        "bias": _pad_blocks(bias.reshape(batch, blocks, heads * num_anchors)).contiguous(),
        "assign": assign.contiguous(),
        "threshold": threshold.contiguous(),
        "anchors": anchors,
        "num_anchors": num_anchors,
    }


__all__ = [
    "ANCHOR_MODES",
    "anchor_summaries",
    "anchor_thresholds",
    "block_mean_queries",
    "compute_anchors",
    "prepare_anchor",
]
