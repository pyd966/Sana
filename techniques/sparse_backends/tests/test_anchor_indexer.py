"""CPU tests for the first-order Taylor (anchor) block indexer.

GPU-only checks against the Triton kernel live at the bottom and are skipped
without CUDA.
"""

from __future__ import annotations

import math

import pytest
import torch

from sol_attn.anchor import (
    anchor_summaries,
    anchor_thresholds,
    block_mean_queries,
    compute_anchors,
    prepare_anchor,
)
from sol_attn.mps.preprocess import _routing_thresholds
from sol_attn.reference import (
    block_log2_partition,
    dense_reference,
    route_mask_reference,
    sol_attn_reference,
)

D = 128
SCALE = D**-0.5
LOG2E = math.log2(math.e)


def _qkv(shape, seed, scale=0.5, dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(shape, generator=gen) * scale).to(dtype) for _ in range(3))


def _legacy_summaries(k, v):
    """Mean-pooled key and summed value per 64-token block, BF16, [B, N, H, D]."""

    batch, tokens, heads, head_dim = k.shape
    blocks = (tokens + 63) // 64
    pad = blocks * 64 - tokens
    kf = torch.nn.functional.pad(k.float(), (0, 0, 0, 0, 0, pad)).view(batch, blocks, 64, heads, head_dim)
    vf = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, pad)).view(batch, blocks, 64, heads, head_dim)
    lengths = torch.full((blocks,), 64.0)
    lengths[-1] = tokens - (blocks - 1) * 64
    kc = kf.sum(dim=2) / lengths[None, :, None, None]
    vc = vf.sum(dim=2)
    return kc.to(torch.bfloat16), vc.to(torch.bfloat16)


# --------------------------------------------------------------------------
# Taylor-expansion properties
# --------------------------------------------------------------------------


def test_tangent_touches_log_partition_at_the_anchor():
    """At q == anchor the first-order expansion is exact (tangent point)."""

    torch.manual_seed(0)
    k = torch.randn(1, 192, 2, D) * 0.6
    v = torch.randn_like(k)
    anchors = torch.randn(1, 2, 1, D) * 0.6  # [B, H, A, D]

    k_prime, _, entropy = anchor_summaries(k, v, anchors, scale=SCALE, temperature=1.0)
    q = anchors[:, :, 0, :][:, None].expand(1, 192, 2, D).contiguous()  # every token = anchor
    truth = block_log2_partition(q, k, scale=SCALE)  # [B, H, T, N] log2
    approx = torch.einsum("bthd,bnhd->bhtn", q, k_prime[:, :, :, 0]) * SCALE * LOG2E
    approx = approx + (entropy[:, :, :, 0] / math.log(2.0)).permute(0, 2, 1)[:, :, None, :]
    torch.testing.assert_close(approx, truth, rtol=1e-4, atol=1e-4)


def test_tangent_is_a_lower_bound_everywhere():
    """log-sum-exp is convex, so every tangent sits below it."""

    torch.manual_seed(1)
    k = torch.randn(1, 256, 1, D) * 0.6
    v = torch.randn_like(k)
    q = torch.randn(1, 256, 1, D) * 0.6
    for temperature in (1.0, 2.0, math.inf):
        anchors = torch.randn(1, 1, 1, D) * 0.6
        k_prime, _, entropy = anchor_summaries(k, v, anchors, scale=SCALE, temperature=temperature)
        truth = block_log2_partition(q, k, scale=SCALE)
        approx = torch.einsum("bthd,bnhd->bhtn", q, k_prime[:, :, :, 0]) * SCALE * LOG2E
        approx = approx + (entropy[:, :, :, 0] / math.log(2.0)).permute(0, 2, 1)[:, :, None, :]
        assert bool((approx <= truth + 1e-4).all()), f"tangent exceeded log Z at T={temperature}"


def test_temperature_equals_anchor_shrink():
    """softmax(s (q'/T).k) is the softmax at anchor q'/T."""

    torch.manual_seed(2)
    k = torch.randn(1, 128, 1, D)
    v = torch.randn_like(k)
    anchors = torch.randn(1, 1, 1, D)
    hot = anchor_summaries(k, v, anchors, scale=SCALE, temperature=3.0)
    shrunk = anchor_summaries(k, v, anchors / 3.0, scale=SCALE, temperature=1.0)
    for a, b in zip(hot, shrunk):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_infinite_temperature_recovers_mean_pooling():
    torch.manual_seed(3)
    k, v, _ = _qkv((1, 200, 2, D), seed=3)  # tail block of 8 tokens
    anchors = torch.randn(1, 2, 1, D)
    k_prime, v_prime, entropy = anchor_summaries(k, v, anchors, scale=SCALE, temperature=math.inf)
    kc, vc = _legacy_summaries(k, v)
    lengths = torch.tensor([64.0, 64.0, 64.0, 8.0])
    torch.testing.assert_close(k_prime[:, :, :, 0], kc.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        v_prime[:, :, :, 0], vc.float() / lengths[None, :, None, None], rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(entropy[:, :, :, 0], torch.log(lengths)[None, :, None].expand(1, 4, 2))


# --------------------------------------------------------------------------
# Anchor selection and tensor layout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["position", "kmeans"])
@pytest.mark.parametrize("num_anchors", [1, 3, 8])
def test_compute_anchors_shapes_and_assignment(mode, num_anchors):
    q, _, _ = _qkv((2, 5 * 64 + 17, 3, D), seed=4)
    anchors, assign = compute_anchors(q, num_anchors, mode=mode)
    blocks = 6
    assert anchors.shape == (2, 3, num_anchors, D)
    assert assign.shape == (2, blocks, 3)
    assert assign.dtype == torch.int32
    assert int(assign.min()) >= 0 and int(assign.max()) < min(num_anchors, blocks)
    if mode == "position":
        # Contiguous ranges, monotone in block index.
        assert bool((assign[:, 1:] >= assign[:, :-1]).all())
    if num_anchors == 1:
        # A single anchor is the mean of the block-mean queries (blocks weigh equally).
        torch.testing.assert_close(anchors[:, :, 0], block_mean_queries(q).mean(dim=1), rtol=1e-3, atol=1e-3)


def test_prepare_anchor_layout_and_padding():
    q, k, v = _qkv((1, 70 * 64 + 5, 2, D), seed=5)  # 71 blocks -> padded to 128
    out = prepare_anchor(q, k, v, scale=SCALE, tau=1.0, num_anchors=4, temperature=2.0)
    assert out["kc"].shape == (1, 128, 8, D) and out["kc"].dtype == torch.bfloat16
    assert out["vc"].shape == (1, 128, 8, D)
    assert out["bias"].shape == (1, 128, 8) and out["bias"].dtype == torch.float32
    assert out["assign"].shape == (1, 71, 2)
    assert out["threshold"].shape == (1, 71, 2)
    assert bool((out["bias"][:, 71:] == 0).all()) and bool((out["kc"][:, 71:] == 0).all())
    # Entropy is bounded by log2 of the block length.
    assert float(out["bias"][:, :70].max()) <= math.log2(64) + 1e-4
    assert float(out["bias"][:, 70].max()) <= math.log2(5) + 1e-4


def test_mean_pooling_threshold_matches_legacy_up_to_constant_bias():
    """A=1, T=inf: scores shift by log2(64) everywhere, so does the threshold."""

    for thresh_type in ("diag", "exact"):
        q, k, v = _qkv((2, 8 * 64, 3, D), seed=6)
        out = prepare_anchor(q, k, v, scale=SCALE, tau=1.3, thresh_type=thresh_type, num_anchors=1)
        kc, _ = _legacy_summaries(k, v)
        q_bar = block_mean_queries(q)
        legacy = _routing_thresholds(
            q_bar.permute(0, 2, 1, 3),  # [B, H, N, D]
            kc.permute(0, 2, 1, 3).float(),
            SCALE,
            1.3,
            thresh_type,
        ).permute(0, 2, 1)
        torch.testing.assert_close(out["threshold"], legacy + math.log2(64), rtol=1e-3, atol=1e-3)


def test_anchor_thresholds_match_materialized_statistics():
    """Closed-form mu/sigma equals the statistics of the materialized score row."""

    torch.manual_seed(7)
    q, k, v = _qkv((1, 12 * 64, 2, D), seed=7)
    anchors, assign = compute_anchors(q, 3, mode="kmeans")
    k_prime, _, entropy = anchor_summaries(k, v, anchors, scale=SCALE, temperature=1.5)
    kc = k_prime.to(torch.bfloat16).float()
    bias = entropy / math.log(2.0)
    q_bar = block_mean_queries(q)
    scores = torch.einsum("bihd,bjhad->bihaj", q_bar, kc) * SCALE * LOG2E
    scores = scores + bias.permute(0, 2, 3, 1)[:, None]  # [B, N(i), H, A, N(j)]
    sel = torch.gather(scores, 3, assign.long()[..., None, None].expand(-1, -1, -1, 1, 12)).squeeze(3)
    mu = sel.mean(dim=-1)
    var = sel.var(dim=-1, correction=0)
    expected = mu + 0.9 * torch.sqrt(var + 1e-6)
    actual = anchor_thresholds(q_bar, kc, bias, assign, scale=SCALE, tau=0.9, thresh_type="exact")
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# Reference forward
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_anchor", [False, True])
def test_reference_all_exact_matches_dense(use_anchor):
    q, k, v = _qkv((1, 3 * 64 + 9, 2, D), seed=8)
    blocks = 4
    if use_anchor:
        out = prepare_anchor(q, k, v, scale=SCALE, tau=1.0, num_anchors=2, temperature=1.0)
        kwargs = dict(kc=out["kc"], vc=out["vc"], bias=out["bias"], assign=out["assign"], threshold=out["threshold"])
    else:
        kc, vc = _legacy_summaries(k, v)
        kwargs = dict(kc=kc, vc=vc, threshold=torch.zeros(1, blocks, 2))
    actual = sol_attn_reference(q, k, v, scale=SCALE, sink_blocks=(0, blocks), **kwargs)
    expected = dense_reference(q, k, v, scale=SCALE)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_anchor_reference_with_mean_pooling_limit_matches_legacy_reference():
    q, k, v = _qkv((1, 6 * 64 + 20, 2, D), seed=9, scale=0.3)
    out = prepare_anchor(q, k, v, scale=SCALE, tau=0.8, num_anchors=1, temperature=math.inf)
    kc, vc = _legacy_summaries(k, v)
    legacy_thr = out["threshold"] - math.log2(64)  # constant bias removed (tail block differs slightly)
    anchor_out, anchor_route = sol_attn_reference(
        q, k, v, scale=SCALE, kc=out["kc"], vc=out["vc"], bias=out["bias"], assign=out["assign"],
        threshold=out["threshold"], return_route=True,
    )
    legacy_out, legacy_route = sol_attn_reference(
        q, k, v, scale=SCALE, kc=kc, vc=vc, threshold=legacy_thr, route=anchor_route, return_route=True,
    )
    assert bool((~anchor_route).any()), "test must exercise approximate blocks"
    torch.testing.assert_close(anchor_out.float(), legacy_out.float(), rtol=3e-2, atol=3e-2)


def test_anchor_reference_tracks_dense_with_sparse_routing():
    q, k, v = _qkv((1, 16 * 64, 1, D), seed=10, scale=0.4)
    out = prepare_anchor(q, k, v, scale=SCALE, tau=1.0, num_anchors=4, temperature=1.0)
    actual, route = sol_attn_reference(
        q, k, v, scale=SCALE, kc=out["kc"], vc=out["vc"], bias=out["bias"], assign=out["assign"],
        threshold=out["threshold"], return_route=True,
    )
    expected = dense_reference(q, k, v, scale=SCALE).float()
    density = route.float().mean().item()
    assert 0.1 < density < 0.9, f"routing density {density:.2f} not informative"
    rel = (actual.float() - expected).norm() / expected.norm()
    assert rel < 0.15, f"relative error {rel:.3f} too large"


# --------------------------------------------------------------------------
# Routing quality on a needle construction
# --------------------------------------------------------------------------


def _needle_problem(seed=11, regions=4, blocks_per_region=4, heads=1):
    """Queries live in `regions` directions; each KV block hides one needle key
    aligned with a random region, buried among random keys."""

    gen = torch.Generator().manual_seed(seed)
    blocks = regions * blocks_per_region
    directions = torch.nn.functional.normalize(torch.randn(regions, D, generator=gen), dim=-1) * math.sqrt(D)
    region_of_query_block = torch.arange(blocks) // blocks_per_region
    q = directions[region_of_query_block][:, None].expand(blocks, 64, D) * 0.9
    q = q + torch.randn(blocks, 64, D, generator=gen) * 0.3
    k = torch.randn(blocks, 64, D, generator=gen) * 0.9
    needle_region = torch.randint(0, regions, (blocks,), generator=gen)
    k[:, 0] = directions[needle_region] * 1.2
    q = q.reshape(1, blocks * 64, 1, D).expand(1, -1, heads, -1).contiguous().to(torch.bfloat16)
    k = k.reshape(1, blocks * 64, 1, D).expand(1, -1, heads, -1).contiguous().to(torch.bfloat16)
    return q, k, torch.randn_like(k)


def _recall(proxy, truth, budget):
    top_true = truth.topk(budget, dim=-1).indices
    top_proxy = proxy.topk(budget, dim=-1).indices
    hits = (top_true[..., :, None] == top_proxy[..., None, :]).any(dim=-1).float().sum(dim=-1)
    return (hits / budget).mean().item()


def test_anchor_indexer_beats_mean_pooling_on_needles():
    q, k, v = _needle_problem()
    blocks = q.shape[1] // 64
    truth = block_log2_partition(q, k, scale=SCALE).view(1, 1, blocks, 64, blocks).mean(dim=3)[0, 0]  # [N, N]

    def proxy_scores(num_anchors, temperature):
        out = prepare_anchor(q, k, v, scale=SCALE, tau=1.0, num_anchors=num_anchors, temperature=temperature)
        q_bar = block_mean_queries(q)[0, :, 0]  # [N, D]
        kc = out["kc"][0, :blocks].float()  # [N, A, D]
        bias = out["bias"][0, :blocks]  # [N, A]
        a = out["assign"][0, :, 0].long()
        scores = torch.einsum("id,jad->iaj", q_bar, kc) * SCALE * LOG2E + bias.T[None]
        return scores[torch.arange(blocks), a]  # [N(i), N(j)]

    budget = blocks // 4
    recall_mean_pool = _recall(proxy_scores(1, math.inf), truth, budget)
    recall_anchor = _recall(proxy_scores(4, 1.0), truth, budget)
    assert recall_anchor > recall_mean_pool + 0.1, (recall_mean_pool, recall_anchor)


# --------------------------------------------------------------------------
# Triton kernel (GPU only)
# --------------------------------------------------------------------------

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU with Triton")
@pytest.mark.parametrize("tokens", [4 * 64, 9 * 64 + 13])
@pytest.mark.parametrize("num_anchors,temperature", [(1, math.inf), (1, 1.0), (4, 2.0)])
def test_triton_anchor_kernel_matches_reference(tokens, num_anchors, temperature):
    from sol_attn.triton_ref import sol_attn as triton_sol_attn

    q, k, v = _qkv((1, tokens, 2, D), seed=12, scale=0.4)
    q, k, v = (x.cuda() for x in (q, k, v))
    out = prepare_anchor(q, k, v, scale=SCALE, tau=1.0, num_anchors=num_anchors, temperature=temperature)
    expected = sol_attn_reference(
        q, k, v, scale=SCALE, kc=out["kc"], vc=out["vc"], bias=out["bias"], assign=out["assign"],
        threshold=out["threshold"],
    )
    actual = triton_sol_attn(
        q, k, v, tau=1.0, num_anchors=num_anchors, anchor_temperature=temperature
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU with Triton")
def test_triton_anchor_mean_pooling_limit_matches_legacy_kernel():
    from sol_attn.triton_ref import sol_attn as triton_sol_attn

    q, k, v = _qkv((1, 12 * 64, 2, D), seed=13, scale=0.4)
    q, k, v = (x.cuda() for x in (q, k, v))
    legacy = triton_sol_attn(q, k, v, tau=1.0)
    anchored = triton_sol_attn(q, k, v, tau=1.0, num_anchors=1, anchor_temperature=math.inf)
    torch.testing.assert_close(anchored.float(), legacy.float(), rtol=2e-2, atol=2e-2)
