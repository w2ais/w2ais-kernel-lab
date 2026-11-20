"""Paged KV-cache decode attention; original baseline supplied by the user."""
import torch
import triton
import triton.language as tl


@triton.jit
def flashattn_with_kvcache(
    Q, K_cache, V_cache, block_table, context_lens, output, LSE,
    num_heads: tl.constexpr, head_dim: tl.constexpr,
    block_size: tl.constexpr, max_blocks_per_seq: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dims = tl.arange(0, BLOCK_D)
    token_lanes = tl.arange(0, BLOCK_N)
    query_base = (batch_idx.to(tl.int64) * num_heads + head_idx) * head_dim
    query = tl.load(Q + query_base + dims, dims < head_dim, other=0).to(tl.float32)
    context_len = tl.load(context_lens + batch_idx)
    table = block_table + batch_idx.to(tl.int64) * max_blocks_per_seq
    # Keep scores and running maxima in base-2 units: exp(x) = exp2(x * log2(e)).
    LOG2_E: tl.constexpr = 1.4426950408889634
    LN_2: tl.constexpr = 0.6931471805599453
    scale: tl.constexpr = head_dim ** -0.5 * LOG2_E

    running_max = tl.full((), float("-inf"), tl.float32)
    normalizer = tl.full((), 0, tl.float32)
    accumulator = tl.full((BLOCK_D,), 0, tl.float32)

    for start in range(0, context_len, BLOCK_N):
        tokens = start + token_lanes
        valid = tokens < context_len
        pages = tl.load(table + tokens // block_size, valid, other=0)
        slots = tokens % block_size
        cache_base = (
            ((pages.to(tl.int64) * block_size + slots) * num_heads + head_idx)
            * head_dim
        )
        offsets = cache_base[:, None] + dims[None, :]
        mask = valid[:, None] & (dims[None, :] < head_dim)

        keys = tl.load(K_cache + offsets, mask, other=0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        # Rescale earlier contributions to the maximum over all tiles so far.
        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        rescale = tl.exp2(running_max - new_max)
        weights = tl.exp2(scores - new_max)
        values = tl.load(V_cache + offsets, mask, other=0).to(tl.float32)
        accumulator = accumulator * rescale + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * rescale + tl.sum(weights, axis=0)
        running_max = new_max

    denominator = tl.where(context_len > 0, normalizer, 1.0)
    tl.store(output + query_base + dims, accumulator / denominator, dims < head_dim)
    if STORE_LSE:
        # Convert base-2 LSE back to natural-log units for the public output.
        # Empty contexts have running_max=-inf and denominator=1, yielding -inf.
        lse = (running_max + tl.log2(denominator)) * LN_2
        tl.store(LSE + batch_idx.to(tl.int64) * num_heads + head_idx, lse)


def solve(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    lse: torch.Tensor | None = None,
):
    """Run decode attention over contiguous GPU tensors.

    Q and output use [batch_size, num_heads, head_dim].
    Both caches use [num_pages, block_size, num_heads, head_dim].
    The block table uses [batch_size, max_blocks_per_seq].
    Optional lse is a contiguous float32 output of shape [batch_size, num_heads].
    It receives natural-log logsumexp of the scaled attention scores; empty
    contexts receive -inf.
    """
    if batch_size == 0 or num_heads == 0:
        return

    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    BLOCK_N = 32
    flashattn_with_kvcache[(batch_size, num_heads)](
        Q, K_cache, V_cache, block_table, context_lens, output, lse,
        num_heads, head_dim, block_size, max_blocks_per_seq, BLOCK_D, BLOCK_N, lse is not None,
        num_warps=4,
    )
