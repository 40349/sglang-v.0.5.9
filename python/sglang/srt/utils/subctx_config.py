"""Switches and support checks for the sub-context path.

Read once from env vars at import. Only the plain device-side radix tree with
``page_size == 1`` is supported.
"""

from __future__ import annotations

import os
from typing import Optional

# Baseline arm: chat requests carry no split and take the stock radix path.
DISABLE_SUBCONTEXT = os.environ.get("SGLANG_DISABLE_SUBCONTEXT", "") not in ("", "0")

# Insert the generated tokens into the last block's namespace at finish. Off frees them.
CACHE_SUBCONTEXT_OUTPUT = os.environ.get(
    "SGLANG_SUBCONTEXT_CACHE_OUTPUT", "1"
) not in ("", "0")

# Stage 1: rotate a displaced hit's cached K by `offset - canonical` and reuse it.
ROTATE_SUBCONTEXT = os.environ.get("SGLANG_SUBCONTEXT_ROTATE", "") not in ("", "0")

# Stage 2: end a prefill chunk at a block edge so the next block can be rotated in.
ROTATE_ACROSS_RECOMPUTE = os.environ.get(
    "SGLANG_SUBCONTEXT_ROTATE_ACROSS", ""
) not in ("", "0")

# Debug: pure-torch rotation instead of the Triton kernel.
ROTATE_NATIVE = os.environ.get("SGLANG_SUBCONTEXT_ROTATE_NATIVE", "") not in ("", "0")

# Name each block's namespace by a hash of its tokens and the request's `extra_key`
# instead of by role ("system_prompt_key", ...). Prefill still takes the stitch path.
HASH_SUBCONTEXT_KEYS = os.environ.get("SGLANG_SUBCTX_HASH_KEYS", "") not in ("", "0")

# Find cached blocks anywhere in the prompt by scanning it against the chunk index, and
# prefill only the gaps between them in one pass. Implies HASH_SUBCONTEXT_KEYS.
INDEX_SUBCONTEXTS = os.environ.get("SGLANG_SUBCTX_INDEX", "") not in ("", "0")

# Scan and report what the index would find, but serve through the stitch path.
INDEX_DRYRUN = os.environ.get("SGLANG_SUBCTX_INDEX_DRYRUN", "") not in ("", "0")

# Shortest chunk the index registers or reuses.
MIN_CHUNK_TOKENS = int(os.environ.get("SGLANG_SUBCTX_MIN_CHUNK", "64"))

# Block boundaries: "blocks" cuts at the roles (system / tools / messages); "cdc" cuts
# where the fingerprint of the next tokens is 0 mod CDC_TARGET_TOKENS, i.e. by content.
SPLIT_MODE = os.environ.get("SGLANG_SUBCTX_SPLIT", "blocks")

# "cdc" average chunk length (rounded down to a power of two), and the forced-cut length.
CDC_TARGET_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_TARGET", "256"))
CDC_MAX_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_MAX", "1024"))

# Fraction of the reused tokens to recompute, picked by how far their key moved.
# 0 recomputes nothing; 1.0 recomputes every reused token.
TOPK_RATIO = float(os.environ.get("SGLANG_SUBCTX_TOPK_RATIO", "0"))

# Layer whose fresh and cached keys are compared. At layer 0 every score is 0.
TOPK_LAYER = int(os.environ.get("SGLANG_SUBCTX_TOPK_LAYER", "1"))


def hash_subcontext_keys() -> bool:
    """Whether block namespaces are addressed by content."""
    return HASH_SUBCONTEXT_KEYS or INDEX_SUBCONTEXTS or INDEX_DRYRUN


def topk_active() -> bool:
    """Whether any reused token is recomputed."""
    return TOPK_RATIO > 0.0 and INDEX_SUBCONTEXTS


if SPLIT_MODE not in ("blocks", "cdc"):
    raise ValueError(
        f"SGLANG_SUBCTX_SPLIT={SPLIT_MODE!r}; want 'blocks' or 'cdc'"
    )

if SPLIT_MODE == "cdc" and not DISABLE_SUBCONTEXT and not hash_subcontext_keys():
    raise ValueError(
        "SGLANG_SUBCTX_SPLIT=cdc needs content-addressed namespaces: set "
        "SGLANG_SUBCTX_INDEX=1 (or SGLANG_SUBCTX_HASH_KEYS=1)"
    )

if not 0.0 <= TOPK_RATIO <= 1.0:
    raise ValueError(
        f"SGLANG_SUBCTX_TOPK_RATIO={TOPK_RATIO}; want a fraction in [0, 1]"
    )

if TOPK_RATIO > 0.0 and not INDEX_SUBCONTEXTS:
    raise ValueError(
        "SGLANG_SUBCTX_TOPK_RATIO needs the index's sparse prefill to have something "
        "to recompute into: set SGLANG_SUBCTX_INDEX=1, or leave the ratio at 0"
    )

if TOPK_LAYER < 0:
    raise ValueError(f"SGLANG_SUBCTX_TOPK_LAYER={TOPK_LAYER}; want a layer index")

if TOPK_LAYER == 0 and 0.0 < TOPK_RATIO < 1.0:
    raise ValueError(
        "SGLANG_SUBCTX_TOPK_LAYER=0 would score every token identically: layer 0's "
        "key depends only on the token and its position, so every deviation is zero. "
        "Use layer 1 or later. (Layer 0 is allowed at ratio 1.0, where the score "
        "decides nothing.)"
    )


def unsupported_reason(tree_cache) -> Optional[str]:
    """Return why ``tree_cache`` cannot serve sub-contexts, or None if it can.

    Explains a False from ``supports_sub_contexts``.
    """
    from sglang.srt.mem_cache.radix_cache import RadixCache

    if tree_cache.supports_sub_contexts():
        return None
    if not isinstance(tree_cache, RadixCache):
        return (
            f"the {type(tree_cache).__name__} prefix cache does not implement the "
            "per-namespace insert path (--disable-radix-cache, hybrid SWA/mamba "
            "pools and the experimental C++ radix tree all select such a cache)"
        )
    if type(tree_cache) is not RadixCache:
        return (
            f"{type(tree_cache).__name__} extends the radix cache with a tier the "
            "per-namespace insert path does not maintain (e.g. the host tier of "
            "--enable-hierarchical-cache)"
        )
    if tree_cache.disable:
        return "prefix caching is disabled (--disable-radix-cache)"
    if tree_cache.page_size != 1:
        return (
            f"page_size is {tree_cache.page_size}, the per-namespace insert path "
            "needs 1"
        )
    if tree_cache.is_eagle:
        return "EAGLE speculative decoding rewrites the radix keys into bigrams"
    return "the prefix cache reports no sub-context support"


def sparse_prefill_unsupported_reason(model_runner) -> Optional[str]:
    """Return why a prefill cannot place reused blocks freely, or None if it can.

    Each case below would attend to the wrong keys without raising.
    """
    backend = model_runner.server_args.get_attention_backends()[0]
    if backend != "triton":
        return (
            f"the {backend} prefill attention backend applies causality by index; "
            "only triton takes the per-position mask a scattered prefill needs "
            "(--attention-backend triton)"
        )
    if model_runner.sliding_window_size is not None and model_runner.sliding_window_size > 0:
        return (
            "sliding-window attention derives each key's absolute position from the "
            "prefix length, which a prefill with holes in it does not have"
        )
    if model_runner.server_args.enable_piecewise_cuda_graph:
        return (
            "piecewise CUDA graphs capture a prefill's shapes, and the mask a "
            "scattered prefill builds is sized per request "
            "(--disable-piecewise-cuda-graph)"
        )
    return None


def rotation_unsupported_reason(model_runner, tree_cache) -> Optional[str]:
    """Return why cached K cannot be re-rotated here, or None if it can.

    Structural checks first; then ``rope_delta_composable_reason`` tests the identity
    on the model's own RoPE module.
    """
    from sglang.srt.layers.rotary_embedding import (
        LinearScalingRotaryEmbedding,
        MRotaryEmbedding,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.rotate_kv import (
        find_rotary_embedding,
        rope_delta_composable_reason,
    )

    sub_context = unsupported_reason(tree_cache)
    if sub_context is not None:
        return f"sub-contexts are unsupported ({sub_context})"

    rotary, why = find_rotary_embedding(model_runner.model)
    if rotary is None:
        return why

    if isinstance(rotary, MRotaryEmbedding):
        return (
            "mrope addresses a position with a 3-vector (text/height/width); the "
            "sub-context offsets are scalars, so there is no one delta to rotate by"
        )
    if isinstance(rotary, LinearScalingRotaryEmbedding):
        return (
            "linear scaling concatenates one cos_sin_cache per scaling factor, so a "
            "row index is a position plus a per-request offset and a delta can "
            "cross into a different cache"
        )

    if not rotary.is_neox_style:
        return "the model uses GPT-J interleaved RoPE, not the neox pairing"
    if rotary.rotary_dim != rotary.head_size:
        return (
            f"partial rotary ({rotary.rotary_dim} of {rotary.head_size} dims); the "
            "rotation kernel assumes the whole head rotates"
        )

    pool = model_runner.token_to_kv_pool
    if not isinstance(pool, MHATokenToKVPool):
        return f"the {type(pool).__name__} KV pool has no plain per-layer K buffer"
    if pool.store_dtype != pool.dtype:
        return (
            f"the KV cache is stored as {pool.store_dtype} for a {pool.dtype} cache "
            "(quantized); rotating would need a dequant/requant round trip"
        )
    if pool.head_dim != rotary.head_size:
        return f"pool head_dim {pool.head_dim} != rope head_size {rotary.head_size}"

    # Last: runs the model's RoPE.
    return rope_delta_composable_reason(rotary)


def topk_unsupported_reason(model_runner) -> Optional[str]:
    """Return why reused tokens cannot be selectively recomputed, or None if they can.

    The probe runs the whole prompt through layers ``[0, TOPK_LAYER]``, then the token
    dimension is cut down to the selected rows. Each case below either cannot run a
    layer range, splits the tokens across ranks, or reads a token count the cut changes.
    """
    from sglang.srt.layers.dp_attention import get_attention_tp_size

    args = model_runner.server_args

    if not hasattr(model_runner.model, "forward_split_prefill"):
        return (
            f"{type(model_runner.model).__name__} does not implement "
            "forward_split_prefill, which is where the token set is cut down between "
            "the scored layers and the rest"
        )

    num_layers = model_runner.model_config.num_hidden_layers
    if TOPK_LAYER >= num_layers - 1:
        return (
            f"SGLANG_SUBCTX_TOPK_LAYER={TOPK_LAYER} leaves no layer after it to "
            f"recompute ({num_layers} layers)"
        )

    if args.pp_size > 1:
        return (
            "pipeline parallelism may put the scored layer on another rank, and a "
            "layer's id is then offset from its index in this rank's stack"
        )

    # Each TP rank scores only its own heads, so ranks would select different tokens.
    if get_attention_tp_size() > 1:
        return (
            "the deviation score sums over the head dimension, which tensor "
            "parallelism splits, so each rank would select a different set of tokens "
            "and the ranks would silently disagree about which token each row is "
            "(--tp-size 1)"
        )

    # These scatter the tokens themselves across ranks.
    if args.enable_dp_attention:
        return (
            "data-parallel attention gives each rank its own slice of the tokens, so "
            "a globally chosen row index addresses a different token on each rank "
            "(--disable-dp-attention)"
        )
    if args.moe_dense_tp_size == 1:
        return (
            "--moe-dense-tp-size 1 makes the layer boundary scattered, so the hidden "
            "states to be cut down are a per-rank slice rather than every token"
        )

    if args.enable_two_batch_overlap:
        return (
            "two-batch overlap splits a batch and re-enters the layer stack per half, "
            "which the probe's single pass over the whole prompt does not survive"
        )
    if args.speculative_algorithm is not None:
        return (
            f"speculative decoding ({args.speculative_algorithm}) captures hidden "
            "states at full length, which the cut invalidates"
        )
    if args.enable_torch_compile:
        return (
            "torch.compile specialises on shapes, and this pass runs two different "
            "token counts through the same layer stack"
        )
    if args.enable_return_hidden_states:
        return (
            "the returned hidden states would cover only the recomputed tokens, not "
            "the prompt"
        )
    if args.enable_lora:
        return "the LoRA backend sizes its batch from extend_num_tokens"

    return None
