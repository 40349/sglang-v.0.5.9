"""Switches and support rules for the sub-context path.

Env-var driven (flipped between runs, not per request), and only supported on the
plain device-side radix tree with ``page_size == 1``. Read by ``server_args``, the chat
serving path and the radix cache.
"""

from __future__ import annotations

import os
from typing import Optional

# A/B switch: when set, chat requests carry no split and take the stock
# single-namespace radix path. The baseline arm, same binary and weights.
DISABLE_SUBCONTEXT = os.environ.get("SGLANG_DISABLE_SUBCONTEXT", "") not in ("", "0")

# Whether generated tokens are written back into the last block's namespace, so the
# next agent turn hits them. Off frees the tail at finish.
CACHE_SUBCONTEXT_OUTPUT = os.environ.get(
    "SGLANG_SUBCONTEXT_CACHE_OUTPUT", "1"
) not in ("", "0")

# Stage 1: re-rotate a block's cached K by `offset - canonical`, so a hit computed at
# another position is reused instead of dropped. Changes which KV the model sees.
ROTATE_SUBCONTEXT = os.environ.get("SGLANG_SUBCONTEXT_ROTATE", "") not in ("", "0")

# Stage 2: force a chunk boundary at a block edge so a block *after* a partially
# recomputed one can still be rotated in. Implies ROTATE_SUBCONTEXT.
ROTATE_ACROSS_RECOMPUTE = os.environ.get(
    "SGLANG_SUBCONTEXT_ROTATE_ACROSS", ""
) not in ("", "0")

# Debug: take the pure-torch rotation path instead of the Triton kernel.
ROTATE_NATIVE = os.environ.get("SGLANG_SUBCONTEXT_ROTATE_NATIVE", "") not in ("", "0")

# Address each block by a hash of its own tokens instead of the role it played
# ("system_prompt_key" and friends). A namespace then holds one content rather than one
# role, and `extra_key` -- cache_salt with lora_id concatenated -- reaches the block
# keys, which the role names dropped.
#
# Separate from INDEX: it changes which hits the stitch path can use without changing
# how a prompt is prefilled.
HASH_SUBCONTEXT_KEYS = os.environ.get("SGLANG_SUBCTX_HASH_KEYS", "") not in ("", "0")

# Find blocks by scanning the query against an index of every registered chunk, rather
# than looking up fixed roles at fixed offsets, and prefill the result in one pass with
# the reused blocks wherever they land. Implies HASH_SUBCONTEXT_KEYS.
INDEX_SUBCONTEXTS = os.environ.get("SGLANG_SUBCTX_INDEX", "") not in ("", "0")

# Scan and report, then take the stock path anyway. Reports the tokens the index finds
# that the contiguity rule cannot reach.
INDEX_DRYRUN = os.environ.get("SGLANG_SUBCTX_INDEX_DRYRUN", "") not in ("", "0")

# Shortest run of tokens the index will take.
MIN_CHUNK_TOKENS = int(os.environ.get("SGLANG_SUBCTX_MIN_CHUNK", "64"))

# Where block boundaries come from. "blocks" uses the roles the prompt was built from
# (system / tools / messages). "cdc" cuts where the token window at a position hashes to
# zero mod CDC_TARGET_TOKENS, which puts the same boundaries in a run of tokens wherever
# that run appears.
SPLIT_MODE = os.environ.get("SGLANG_SUBCTX_SPLIT", "blocks")

# Average chunk length "cdc" aims for, rounded down to a power of two, and the length at
# which it forces a boundary the content did not produce.
CDC_TARGET_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_TARGET", "256"))
CDC_MAX_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_MAX", "1024"))

# Fraction of the tokens a prompt reuses that get recomputed anyway, chosen by how far
# their key has moved. Rotation fixes a block's position; it does not fix the context it
# was computed under, and this is what buys that back.
#
# A dial, not a switch. 0.0 recomputes nothing and is the arm as it stands; 1.0
# recomputes every reused token, which is a correctness configuration rather than a
# useful one.
TOPK_RATIO = float(os.environ.get("SGLANG_SUBCTX_TOPK_RATIO", "0"))

# The layer whose keys are compared. Layer 0's key is a function of the token and its
# position alone, so a rotated cache row already equals what a full prefill would
# produce there and the deviation is identically zero -- the comparison needs at least
# one layer of attention to have mixed the new context in.
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
        "key is a function of the token and its position alone, so a rotated cache "
        "row already equals what a full prefill computes there and every deviation "
        "is zero. Use layer 1 or later. (Layer 0 is allowed at ratio 1.0, where the "
        "score decides nothing -- that is the end-to-end correctness configuration.)"
    )


def unsupported_reason(tree_cache) -> Optional[str]:
    """Return why ``tree_cache`` cannot serve sub-contexts, or None if it can.

    Read (``Req._stitch_sub_contexts``) and write
    (``RadixCache._cache_unfinished_sub_contexts``) must agree: matching per
    namespace while inserting into the default one hits 0% forever. Both gate on
    ``supports_sub_contexts``; this turns a False into the actual reason.
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

    Refusing loudly, as everywhere else on this path. Every one of these produces a
    *quietly* wrong answer rather than an error: the backends below decide what a query
    may attend to by comparing indices, and a sparse prefill's queries are not at the
    indices their positions imply, so they would silently attend to the wrong keys --
    including keys ahead of themselves.
    """
    backend = model_runner.server_args.attention_backend
    if backend not in (None, "triton"):
        return (
            f"the {backend} attention backend applies causality by index; only triton "
            "takes the explicit per-position mask a scattered prefill needs "
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

    Rotating by a delta is only valid when the cache row for a position is a rotation
    through an angle linear in that position. Which RoPEs have that property is
    *measured* here (``rope_delta_composable_reason`` runs the identity on the model's
    own module at startup) rather than listed by class name, so a scaled RoPE that does
    compose -- llama3's remap, dynamic NTK, YaRN once its ``mscale`` is divided out --
    is admitted on evidence instead of being refused on a guess.

    Two exclusions still come first, because they are invisible to a measurement that
    can only feed the model scalar positions from one cache.
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
            "linear scaling concatenates one cos_sin_cache per LoRA scaling factor "
            "(rotary_embedding.py:471-504), so a row index is a position plus a "
            "per-request offset and a delta can cross into a different cache"
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

    # Last, because it runs the model's own RoPE: everything above is a cheap structural
    # veto, and none of it should be reached through a forward pass.
    return rope_delta_composable_reason(rotary)


def topk_unsupported_reason(model_runner) -> Optional[str]:
    """Return why reused tokens cannot be selectively recomputed, or None if they can.

    Scoring a token needs a *freshly computed* key for a position the reuse path never
    recomputes, so the probe runs the whole prompt through the first layers and the
    token dimension is then cut down to what was selected. Everything refused here
    either moves that cut (the model must be able to run a layer range, and the layer
    boundary must hold every token on every rank) or reads a token count the cut has
    just changed.

    All of it is quiet when wrong. Nothing below raises on its own.
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

    # The score sums over the head dimension, which is the dimension TP splits, so each
    # rank scores only its own heads and `topk` picks a *different* token set per rank.
    # The shapes still agree, so nothing fails -- the later all-reduces just mix tokens
    # that are not the same tokens. Reconstructing the global score is one SUM
    # all-reduce over the attention TP group; until that is written and tested, refuse.
    if get_attention_tp_size() > 1:
        return (
            "the deviation score sums over the head dimension, which tensor "
            "parallelism splits, so each rank would select a different set of tokens "
            "and the ranks would silently disagree about which token each row is "
            "(--tp-size 1)"
        )

    # Unlike TP, these do not become supportable with a reduction: they scatter the
    # tokens themselves across ranks, so a row index chosen from the global score does
    # not address the same token on every rank.
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
