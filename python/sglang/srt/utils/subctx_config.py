"""Switches and support rules for the sub-context path.

Env-var driven (flipped between runs, not per request), and only supported on the
plain device-side radix tree with ``page_size == 1``. Lives here because
``server_args``, the chat serving path and the radix cache all need it and must
not import each other.
"""

from __future__ import annotations

import os
from typing import Optional

# A/B switch: when set, chat requests carry no split and take the stock
# single-namespace radix path -- the baseline arm, same binary and weights.
DISABLE_SUBCONTEXT = os.environ.get("SGLANG_DISABLE_SUBCONTEXT", "") not in ("", "0")

# Whether generated tokens are written back into the last block's namespace, so the
# next agent turn hits them instead of re-prefilling. Off frees the tail at finish.
CACHE_SUBCONTEXT_OUTPUT = os.environ.get(
    "SGLANG_SUBCONTEXT_CACHE_OUTPUT", "1"
) not in ("", "0")

# Stage 1: re-rotate a block's cached K by `offset - canonical` so a hit computed at
# another position becomes reusable instead of being dropped. Off by default -- it
# changes which KV the model sees, so the plain ON arm has to stay comparable to the
# numbers measured before it existed.
ROTATE_SUBCONTEXT = os.environ.get("SGLANG_SUBCONTEXT_ROTATE", "") not in ("", "0")

# Stage 2: force a chunk boundary at a block edge so a block *after* a partially
# recomputed one can still be rotated in. Implies ROTATE_SUBCONTEXT.
ROTATE_ACROSS_RECOMPUTE = os.environ.get(
    "SGLANG_SUBCONTEXT_ROTATE_ACROSS", ""
) not in ("", "0")

# Debug: take the pure-torch rotation path instead of the Triton kernel.
ROTATE_NATIVE = os.environ.get("SGLANG_SUBCONTEXT_ROTATE_NATIVE", "") not in ("", "0")

# Address each block by a hash of its own tokens instead of the role it played
# ("system_prompt_key" and friends). Two consequences, and the second is the point:
# a namespace now holds one content, so the first request to insert one no longer
# freezes the position every later request must match; and `extra_key` -- cache_salt
# with lora_id concatenated -- finally reaches the block keys, which the role names
# dropped, so one adapter's KV stops being served to another's request.
#
# Separate from INDEX because it is separately measurable: it changes which hits the
# existing stitch path can use without changing how a prompt is prefilled.
HASH_SUBCONTEXT_KEYS = os.environ.get("SGLANG_SUBCTX_HASH_KEYS", "") not in ("", "0")

# Find blocks by scanning the query against an index of every registered chunk, rather
# than looking up fixed roles at fixed offsets, and prefill the result in one pass with
# the reused blocks wherever they land. Implies HASH_SUBCONTEXT_KEYS: a scan has
# nothing to look up unless chunks are addressed by content.
INDEX_SUBCONTEXTS = os.environ.get("SGLANG_SUBCTX_INDEX", "") not in ("", "0")

# Scan and report, then take the stock path anyway. Answers "how much could this win"
# before any of it is wired to the forward pass -- the tokens the index finds that the
# contiguity rule cannot currently reach are the ceiling for the rest of the work.
INDEX_DRYRUN = os.environ.get("SGLANG_SUBCTX_INDEX_DRYRUN", "") not in ("", "0")

# Shortest run of tokens the index will take. Read here so the switch sits with the
# others; the default lives with the code that has to justify it.
MIN_CHUNK_TOKENS = int(os.environ.get("SGLANG_SUBCTX_MIN_CHUNK", "64"))

# Where block boundaries come from.
#
#   "blocks"  the roles the prompt was built from -- system / tools / messages. Every
#             number measured before this switch existed was taken under it, so it
#             stays the default and the older arms stay comparable.
#   "cdc"     content decides. A boundary falls where the token window at it hashes
#             to zero mod the target, so a run of tokens is cut the same way wherever
#             it appears and whatever precedes it -- which is what lets a document
#             quoted at a new offset, or quoted in part, still be found. Roles cannot:
#             one edit anywhere in a block loses the whole block.
SPLIT_MODE = os.environ.get("SGLANG_SUBCTX_SPLIT", "blocks")

# Average chunk length "cdc" aims for, and the length at which it gives up waiting for
# a boundary and forces one. Rounded down to a power of two (the rule is a bit mask).
# Keep MAX well above TARGET: a forced cut is the one boundary content did not choose,
# so it is also the one a shifted occurrence can fail to reproduce.
CDC_TARGET_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_TARGET", "256"))
CDC_MAX_TOKENS = int(os.environ.get("SGLANG_SUBCTX_CDC_MAX", "1024"))


def hash_subcontext_keys() -> bool:
    """Whether block namespaces are addressed by content."""
    return HASH_SUBCONTEXT_KEYS or INDEX_SUBCONTEXTS or INDEX_DRYRUN


if SPLIT_MODE not in ("blocks", "cdc"):
    raise ValueError(
        f"SGLANG_SUBCTX_SPLIT={SPLIT_MODE!r}; want 'blocks' or 'cdc'"
    )

if SPLIT_MODE == "cdc" and not DISABLE_SUBCONTEXT and not hash_subcontext_keys():
    # Content-defined blocks have no role to be named after, so their namespaces would
    # be numbered by position -- one namespace for every content, pinned to the first
    # offset anyone inserted it at. That is the bug content hashing exists to fix, and
    # it would silently undo the split rather than fail. Moot when the split is off
    # altogether, which is why the baseline arm can leave this exported and still run.
    raise ValueError(
        "SGLANG_SUBCTX_SPLIT=cdc needs content-addressed namespaces: set "
        "SGLANG_SUBCTX_INDEX=1 (or SGLANG_SUBCTX_HASH_KEYS=1)"
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
