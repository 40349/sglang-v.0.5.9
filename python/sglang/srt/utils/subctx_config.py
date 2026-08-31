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


def rotation_unsupported_reason(model_runner, tree_cache) -> Optional[str]:
    """Return why cached K cannot be re-rotated here, or None if it can.

    Rotating by a delta is only valid when RoPE's angle is linear in the position and
    the cache row for that delta is a unit rotation. That is true for the plain
    ``RotaryEmbedding`` and false for most of its subclasses, so the check is on the
    exact type rather than ``isinstance`` -- a YaRN cache row carries an ``mscale``
    factor and would quietly scale K by ``mscale**2`` on every reuse.
    """
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.rotate_kv import find_rotary_embedding

    sub_context = unsupported_reason(tree_cache)
    if sub_context is not None:
        return f"sub-contexts are unsupported ({sub_context})"

    rotary, why = find_rotary_embedding(model_runner.model)
    if rotary is None:
        return why
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
    return None
