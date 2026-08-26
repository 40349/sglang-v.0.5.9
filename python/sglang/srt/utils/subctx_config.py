"""Switches and support rules for the sub-context path.

The sub-context split is driven by environment variables (it is a benchmarking
feature, flipped between runs rather than per request), and it only works on the
plain device-side radix tree with ``page_size == 1``. Both facts are needed in
places that must not import each other -- ``server_args``, the OpenAI chat
serving path, the radix cache -- so they live here.
"""

from __future__ import annotations

import os
from typing import Optional

# A/B switch: when set, chat requests carry no sub-context split, so the prompt
# takes the stock single-namespace radix path -- the baseline to measure against,
# on the same binary and the same loaded weights.
DISABLE_SUBCONTEXT = os.environ.get("SGLANG_DISABLE_SUBCONTEXT", "") not in ("", "0")

# Whether a finished sub-context request writes its generated tokens back into the
# last block's namespace (the ``messages_key`` block with the standard split), so
# the next turn of an agent loop -- whose message list now contains that reply --
# hits them instead of re-prefilling. Off restores the previous behaviour: the
# generated tail is freed at finish time.
CACHE_SUBCONTEXT_OUTPUT = os.environ.get(
    "SGLANG_SUBCONTEXT_CACHE_OUTPUT", "1"
) not in ("", "0")


def unsupported_reason(tree_cache) -> Optional[str]:
    """Return why ``tree_cache`` cannot serve sub-contexts, or None if it can.

    The read path (``Req._stitch_sub_contexts``) and the write path
    (``RadixCache._cache_unfinished_sub_contexts``) must agree: a cache that
    matches per namespace but inserts into the default one hits 0% forever and
    fills the tree with entries nobody will ever look up. So both consult
    ``BasePrefixCache.supports_sub_contexts`` and this function turns a False
    into a message that names the actual reason.
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
