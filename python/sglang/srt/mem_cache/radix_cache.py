from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64
from sglang.srt.utils import host_timer
from sglang.srt.utils import subctx_config
from sglang.srt.utils.subctx_config import CACHE_SUBCONTEXT_OUTPUT
from sglang.srt.utils.subctx_trace import TRACE_ON, trace

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority
        # Absolute position in the prompt of this node's first token when its KV was
        # computed. RoPE rotates K by absolute position, so KV is only reusable at the
        # position it was built for. Redundant in a single namespace (the path from the
        # root IS the position), load-bearing per block: a block's path spans only the
        # block, so its position is the sum of the preceding blocks' lengths. Recorded
        # per node rather than folded into the key, which keeps one copy per token
        # sequence and leaves a later stage the delta it needs to rotate a hit.
        self.canonical_position: int = 0

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()
        # Set by the scheduler once the model is up, when the model's RoPE is
        # delta-composable (see `subctx_config.rotation_unsupported_reason`). None
        # means displaced sub-context hits are dropped rather than rotated.
        self.kv_rotator = None
        # Tokens re-filed by `_reverse_rotate_insert_sub_contexts`, drained by the
        # forward trace. Kept on the cache rather than the request because the re-file
        # happens at finish, after that request's last forward pass.
        self.sub_context_reinserted_tokens = 0
        # Pool-accounting audit (see `_audit_sub_context_finish`). The pool size is not
        # known here; the scheduler fills it in once it has one. 0 disables the check.
        self.max_total_num_tokens_for_audit = 0
        self._sub_context_audit_drift = 0

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._record_all_cleared_event()

    def supports_sub_contexts(self) -> bool:
        """Both per-namespace paths are implemented here, but only in the plain
        device-side form.

        ``type(self) is RadixCache`` rather than ``isinstance``: a subclass adds a tier
        this insert path does not maintain (HiRadixCache matches against a host tier),
        so it would read and write different places. EAGLE rewrites keys into bigrams
        and ``page_size > 1`` inserts page-aligned prefixes; the per-block slicing
        accounts for neither.
        """
        return (
            type(self) is RadixCache
            and not self.disable
            and self.page_size == 1
            and not self.is_eagle
        )

    def matched_canonical_position(
        self, last_node: Optional[TreeNode], hit_len: int
    ) -> Optional[int]:
        """Where the KV behind a match was computed.

        The match runs from the namespace root down to ``last_node``, whose key is
        fully matched (``match_prefix`` splits a node when the match ends inside it),
        so the chain starts ``hit_len - len(last_node.key)`` tokens before that node's
        own position. Callers compare this against the position they are about to
        reuse at: equal is the fast path, different means real content rotated for
        somewhere else -- dropped today, and the difference is the delta a later stage
        would rotate by.
        """
        if last_node is None or hit_len <= 0:
            return None
        return last_node.canonical_position - (hit_len - len(last_node.key))

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:

        # --- 追蹤 Radix Tree ---
        if TRACE_ON:
            trace(f"[TRACE-4 RadixCache] 準備插入節點")
            trace(f"[TRACE-4 RadixCache] 插入的 key.extra_key={params.key.extra_key}")
        # ----------------------

        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(
            self.root_node, key, value, priority, params.canonical_position
        )
        return InsertResult(prefix_len=prefix_len)

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    @host_timer.timed("cache_finished")
    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if TRACE_ON:
            trace(
                f"[TRACE-4 FINISHED cls={type(self).__name__}] rid={req.rid} "
                f"has_sub={req.has_sub_contexts} page={self.page_size} "
                f"eagle={self.is_eagle} protected={req.cache_protected_len} "
                f"sub_nodes={req.sub_context_last_nodes is not None}"
            )

        # Safety net: a request that finished without ever reaching
        # `_cache_unfinished_sub_contexts` (gate declined, or aborted while queued) still
        # holds its scheduling-time match locks. Idempotent if already released there.
        req.release_sub_context_match_locks(self)
        # Likewise for rotated copies it allocated but never handed to req_to_token.
        # `prepare_for_extend` clears the list, so this is a no-op once it has run and
        # the slots are covered by the ordinary free-from-req_to_token paths.
        req.release_sub_context_rotated_slots(self)

        # The prompt was already inserted per-namespace in `cache_unfinished_req`, so
        # this unlocks those leaves rather than re-inserting, and hands the generated
        # continuation to the last block's namespace.
        if req.sub_context_last_nodes is not None:
            # `_finish_sub_contexts` clears these on its way out.
            owned_at_entry = list(req.sub_context_tree_owned or [])
            lens_at_entry = list(req.sub_context_owned_lens or [])
            self._finish_sub_contexts(req, token_ids, kv_indices, is_insert)
            if self.max_total_num_tokens_for_audit:
                self._audit_sub_context_finish(
                    req, token_ids, owned_at_entry, lens_at_entry
                )
            return

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def _audit_sub_context_finish(
        self,
        req: Req,
        token_ids: List[int],
        owned_at_entry: List[bool],
        lens_at_entry: List[int],
    ) -> None:
        """Name the request that broke the pool's accounting, the moment it breaks it.

        sglang checks ``available + evictable == max - protected`` only when the
        scheduler goes idle, by which point hundreds of requests have finished and the
        culprit is unrecoverable. This is the same identity, evaluated after every
        sub-context finish -- three O(1) reads -- and it reports only when the drift
        *changes*, so the first line names the one request that caused it and how.

        Both signs matter and mean different things:
          drift < 0  slots that are neither free nor in the tree -- lost
          drift > 0  the tree claims more than the pool holds -- two nodes owning the
                     same slots, which is the more dangerous one: those slots get
                     handed out twice.
        """
        alloc = self.token_to_kv_pool_allocator
        drift = (
            alloc.available_size() + self.evictable_size() + self.protected_size()
        ) - self.max_total_num_tokens_for_audit
        if drift == self._sub_context_audit_drift:
            return
        delta = drift - self._sub_context_audit_drift
        self._sub_context_audit_drift = drift
        segs = [(k, len(i), o) for i, k, o in req.iter_sub_contexts()]
        logger.error(
            "SUBCTX-IMBALANCE rid=%s this_request=%+d running=%+d (%s) "
            "prompt=%d committed=%d tree_owned=%s owned_lens=%s blocks=%s "
            "reinserted=%d rotated=%d moved=%d",
            req.rid,
            delta,
            drift,
            "tree owns slots twice" if delta > 0 else "slots lost",
            sum(n for _, n, _ in segs),
            len(token_ids),
            owned_at_entry,
            lens_at_entry,
            segs,
            req.sub_context_reinserted,
            req.sub_context_rotated,
            req.sub_context_moved,
        )

    def _finish_sub_contexts(
        self,
        req: Req,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        is_insert: bool = True,
    ):
        """Finish a request whose prompt was cached per-namespace.

        The prompt KV [0:cache_protected_len) stays in the tree, owned by the leaves in
        ``req.sub_context_last_nodes``; we only release those locks. The continuation
        goes to the last block's namespace (``_cache_sub_context_output``); whatever the
        tree does not take is freed here.
        """
        # Re-file blocks the namespace refused first: it is what makes the last block
        # tree-owned, which is what the output insert below requires.
        if is_insert:
            self._reverse_rotate_insert_sub_contexts(req, token_ids, kv_indices)

        kept = False
        if is_insert and CACHE_SUBCONTEXT_OUTPUT:
            kept = self._cache_sub_context_output(req, token_ids, kv_indices)

        # Free everything the tree did not take. That is not simply the tail past
        # `cache_protected_len`: a block the tree declined (a rotated copy, or a
        # position conflict) leaves a hole that later tree-owned blocks sit after, and
        # freeing from the first hole onwards would free slots the tree now owns.
        prompt_len = 0
        if req.sub_context_tree_owned is not None:
            for i, (seg_ids, _seg_key, offset) in enumerate(req.iter_sub_contexts()):
                end = min(offset + len(seg_ids), len(kv_indices))
                if end <= offset:
                    break
                prompt_len = end
                if not req.sub_context_tree_owned[i]:
                    self.token_to_kv_pool_allocator.free(kv_indices[offset:end])
        else:
            prompt_len = req.cache_protected_len

        if not kept:
            # Free the generated tail (not owned by any namespace node). When the tail
            # WAS taken, every block must have been tree-owned for the gate to pass, so
            # the loop above freed nothing and this is the only exclusion needed.
            if prompt_len < len(kv_indices):
                self.token_to_kv_pool_allocator.free(kv_indices[prompt_len:])

        # Release the per-namespace prompt locks taken in cache_unfinished_req.
        for node in req.sub_context_last_nodes:
            self.dec_lock_ref(node)
        req.sub_context_last_nodes = None
        req.sub_context_owned_lens = None
        req.sub_context_tree_owned = None
        req.sub_context_tree_canonical = None

    def _reverse_rotate_insert_sub_contexts(
        self, req: Req, token_ids: List[int], kv_indices: torch.Tensor
    ) -> int:
        """File a declined block under the position its namespace already stands for.

        A block whose namespace holds the same tokens at another position is refused by
        `_cache_unfinished_sub_contexts`: one node cannot stand for two rotations. But
        the request *has* that block's KV, only rotated for its own offset -- so
        rotating it back by ``canonical - offset`` makes it exactly what the namespace
        already means, and it can be inserted there with the chain intact. This is the
        read path's rotation run backwards, and it is what lets the generated reply be
        cached: `_cache_sub_context_output` only extends a namespace this request owns
        its whole last block in.

        **This runs at finish, not per chunk, and that is load-bearing.** The rotation
        is in place, and `cache_unfinished_req` fires right after prefill on a request
        that is still decoding (`scheduler_output_processor_mixin.py:183`) -- its own
        attention reads these slots on every step that follows, so moving them to
        another position would silently corrupt the generation still in flight.
        A finished request reads them never again.

        Returns how many tokens were re-filed.
        """
        if self.kv_rotator is None or req.sub_context_tree_owned is None:
            return 0

        prompt_len = sum(len(seg) for seg in req.sub_context_ids)
        if len(kv_indices) < prompt_len:
            return 0  # aborted mid-prefill; nothing settled enough to re-file

        reinserted = 0
        for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
            if not seg_ids or req.sub_context_tree_owned[i]:
                continue
            end = offset + len(seg_ids)
            radix_key = RadixKey(token_ids[offset:end], seg_key)
            probe = self.match_prefix(MatchPrefixParams(key=radix_key))
            canonical = self.matched_canonical_position(
                probe.last_device_node, len(probe.device_indices)
            )
            if canonical is None:
                # Whatever conflicted has since been evicted: the namespace is free to
                # take this block where it actually sits.
                canonical = offset
            delta = canonical - offset
            if delta != 0:
                if not self.kv_rotator.can_rotate(delta):
                    continue  # out of the cos_sin_cache's range; drop as before
                seg_slots = kv_indices[offset:end]
                # In place: these slots are this request's own (freshly computed, or a
                # rotated copy it owns), never a node other requests hold a lock on.
                self.kv_rotator.rotate_into(seg_slots, seg_slots, delta)

            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=kv_indices[offset:end].to(dtype=torch.int64, copy=True),
                    priority=getattr(req, "priority", 0) or 0,
                    canonical_position=canonical,
                )
            )
            # Nothing of this block was ever tree-owned, so everything the namespace
            # already had is a duplicate this request must give back.
            if result.prefix_len > 0:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[offset : offset + result.prefix_len]
                )
            seg_match = self.match_prefix(MatchPrefixParams(key=radix_key))
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(offset, end)), seg_match.device_indices
            )
            req.sub_context_tree_owned[i] = True
            req.sub_context_tree_canonical[i] = canonical
            req.sub_context_owned_lens[i] = len(seg_ids)
            reinserted += len(seg_ids)
            if TRACE_ON:
                trace(
                    f"[TRACE-4 SUBCTX-REVERSE-ROTATE] rid={req.rid} "
                    f"extra_key={seg_key!r} offset={offset} canonical={canonical} "
                    f"delta={delta} tokens={len(seg_ids)} dup={result.prefix_len}"
                )
        req.sub_context_reinserted += reinserted
        self.sub_context_reinserted_tokens += reinserted
        if reinserted:
            # Blocks that were holes a moment ago are tree-owned now, so the protected
            # prefix has grown -- and `_cache_sub_context_output` gates on it covering
            # the whole prompt.
            req.cache_protected_len = self._sub_context_protected_len(
                req, len(kv_indices)
            )
        return reinserted

    def _cache_sub_context_output(
        self, req: Req, token_ids: List[int], kv_indices: torch.Tensor
    ) -> bool:
        """Extend the last block's namespace with the generated tokens.

        In an agent loop this reply is part of the *next* turn's prompt, so dropping
        its KV means re-prefilling it every turn. The namespace is extended with
        ``block ++ generated`` under the same ``extra_key``, continuing the node the
        block already occupies, so next turn matches through the reply and stops where
        the render diverges. The tail sits immediately after the block, so it inherits
        the block's canonical position -- which is the *tree's*, not necessarily this
        request's: `_reverse_rotate_insert_sub_contexts` may have filed the block under
        a position it was not computed at, and then the reply has to make the same trip.

        Only a fully prefilled prompt qualifies: an extension off a partial block would
        be keyed to a prefix no later request reproduces.

        Returns True if the tree took the tail (caller must not free it), False if this
        request was declined and the tail is still the caller's to free.
        """
        if not req.sub_context_ids or not req.sub_context_extra_keys:
            return False

        last_seg = req.sub_context_ids[-1]
        prompt_len = sum(len(seg) for seg in req.sub_context_ids)
        if (
            not last_seg
            or req.cache_protected_len != prompt_len  # prompt not fully inserted
            or req.sub_context_owned_lens is None
            or req.sub_context_owned_lens[-1] != len(last_seg)
            or len(token_ids) <= prompt_len  # nothing generated (or nothing committed)
        ):
            return False

        offset = prompt_len - len(last_seg)
        seg_key = req.sub_context_extra_keys[-1]

        # Where the tree holds the block -- `offset` on the fast path, and the position
        # the namespace already stood for when the block was re-filed there.
        canonical = offset
        if req.sub_context_tree_canonical is not None:
            canonical = req.sub_context_tree_canonical[-1]
            if canonical is None:
                return False
        delta = canonical - offset
        if delta != 0:
            # The reply was computed at `prompt_len` but continues a block the tree
            # holds `delta` earlier, so it has to move by the same delta. In place: the
            # generated slots are this request's and it is finished reading them.
            if self.kv_rotator is None or not self.kv_rotator.can_rotate(delta):
                return False
            self.kv_rotator.rotate_into(
                kv_indices[prompt_len:], kv_indices[prompt_len:], delta
            )

        radix_key = RadixKey(token_ids[offset:], seg_key)
        values = kv_indices[offset:].to(dtype=torch.int64, copy=True)
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                priority=getattr(req, "priority", 0) or 0,
                canonical_position=canonical,
            )
        )

        # This request already owns [offset, offset + len(last_seg)) in the tree, so
        # only a match reaching *past* it is a freshly computed duplicate. Everything
        # the insert did not match now belongs to the tree and must not be freed.
        owned = len(last_seg)
        if result.prefix_len > owned:
            self.token_to_kv_pool_allocator.free(
                kv_indices[offset + owned : offset + result.prefix_len]
            )

        if TRACE_ON:
            trace(
                f"[TRACE-4 SUBCTX-OUTPUT] rid={req.rid} extra_key={seg_key!r} "
                f"block={owned} generated={len(token_ids) - prompt_len} "
                f"canonical={canonical} delta={delta} "
                f"dup={max(result.prefix_len - owned, 0)}"
            )
        return True

    @host_timer.timed("cache_unfinished")
    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if TRACE_ON:
            trace(
                f"[TRACE-4 UNFINISHED cls={type(self).__name__}] rid={req.rid} "
                f"chunked={chunked} has_sub={req.has_sub_contexts} "
                f"protected={req.cache_protected_len}"
            )

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Insert the prompt as one node PER namespace instead of a single
        # default-namespace node; under chunked prefill each chunk extends every
        # namespace by its newly-covered slice. Reuse across blocks is still an
        # approximation (no cross-namespace attention correction), and the generated
        # continuation is added at finish time, not here.
        #
        # This gate must stay equivalent to the read path's in
        # `Req.init_next_round_input`, or matches and inserts land in different
        # namespaces; `supports_sub_contexts` is what both consult.
        if (
            req.has_sub_contexts
            and self.supports_sub_contexts()
            and len(token_ids) <= sum(len(s) for s in req.sub_context_ids)
        ):
            self._cache_unfinished_sub_contexts(req, token_ids, kv_indices)
            return

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def _sub_context_protected_len(self, req: Req, covered: int) -> int:
        """How much of the prompt prefix the tree owns, contiguously from 0.

        Stops at the first block the tree did not take. Past that point ownership is
        interleaved, and the callers of ``cache_protected_len`` -- which all want "the
        prefix I must not free" -- can only use the contiguous part.
        """
        protected = 0
        for i, (seg_ids, _seg_key, offset) in enumerate(req.iter_sub_contexts()):
            covered_end = min(offset + len(seg_ids), covered)
            if covered_end <= offset:
                break
            if not req.sub_context_tree_owned[i]:
                break
            protected = covered_end
        return protected

    def _rotate_append_sub_contexts(self, req: Req, end_k: int) -> int:
        """Attach blocks that only needed a different position to the covered prefix.

        This is what lets a block be reused *after* a partially recomputed one. The
        stitch could not use it -- the tokens before it did not exist yet -- but now
        that this chunk has computed them, the block's cached KV can be copied to fresh
        slots with K rotated by ``offset - canonical`` and written straight into
        req_to_token, so the next chunk starts after it.

        Returns how many tokens were appended. Must be called before
        ``release_sub_context_match_locks``: it reads the match results those locks
        protect, and releasing clears them.
        """
        if not subctx_config.ROTATE_ACROSS_RECOMPUTE or self.kv_rotator is None:
            return 0
        if req.sub_context_match_indices is None or req.sub_context_match_positions is None:
            return 0

        # Never cover the whole prompt: a forward pass with nothing to compute is not a
        # valid batch (the position/cumsum kernels launch with an empty grid and CUDA
        # rejects it). This is the same bound the stitch enforces as
        # `max_prefix_len = len(fill_ids) - 1`.
        cap = max(sum(len(s) for s in req.sub_context_ids) - 1, 0)

        appended = 0
        cursor = end_k
        for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
            if offset + len(seg_ids) <= cursor:
                continue  # already covered by this chunk
            if offset != cursor or not seg_ids:
                break  # not flush with the end of coverage; nothing to attach to
            indices = req.sub_context_match_indices[i]
            canonical = req.sub_context_match_positions[i]
            if indices is None or canonical is None or len(indices) != len(seg_ids):
                break  # a miss or a partial hit: this block has to be computed
            delta = offset - canonical
            if delta == 0:
                break  # not displaced -- the stitch would already have taken it
            # A partial take is sound: the delta belongs to the block, so it moves every
            # one of its tokens by the same amount. It does end the walk, though -- the
            # next block is no longer flush with the covered prefix.
            take = min(len(seg_ids), cap - cursor)
            if take <= 0:
                break
            dst = req._rotate_sub_context_block(self, indices[:take], delta)
            if dst is None:
                break
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(offset, offset + take)), dst
            )
            req.sub_context_rotated += take
            cursor += take
            appended += take
            if TRACE_ON:
                trace(
                    f"[TRACE-4 SUBCTX-ROTATE-APPEND] rid={req.rid} "
                    f"extra_key={seg_key!r} offset={offset} canonical={canonical} "
                    f"delta={delta} tokens={take}/{len(seg_ids)}"
                )
            if take < len(seg_ids):
                break
        return appended

    def _cache_unfinished_sub_contexts(
        self, req: Req, token_ids: List[int], kv_indices: torch.Tensor
    ):
        """Insert the (possibly partial) prompt block-by-block, each under its own
        ``extra_key`` namespace, locking each leaf so the KV survives chunks and decode.

        ``token_ids`` is the cumulative prefill prefix so far: each call extends every
        reached segment to its currently-covered end, with per-segment progress in
        ``req.sub_context_owned_lens`` so only duplicate fresh slots are freed. Locks
        are self-owned -- the scheduler's per-chunk lock sits on ``req.last_node``,
        which we reset to the (no-op) root, so namespace locks are released and re-taken
        symmetrically.
        """
        values = kv_indices.to(dtype=torch.int64, copy=True)
        priority = getattr(req, "priority", 0) or 0
        end_k = len(token_ids)  # cumulative prefill length so far

        # Release the scheduler lock (root => no-op) and the previous chunk's namespace
        # locks; we re-take fresh namespace locks below.
        self.dec_lock_ref(req.last_node)
        if req.sub_context_last_nodes is not None:
            for node in req.sub_context_last_nodes:
                self.dec_lock_ref(node)

        if req.sub_context_owned_lens is None:
            req.sub_context_owned_lens = [0] * len(req.sub_context_extra_keys)
        if req.sub_context_tree_owned is None:
            req.sub_context_tree_owned = [False] * len(req.sub_context_extra_keys)
        if req.sub_context_tree_canonical is None:
            req.sub_context_tree_canonical = [None] * len(req.sub_context_extra_keys)

        seg_last_nodes = []
        for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
            covered_end = min(offset + len(seg_ids), end_k)
            if covered_end <= offset:
                continue  # this segment not reached by the current chunk yet
            seg_key_ids = token_ids[offset:covered_end]
            radix_key = RadixKey(seg_key_ids, seg_key)

            # First writer wins. If the namespace already holds this sequence from a
            # request that computed it elsewhere, merging would leave one node standing
            # for two rotations, so this block is not inserted: its slots stay this
            # request's (freed at finish) and the tree keeps its single canonical copy.
            #
            # Skip the block, do not stop. Namespaces are independent trees, so a
            # position conflict here says nothing about the next block, and stopping
            # would keep the growing `messages` block out of the cache for the rest of
            # the conversation. The resulting hole is why ownership is tracked per
            # block instead of as one protected prefix length.
            probe = self.match_prefix(MatchPrefixParams(key=radix_key))
            probe_hit = len(probe.device_indices)
            existing = self.matched_canonical_position(
                probe.last_device_node, probe_hit
            )
            if existing is not None and existing != offset:
                if TRACE_ON:
                    trace(
                        f"[TRACE-4 SUBCTX-MOVED-WRITE] rid={req.rid} "
                        f"extra_key={seg_key!r} offset={offset} canonical={existing} "
                        f"hit={probe_hit} -- not inserted"
                    )
                continue

            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values[offset:covered_end],
                    priority=priority,
                    canonical_position=offset,
                )
            )
            # Free freshly-computed slots that duplicate what is already in this
            # namespace beyond what THIS request had previously inserted.
            owned = req.sub_context_owned_lens[i]
            if result.prefix_len > owned:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[offset + owned : offset + result.prefix_len]
                )

            # Re-point req_to_token at the tree-owned slots for this block.
            seg_match = self.match_prefix(MatchPrefixParams(key=radix_key))
            seg_indices = seg_match.device_indices
            assert len(seg_indices) == covered_end - offset, (
                f"{len(seg_indices)=}, {covered_end - offset=}, extra_key={seg_key!r}"
            )
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(offset, covered_end)), seg_indices
            )

            self.inc_lock_ref(seg_match.last_device_node)
            seg_last_nodes.append(seg_match.last_device_node)
            req.sub_context_tree_owned[i] = True
            req.sub_context_tree_canonical[i] = offset
            req.sub_context_owned_lens[i] = covered_end - offset

        req.sub_context_last_nodes = seg_last_nodes

        # Stage 2: the tokens up to end_k are settled now, so a block sitting exactly
        # there can be rotated onto the end of the prefix and skipped by the next
        # chunk. This MUST run before the match locks are released -- it reads the
        # matches those locks protect, and `release_sub_context_match_locks` clears
        # them along with the locks.
        appended = self._rotate_append_sub_contexts(req, end_k)
        # Settle what the stitch held back for these blocks. Whatever the append did
        # not take is a real drop, reported now by the pass that gave up on it rather
        # than by the one that only intended to rotate it.
        if req.sub_context_deferred_moved:
            req.sub_context_moved += max(req.sub_context_deferred_moved - appended, 0)
            req.sub_context_deferred_moved = 0

        # Only now drop the scheduling-time match locks: the fresh per-namespace locks
        # above already cover the prompt, so protection is continuous across the
        # `insert` calls (which can evict).
        req.release_sub_context_match_locks(self)

        # req_to_token is the authoritative position -> slot map: it already holds the
        # tree-owned blocks, the freshly computed ones and any rotated copy appended
        # above. Rebuilding from it keeps this correct when a block in the middle was
        # skipped, which concatenating the tree-owned pieces would not.
        covered = end_k + appended
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :covered
        ].to(dtype=torch.int64, copy=True)
        req.cache_protected_len = self._sub_context_protected_len(req, covered)
        # Neutralize the scheduler's per-chunk lock pairing: root inc/dec are no-ops.
        req.last_node = self.root_node

        if TRACE_ON:
            trace(
                f"[TRACE-4 SUBCTX-INSERT] rid={req.rid} end_k={end_k} "
                f"segments={[(k, o) for k, o in zip(req.sub_context_extra_keys, req.sub_context_owned_lens)]}"
            )

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        new_node.canonical_position = child.canonical_position
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        child.canonical_position += split_len
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        canonical_position: int = 0,
    ):
        # `canonical_position` is where key[0] sits in the prompt: 0 for an ordinary
        # request, the block's offset for a sub-context block. Each node created below
        # records where its own first token sits.
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.canonical_position = canonical_position + total_prefix_length
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"extra_key={current_node.key.extra_key}",
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key='tool_def')))
    # tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    # tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key='tool_def'))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key='system_prompt'))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key='tool_def'))
        )
    )