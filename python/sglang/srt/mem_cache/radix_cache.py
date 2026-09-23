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
import os
import sys
import time
from collections import Counter, defaultdict
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
from sglang.srt.utils import host_timer, subctx_config
from sglang.srt.utils.subctx_config import CACHE_SUBCONTEXT_OUTPUT
from sglang.srt.utils.subctx_trace import TRACE_ON, trace

# Slot-ownership audits on the sub-context insert and finish paths.
AUDIT_ON = os.environ.get("SGLANG_SUBCTX_AUDIT", "") not in ("", "0")

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
        # Absolute prompt position this node's first token was computed at.
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


def _tree_held_mask(tree_slots: torch.Tensor, our_slots: torch.Tensor) -> torch.Tensor:
    """Per position of ``our_slots``: whether the tree's slot there is the same slot.

    Compared at every position, not as a leading run. A declined block can still hold
    some of a node's own slots; those are neither freed nor rotated in place.
    """
    mask = torch.zeros(len(our_slots), dtype=torch.bool, device=our_slots.device)
    n = min(len(tree_slots), len(our_slots))
    if n:
        mask[:n] = tree_slots[:n] == our_slots[:n]
    return mask


def _free_only_ours(allocator, slots: torch.Tensor, tree_held: torch.Tensor) -> None:
    """Hand back every slot in ``slots`` the tree is not holding."""
    ours = slots[~tree_held[: len(slots)]]
    if len(ours):
        allocator.free(ours)


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
        # KVRotator, set by the scheduler when rotation is on. None: displaced hits
        # are dropped.
        self.kv_rotator = None
        # SubContextIndex, set by the scheduler when the index is on.
        self.sub_context_index = None
        # Tokens re-filed at finish. Drained by the forward trace.
        self.sub_context_reinserted_tokens = 0
        # Doubly-owned slot count at the last `_audit_tree_duplicates` walk.
        self._audit_dup_seen = 0

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
        """Only the plain device-side tree: no subclass (e.g. HiRadixCache), no
        EAGLE bigram keys, and ``page_size == 1``.
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
        """The position the first token of a ``hit_len`` match was computed at.

        ``last_node`` is fully matched (``match_prefix`` splits a partial node), so the
        chain starts ``hit_len - len(last_node.key)`` tokens before its position.
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
        # A view of req_to_token: re-pointed slots show through it.
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

        # Release what a request that never reached the insert path still holds: its
        # match locks and unplaced rotated copies. Both calls are idempotent.
        req.release_sub_context_match_locks(self)
        req.release_sub_context_rotated_slots(self)

        if self.serves_sub_contexts(req):
            if not AUDIT_ON:
                self._finish_sub_contexts(req, token_ids, kv_indices, is_insert)
                return
            held = kv_indices.tolist()
            # `_finish_sub_contexts` clears these on its way out.
            owned_at_entry = list(req.sub_context_tree_owned or [])
            lens_at_entry = list(req.sub_context_owned_lens or [])
            freed, freed_at = self._capture_frees(
                lambda: self._finish_sub_contexts(
                    req, token_ids, kv_indices, is_insert
                )
            )
            self._audit_sub_context_finish(
                req, token_ids, held, freed, freed_at, owned_at_entry, lens_at_entry
            )
            self._audit_tree_duplicates("finish", req)
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

    def _capture_frees(self, fn) -> Tuple[set, dict]:
        """Run ``fn``; return the slots it passed to ``free`` and the line of each call.

        Wraps ``free`` directly: inside a ``free_group`` the allocator's size does not
        change until the group ends.
        """
        alloc = self.token_to_kv_pool_allocator
        real_free, freed, freed_at = alloc.free, set(), {}

        def _counting_free(indices):
            site = sys._getframe(1).f_lineno  # the caller's line
            slots = indices.tolist()
            freed.update(slots)
            for slot in slots:
                freed_at.setdefault(slot, set()).add(site)
            return real_free(indices)

        alloc.free = _counting_free
        try:
            fn()
        finally:
            alloc.free = real_free
        return freed, freed_at

    def _tree_owned_slots(self, req: Req, token_ids: List[int]) -> set:
        """Every slot reachable from a namespace node for this request's blocks.

        Each block is looked up under the namespace holding its slots
        (``sub_context_source_key`` when set).
        """
        owned = set()
        segs = list(req.iter_sub_contexts())
        source_keys = getattr(req, "sub_context_source_key", None)
        for i, (seg_ids, seg_key, offset) in enumerate(segs):
            end = min(offset + len(seg_ids), len(token_ids))
            if source_keys is not None and source_keys[i] is not None:
                seg_key = source_keys[i]
            if end > offset:
                owned.update(
                    self.match_prefix(
                        MatchPrefixParams(key=RadixKey(token_ids[offset:end], seg_key))
                    ).device_indices.tolist()
                )
        if segs:
            # The last namespace may also hold the generated tail.
            _ids, last_key, last_off = segs[-1]
            owned.update(
                self.match_prefix(
                    MatchPrefixParams(key=RadixKey(token_ids[last_off:], last_key))
                ).device_indices.tolist()
            )
        return owned

    def _double_owners(self, double: set) -> List[Tuple[str, int]]:
        """Per namespace, how many of ``double`` its nodes hold. Walks the whole tree;
        called only after a check has failed.
        """
        owners = Counter()
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            value = node.value
            if value is None or len(value) == 0 or not torch.is_tensor(value):
                continue
            hit = len(double.intersection(value.tolist()))
            if hit:
                owners[str(node.key.extra_key)] += hit
        return sorted(owners.items())

    def _audit_sub_context_chunk(
        self, req: Req, token_ids: List[int], freed: set, freed_at: dict
    ) -> None:
        """Report slots this insert pass freed that a namespace still holds.

        Only ``double`` is checked; mid-prefill a request may hold slots that are
        neither freed nor tree-owned yet.
        """
        double = freed & self._tree_owned_slots(req, token_ids)
        if not double:
            return
        logger.error(
            "SUBCTX-AUDIT-CHUNK rid=%s double_owned=%d freed=%d covered=%d "
            "tree_owned=%s owned_lens=%s double_at=%s owners=%s",
            req.rid,
            len(double),
            len(freed),
            len(token_ids),
            req.sub_context_tree_owned,
            req.sub_context_owned_lens,
            sorted(
                Counter(
                    site for slot in double for site in freed_at.get(slot, ())
                ).items()
            ),
            self._double_owners(double),
        )

    def _audit_sub_context_finish(
        self,
        req: Req,
        token_ids: List[int],
        held: List[int],
        freed: set,
        freed_at: dict,
        owned_at_entry: List[bool],
        lens_at_entry: List[int],
    ) -> None:
        """Check every slot this request held ends up freed or tree-owned, not both.

        ``freed`` is the set of slots passed to ``free`` (see ``_capture_frees``).

            lost    neither freed nor held by a namespace
            double  freed while a namespace still holds it
            dup     positions in req_to_token sharing a slot with another position
        """
        segs = list(req.iter_sub_contexts())
        owned = self._tree_owned_slots(req, token_ids)

        held_set = set(held)
        lost = held_set - freed - owned
        double = held_set & freed & owned
        if not lost and not double:
            return
        logger.error(
            "SUBCTX-AUDIT rid=%s lost=%d double_owned=%d held=%d dup=%d freed=%d "
            "kept=%d prompt=%d committed=%d tree_owned=%s owned_lens=%s blocks=%s "
            "reinserted=%d rotated=%d moved=%d lost_at=%s double_at=%s double_in=%s "
            "owners=%s double_pos=%s double_slots=%s",
            req.rid,
            len(lost),
            len(double),
            len(held_set),
            len(held) - len(held_set),
            len(freed),
            len(held_set & owned),
            sum(len(i) for i, _, _ in segs),
            len(token_ids),
            owned_at_entry,
            lens_at_entry,
            [(k, len(i), o) for i, k, o in segs],
            req.sub_context_reinserted,
            req.sub_context_rotated,
            req.sub_context_moved,
            # Which block the lost slots sat in, by position in req_to_token.
            sorted(
                {
                    k
                    for i, k, o in segs
                    for pos, slot in enumerate(held)
                    if slot in lost and o <= pos < o + len(i)
                }
                | ({"generated_tail"} if any(
                    slot in lost
                    for slot in held[sum(len(i) for i, _, _ in segs):]
                ) else set())
            ) if lost else [],
            # Source lines that freed the doubly-owned slots, and their blocks.
            sorted(
                Counter(
                    site for slot in double for site in freed_at.get(slot, ())
                ).items()
            ) if double else [],
            sorted(
                {
                    k
                    for i, k, o in segs
                    for pos, slot in enumerate(held)
                    if slot in double and o <= pos < o + len(i)
                }
            ) if double else [],
            self._double_owners(double) if double else [],
            # Positions and slot ids of the doubly-owned slots.
            [pos for pos, slot in enumerate(held) if slot in double][:8],
            sorted(double)[:8],
        )

    def _finish_sub_contexts(
        self,
        req: Req,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        is_insert: bool = True,
    ):
        """Finish a request whose prompt was cached per namespace.

        Re-files declined blocks, offers the generated tail to the last block's
        namespace, frees every slot the tree does not hold, and releases the locks.
        """
        # First, so the last block can become tree-owned for the output insert.
        if is_insert:
            self._reverse_rotate_insert_sub_contexts(req, token_ids, kv_indices)

        kept = False
        if is_insert and CACHE_SUBCONTEXT_OUTPUT:
            kept = self._cache_sub_context_output(req, token_ids, kv_indices)

        # Free per block: a declined block can sit between tree-owned ones.
        tree_owned = req.sub_context_tree_owned
        if tree_owned is None and req.sub_context_extra_keys:
            # No insert pass ran: every block is treated as declined.
            tree_owned = [False] * len(req.sub_context_extra_keys)

        prompt_len = 0
        if tree_owned is not None:
            for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
                end = min(offset + len(seg_ids), len(kv_indices))
                if end <= offset:
                    break
                prompt_len = end
                if tree_owned[i]:
                    continue
                # Free only the slots the tree does not hold, asking the namespace the
                # slots live in (`sub_context_source_key` for a partly reused run).
                block = kv_indices[offset:end]
                held_key = seg_key
                source_keys = getattr(req, "sub_context_source_key", None)
                if source_keys is not None and source_keys[i] is not None:
                    held_key = source_keys[i]
                match = self.match_prefix(
                    MatchPrefixParams(key=RadixKey(token_ids[offset:end], held_key))
                )
                _free_only_ours(
                    self.token_to_kv_pool_allocator,
                    block,
                    _tree_held_mask(match.device_indices, block),
                )
        else:
            prompt_len = req.cache_protected_len

        if not kept:
            # The generated tail, if the tree did not take it.
            if prompt_len < len(kv_indices):
                self.token_to_kv_pool_allocator.free(kv_indices[prompt_len:])

        # The per-namespace locks from cache_unfinished_req (None if it never ran).
        for node in req.sub_context_last_nodes or []:
            self.dec_lock_ref(node)
        req.sub_context_last_nodes = None
        req.sub_context_owned_lens = None
        req.sub_context_tree_owned = None
        req.sub_context_tree_canonical = None

    @host_timer.timed("subctx_rev_rotate")
    def _reverse_rotate_insert_sub_contexts(
        self, req: Req, token_ids: List[int], kv_indices: torch.Tensor
    ) -> int:
        """Rotate each declined block to its namespace's position and insert it there.

        A block declined because its namespace holds it at ``canonical != offset`` is
        rotated in place by ``canonical - offset``. Runs at finish only: the rotation is
        in place, and a decoding request still reads these slots.

        Returns how many tokens were re-filed.
        """
        if self.kv_rotator is None or req.sub_context_tree_owned is None:
            return 0

        prompt_len = sum(len(seg) for seg in req.sub_context_ids)
        if len(kv_indices) < prompt_len:
            return 0  # aborted mid-prefill; nothing settled enough to re-file

        reinserted = 0
        no_insert = getattr(req, "sub_context_no_insert", None)
        for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
            if not seg_ids or req.sub_context_tree_owned[i]:
                continue
            if no_insert is not None and no_insert[i]:
                continue
            end = offset + len(seg_ids)
            radix_key = RadixKey(token_ids[offset:end], seg_key)
            probe = self.match_prefix(MatchPrefixParams(key=radix_key))
            canonical = self.matched_canonical_position(
                probe.last_device_node, len(probe.device_indices)
            )
            tree_held = _tree_held_mask(probe.device_indices, kv_indices[offset:end])
            if canonical is None:
                # The namespace is empty now: file the block where it sits.
                canonical = offset
            delta = canonical - offset
            if delta != 0:
                if bool(tree_held.any()):
                    # Some slots are a node's: never rotate those in place. Skipped.
                    continue
                if not self.kv_rotator.can_rotate(delta):
                    continue  # out of the cos_sin_cache's range
                seg_slots = kv_indices[offset:end]
                # In place: every slot here is this request's own.
                self.kv_rotator.rotate_into(
                    seg_slots, seg_slots, delta, stage="subctx_rotate_finish"
                )

            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=kv_indices[offset:end].to(dtype=torch.int64, copy=True),
                    priority=getattr(req, "priority", 0) or 0,
                    canonical_position=canonical,
                )
            )
            # Free this request's duplicates of what the tree already held.
            _free_only_ours(
                self.token_to_kv_pool_allocator,
                kv_indices[offset : offset + result.prefix_len],
                tree_held,
            )
            seg_match = self.match_prefix(MatchPrefixParams(key=radix_key))
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(offset, end)), seg_match.device_indices
            )
            # Locked like every inserted block; released at the end of the caller.
            self.inc_lock_ref(seg_match.last_device_node)
            req.sub_context_last_nodes.append(seg_match.last_device_node)
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
            req.cache_protected_len = self._sub_context_protected_len(
                req, len(kv_indices)
            )
        return reinserted

    def _cache_sub_context_output(
        self, req: Req, token_ids: List[int], kv_indices: torch.Tensor
    ) -> bool:
        """Insert ``last block ++ generated tokens`` into the last block's namespace.

        Only when the whole prompt is tree-owned. If the tree holds the block at another
        position, the tail is first rotated by the same delta.

        Returns True if the tree took the tail (the caller must not free it).
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

        # Where the tree holds the block.
        canonical = offset
        if req.sub_context_tree_canonical is not None:
            canonical = req.sub_context_tree_canonical[-1]
            if canonical is None:
                return False
        delta = canonical - offset
        if delta != 0:
            # Rotate the tail in place; its slots are this finished request's own.
            if self.kv_rotator is None or not self.kv_rotator.can_rotate(delta):
                return False
            self.kv_rotator.rotate_into(
                kv_indices[prompt_len:],
                kv_indices[prompt_len:],
                delta,
                stage="subctx_rotate_finish",
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

        # The block itself is already tree-owned; only a match past it is a duplicate.
        owned = len(last_seg)
        # The insert must match the whole block; otherwise part of it was stored twice.
        if result.prefix_len < owned:
            logger.error(
                "SUBCTX-OUTPUT-UNDERMATCH rid=%s extra_key=%r offset=%d "
                "block=%d matched=%d -- the namespace lost the block between the "
                "insert and here; %d slots may now be owned twice",
                req.rid,
                seg_key,
                offset,
                owned,
                result.prefix_len,
                owned - result.prefix_len,
            )
        if result.prefix_len > owned:
            # Free the tail's duplicates, except slots the tree itself holds.
            tail = kv_indices[offset + owned : offset + result.prefix_len]
            tree_tail = self.match_prefix(
                MatchPrefixParams(key=radix_key)
            ).device_indices[owned : result.prefix_len]
            _free_only_ours(
                self.token_to_kv_pool_allocator,
                tail,
                _tree_held_mask(tree_tail, tail),
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

        # Sub-context: insert each block into its own namespace, up to what this pass
        # covers. The generated tail is offered at finish.
        if self.serves_sub_contexts(req):
            if not AUDIT_ON:
                self._cache_unfinished_sub_contexts(req, token_ids, kv_indices)
                return
            freed, freed_at = self._capture_frees(
                lambda: self._cache_unfinished_sub_contexts(
                    req, token_ids, kv_indices
                )
            )
            self._audit_sub_context_chunk(req, token_ids, freed, freed_at)
            self._audit_tree_duplicates("chunk", req)
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
        """Length of the tree-owned prompt prefix, up to the first block not owned."""
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
        """Stage 2: rotate displaced full-hit blocks starting at ``end_k`` into place.

        Copies each block's cached KV to fresh slots, rotated by ``offset - canonical``,
        and writes them into req_to_token, so the next chunk starts after it. Must run
        before ``release_sub_context_match_locks``, which clears the matches it reads.

        Returns how many tokens were appended.
        """
        if not subctx_config.ROTATE_ACROSS_RECOMPUTE or self.kv_rotator is None:
            return 0
        if req.sub_context_match_indices is None or req.sub_context_match_positions is None:
            return 0

        # Leave at least the last token to compute, as the stitch does.
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
            # A partial take ends the walk.
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
        """Insert the prompt covered so far block by block, each in its own namespace.

        Each call extends every reached block to its covered end and re-locks the
        leaves; ``sub_context_owned_lens`` tracks progress so only duplicate fresh slots
        are freed. ``req.last_node`` is set to the root, so the scheduler's own
        per-chunk lock is a no-op.
        """
        values = kv_indices.to(dtype=torch.int64, copy=True)
        priority = getattr(req, "priority", 0) or 0
        end_k = len(token_ids)  # cumulative prefill length so far

        # Release the scheduler lock (root: no-op) and the previous chunk's locks.
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
        no_insert = getattr(req, "sub_context_no_insert", None)

        seg_last_nodes = []
        for i, (seg_ids, seg_key, offset) in enumerate(req.iter_sub_contexts()):
            covered_end = min(offset + len(seg_ids), end_k)
            if covered_end <= offset:
                continue  # this segment not reached by the current chunk yet
            seg_key_ids = token_ids[offset:covered_end]
            radix_key = RadixKey(seg_key_ids, seg_key)

            # Skipped blocks stay this request's and are freed at finish; later blocks
            # are still inserted.
            if no_insert is not None and no_insert[i]:
                continue

            probe = self.match_prefix(MatchPrefixParams(key=radix_key))
            probe_hit = len(probe.device_indices)
            existing = self.matched_canonical_position(
                probe.last_device_node, probe_hit
            )
            # First writer wins: a namespace holding these tokens at another position
            # is left as it is.
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
            if self.sub_context_index is not None:
                # The whole block: its tokens are the namespace's address.
                self.sub_context_index.register(seg_ids, req.extra_key)

        req.sub_context_last_nodes = seg_last_nodes

        # Stage 2. Before the match locks are released.
        appended = self._rotate_append_sub_contexts(req, end_k)
        # What the stitch deferred and the append did not take is dropped.
        if req.sub_context_deferred_moved:
            shortfall = max(req.sub_context_deferred_moved - appended, 0)
            req.sub_context_moved += shortfall
            req.sub_context_discarded += shortfall
            req.sub_context_deferred_moved = 0

        # The namespace locks above now cover the prompt.
        req.release_sub_context_match_locks(self)

        # Rebuilt from req_to_token, which holds every block's current slots.
        covered = end_k + appended
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :covered
        ].to(dtype=torch.int64, copy=True)
        req.cache_protected_len = self._sub_context_protected_len(req, covered)
        # The scheduler's per-chunk lock on the root is a no-op.
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
            was_namespace_root = x.parent is self.root_node
            self._delete_leaf(x)

            if was_namespace_root and self.sub_context_index is not None:
                # The namespace's first node is gone: forget the chunk.
                if x.key.extra_key is not None:
                    self.sub_context_index.unregister(x.key.extra_key)

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

    def audit_pool_invariant(self) -> str:
        """Describe a failed pool invariant: slots both in the tree and free
        (``IN_TREE_AND_FREE``), duplicate tree slots, and the evictable counter against
        a tree walk. Called by the idle memory check; independent of the audit flag.
        """
        alloc = self.token_to_kv_pool_allocator
        free = torch.cat(
            [
                t
                for t in (
                    getattr(alloc, "free_pages", None),
                    getattr(alloc, "release_pages", None),
                )
                if t is not None and t.numel()
            ]
            or [torch.empty(0, dtype=torch.int64)]
        )

        nodes, slots, keyed_unlocked = self._walk_tree_slots()
        uniq, counts = torch.unique(slots, return_counts=True)
        dup_slots = uniq[counts > 1]
        # A slot the tree serves that the pool also considers free.
        both = uniq[torch.isin(uniq, free.to(uniq.device))] if uniq.numel() else uniq

        return (
            f"tree walk: nodes={nodes} slots={slots.numel()} distinct={uniq.numel()} "
            f"dup_within_tree={slots.numel() - uniq.numel()} "
            f"dup_owners={self._double_owners(set(dup_slots.tolist()))} "
            f"dup_slots={dup_slots[:16].tolist()} "
            f"evictable_counter={self.evictable_size_} keyed_unlocked={keyed_unlocked} "
            f"counter_minus_walk={self.evictable_size_ - keyed_unlocked} "
            f"IN_TREE_AND_FREE={both.numel()} {both[:16].tolist()}"
        )

    def _walk_tree_slots(self) -> Tuple[int, torch.Tensor, int]:
        """Every slot the tree points at, one entry per (node, position), with duplicates."""
        held, keyed_unlocked, nodes = [], 0, 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is self.root_node or node.value is None:
                continue
            nodes += 1
            held.append(node.value)
            if node.lock_ref == 0:
                keyed_unlocked += len(node.key)
        slots = torch.cat(held) if held else torch.empty(0, dtype=torch.int64)
        return nodes, slots, keyed_unlocked

    def _audit_tree_duplicates(self, where: str, req: Req) -> None:
        """Log when the number of slots held by two tree nodes rises.

        A full tree walk; called after each sub-context insert under
        SGLANG_SUBCTX_AUDIT.
        """
        _nodes, slots, _keyed = self._walk_tree_slots()
        uniq, counts = torch.unique(slots, return_counts=True)
        dup = int(slots.numel() - uniq.numel())
        if dup > self._audit_dup_seen:
            dup_slots = uniq[counts > 1]
            logger.error(
                "SUBCTX-TREE-DUP at=%s rid=%s dup=%d (was %d) owners=%s slots=%s",
                where,
                req.rid,
                dup,
                self._audit_dup_seen,
                self._double_owners(set(dup_slots.tolist())),
                dup_slots[:16].tolist(),
            )
        self._audit_dup_seen = dup

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
        # `canonical_position`: prompt position of key[0] (a block's offset, else 0).
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