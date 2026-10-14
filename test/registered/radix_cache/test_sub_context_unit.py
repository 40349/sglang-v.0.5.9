"""
Unit tests for the sub-context path of RadixCache.

A sub-context request carries its prompt cut into blocks by content. It is matched on
its prefix, then the index finds its registered chunks past the prefix; each found run
is copied (rotated when it sits at another position) into the request's own slots, and
the request is stored as one chain like a plain one. These drive match, insert and
finish on CPU tensors and cover:

- which caches can serve the split at all,
- a node's position along its chain,
- auto-truncation of the block list,
- the index: chunk locations, copies, rotation, fall backs, locks and slot ownership.

Usage:
    python test_sub_context_unit.py
    python -m pytest test_sub_context_unit.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import itertools
import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.subctx_index import SubContextIndex, chunk_id
from sglang.srt.utils import subctx_config
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

class FakeReqToTokenPool:
    """Just the two members the cache paths touch."""

    def __init__(self, size: int = 4, max_context_len: int = 64):
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class FakeAllocator:
    """Records freed slots instead of managing a pool."""

    def __init__(self, next_slot: int = 900, capacity: int = 10**6):
        self.device = torch.device("cpu")
        self.freed = []
        self.allocated = []
        self.next_slot = next_slot
        self.capacity = capacity

    def alloc(self, need_size: int):
        if need_size > self.capacity:
            return None  # pool full: the caller must fall back to recomputing
        self.capacity -= need_size
        out = torch.arange(
            self.next_slot, self.next_slot + need_size, dtype=torch.int64
        )
        self.next_slot += need_size
        self.allocated.extend(out.tolist())
        return out

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def available_size(self):
        # The number of slots freed so far.
        return len(self.freed)


class FakeRotator:
    """Stands in for the Triton kernel: records the calls, moves no KV.

    The kernel's arithmetic is tested on GPU by ``test_rotate_kv.py``.
    """

    def __init__(self, max_delta: int = 10**6):
        self.max_delta = max_delta
        self.calls = []
        self.copies = []

    def can_rotate(self, delta: int) -> bool:
        return delta != 0 and abs(delta) <= self.max_delta

    def rotate_into(self, dst_loc, src_loc, delta, stage="subctx_rotate"):
        self.calls.append((src_loc.tolist(), dst_loc.tolist(), delta))

    def copy_into(self, dst_loc, src_loc, stage="subctx_copy"):
        self.copies.append((src_loc.tolist(), dst_loc.tolist()))


def make_cache(
    page_size: int = 1,
    is_eagle: bool = False,
    disable: bool = False,
    rotator=None,
    capacity: int = 10**6,
):
    pool = FakeReqToTokenPool()
    allocator = FakeAllocator(capacity=capacity)
    cache = RadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            is_eagle=is_eagle,
        )
    )
    cache.kv_rotator = rotator
    return cache, pool, allocator


def make_req(rid: str, blocks, req_pool_idx: int = 0) -> Req:
    ids = [tok for block in blocks for tok in block]
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=ids,
        sampling_params=SamplingParams(max_new_tokens=8),
        sub_context_ends=list(itertools.accumulate(len(block) for block in blocks)),
    )
    req.req_pool_idx = req_pool_idx
    return req


def prefill(cache, pool, req: Req, first_slot: int) -> None:
    """Match, hand the request its KV slots, then insert the prompt."""
    req.init_next_round_input(cache)
    n = len(req.origin_input_ids)
    slots = torch.arange(first_slot, first_slot + n, dtype=torch.int64)
    # The slots the match already reused stay where the tree put them; only the
    # freshly computed tail gets new slots, exactly as the allocator would do it.
    reused = len(req.prefix_indices)
    pool.req_to_token[req.req_pool_idx, :reused] = req.prefix_indices
    pool.req_to_token[req.req_pool_idx, reused:n] = slots[reused:n]
    # `prepare_for_extend` does this: once req_to_token holds them, rotated copies are
    # freed from there, and keeping the request's own claim would free them twice.
    req.sub_context_rotated_slots = None
    req.fill_ids = list(req.origin_input_ids)
    cache.cache_unfinished_req(req)


def decode_and_finish(cache, pool, req: Req, output_ids, first_slot: int) -> None:
    """Append generated tokens with their own slots, then finish the request."""
    n = len(req.origin_input_ids)
    req.output_ids = list(output_ids)
    pool.req_to_token[req.req_pool_idx, n : n + len(output_ids)] = torch.arange(
        first_slot, first_slot + len(output_ids), dtype=torch.int64
    )
    req.kv_committed_len = n + len(output_ids)
    cache.cache_finished_req(req)


class TestSubContextSupportGate(unittest.TestCase):
    """Which caches can serve the split."""

    def test_plain_radix_cache_supports(self):
        cache, _, _ = make_cache()
        self.assertTrue(cache.supports_sub_contexts())

    def test_unsupported_configurations(self):
        for kwargs in ({"page_size": 4}, {"is_eagle": True}, {"disable": True}):
            with self.subTest(**kwargs):
                cache, _, _ = make_cache(**kwargs)
                self.assertFalse(cache.supports_sub_contexts())

    def test_chunk_cache_does_not_support(self):
        cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=FakeReqToTokenPool(),
                token_to_kv_pool_allocator=FakeAllocator(),
                page_size=1,
            )
        )
        self.assertFalse(cache.supports_sub_contexts())

    def test_unsupported_cache_falls_back_instead_of_raising(self):
        """ChunkCache has no root_node: the split must not be attempted on it."""
        cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=FakeReqToTokenPool(),
                token_to_kv_pool_allocator=FakeAllocator(),
                page_size=1,
            )
        )
        req = make_req("r", [[1, 2, 3], [4, 5]])
        req.init_next_round_input(cache)  # used to raise AttributeError
        self.assertEqual(len(req.prefix_indices), 0)

    def test_page_size_gt_1_serves_the_split_request_as_a_plain_one(self):
        """The prompt is stored as a plain request, and a second request with the same
        prompt hits it."""
        cache, pool, _ = make_cache(page_size=4)
        req = make_req("r", [[1, 2, 3, 4], [5, 6, 7, 8]])
        prefill(cache, pool, req, first_slot=100)
        self.assertIsNone(req.sub_context_layout)
        again = make_req("r2", [[1, 2, 3, 4], [5, 6, 7, 8]])
        again.init_next_round_input(cache)
        self.assertGreater(len(again.prefix_indices), 0)


class TestCanonicalPosition(unittest.TestCase):
    """A node remembers the absolute position its KV was computed at."""

    def test_ordinary_insert_positions_follow_the_path(self):
        cache, _, _ = make_cache()
        cache.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=torch.tensor([10, 11, 12]))
        )
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]))
        ).last_device_node
        self.assertEqual(node.canonical_position, 0)

    def test_split_divides_the_position(self):
        cache, _, _ = make_cache()
        cache.insert(
            InsertParams(key=RadixKey([7, 8, 9]), value=torch.tensor([70, 80, 90]))
        )
        # Matching a shorter key splits the node; both halves keep a true position.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8])))
        self.assertEqual(m.last_device_node.canonical_position, 0)
        tail = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8, 9])))
        self.assertEqual(tail.last_device_node.canonical_position, 2)


class TestSubContextTruncation(unittest.TestCase):
    """Auto-truncation must not leave the block ends describing a longer prompt."""

    def setUp(self):
        from sglang.srt.mem_cache.subctx_index import clip_block_ends

        self.clip = clip_block_ends

    def test_untouched_prompt_is_passed_through(self):
        self.assertEqual(self.clip([2, 4], 4, "r"), [2, 4])

    def test_trailing_block_is_clipped(self):
        self.assertEqual(self.clip([2, 4], 3, "r"), [2, 3])

    def test_blocks_past_the_cut_are_dropped(self):
        self.assertEqual(self.clip([2, 4], 2, "r"), [2])
        self.assertEqual(self.clip([2, 4, 6, 8], 5, "r"), [2, 4, 5])

    def test_a_cut_inside_the_first_block_leaves_one_block(self):
        self.assertEqual(self.clip([3, 6], 1, "r"), [1])

    def test_split_is_dropped_when_nothing_is_left(self):
        self.assertIsNone(self.clip([2, 4], 0, "r"))

    def test_blocks_shorter_than_the_prompt_drop_the_split(self):
        self.assertIsNone(self.clip([2], 5, "r"))


class TestSubContextBlockEnds(unittest.TestCase):
    """The tokenizer manager accepts only block ends that tile the prompt."""

    def setUp(self):
        from sglang.srt.mem_cache.subctx_index import valid_block_ends

        self.valid = valid_block_ends

    def test_increasing_ends_at_the_prompt_length_are_accepted(self):
        self.assertTrue(self.valid([2, 4], 4))
        self.assertTrue(self.valid([4], 4))
        self.assertTrue(self.valid((2, 4), 4))

    def test_ends_short_or_past_the_prompt_are_rejected(self):
        self.assertFalse(self.valid([2, 3], 4))
        self.assertFalse(self.valid([2, 5], 4))

    def test_an_empty_block_is_rejected(self):
        self.assertFalse(self.valid([2, 2, 4], 4))
        self.assertFalse(self.valid([0, 4], 4))

    def test_decreasing_ends_are_rejected(self):
        self.assertFalse(self.valid([3, 2, 4], 4))


def tree_slot_owners(cache):
    """Every slot the tree holds, and the ``extra_key`` scopes that hold it."""
    owners = {}
    stack = [(child, key[0] if isinstance(key, tuple) else None)
             for key, child in cache.root_node.children.items()]
    while stack:
        node, scope = stack.pop()
        for slot in node.value.tolist():
            owners.setdefault(slot, set()).add(scope)
        for key, child in node.children.items():
            stack.append((child, scope))
    return owners


def wide_pool(pool, width: int = 512):
    """The default fake row is 64 tokens; these prompts are longer."""
    pool.req_to_token = torch.zeros((4, width), dtype=torch.int64)
    return pool


def prefill_chained(cache, pool, req: Req, first_slot: int) -> None:
    """Match, write req_to_token as the sparse prefill does, then cache the prompt.

    Takes the scheduler's lock on ``last_node`` as the prefill adder does.
    """
    req.init_next_round_input(cache)
    cache.inc_lock_ref(req.last_node)
    n = len(req.fill_ids)
    row = req.req_pool_idx
    if req.sub_context_layout is None:
        reused = len(req.prefix_indices)
        pool.req_to_token[row, :reused] = req.prefix_indices
        pool.req_to_token[row, reused:n] = torch.arange(
            first_slot, first_slot + n - reused, dtype=torch.int64
        )
    else:
        for start, end, slots in req.sub_context_layout:
            pool.req_to_token[row, start:end] = slots
        fresh = req.sub_context_fresh_positions()
        pool.req_to_token[row, fresh] = torch.arange(
            first_slot, first_slot + len(fresh), dtype=torch.int64
        )
    req.sub_context_rotated_slots = None  # as prepare_for_extend does
    cache.cache_unfinished_req(req)


def chain_slots(cache, ids, extra_key=None):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(list(ids), extra_key))
    ).device_indices.tolist()


class TestChainedIndex(unittest.TestCase):
    """A request is matched and stored as one chain like a plain request; the index
    records where each registered chunk sits in the tree."""

    HEAD_A = list(range(1, 41))
    HEAD_B = list(range(900, 910))
    HEAD_C = list(range(800, 840))
    TAIL = list(range(500, 540))
    MORE = list(range(700, 740))

    def setUp(self):
        self._saved_ratios = subctx_config.TOPK_RATIOS

    def tearDown(self):
        subctx_config.TOPK_RATIOS = self._saved_ratios

    def _cache(self):
        cache, pool, allocator = make_cache(rotator=FakeRotator())
        wide_pool(pool)
        cache.sub_context_index = SubContextIndex(min_chunk_tokens=32)
        return cache, pool, allocator

    def _seeded(self, first_blocks, output=()):
        cache, pool, allocator = self._cache()
        first = make_req("r1", first_blocks)
        prefill_chained(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, list(output), first_slot=400)
        return cache, pool, allocator

    def _cid(self, ids, extra_key=None):
        return chunk_id(ids, extra_key)

    def test_a_request_is_stored_as_one_chain_with_its_blocks_located(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])

        self.assertEqual(
            chain_slots(cache, self.HEAD_A + self.TAIL), list(range(100, 180))
        )
        # No per-block namespace.
        self.assertEqual(chain_slots(cache, self.TAIL, self._cid(self.TAIL)), [])
        self.assertEqual(len(cache.sub_context_index), 2)
        slots, _node, start = cache.chain_chunk_slots(self._cid(self.TAIL))
        self.assertEqual(start, 40)
        self.assertEqual(slots.tolist(), list(range(140, 180)))
        self.assertEqual(cache.protected_size(), 0)

    def test_a_block_behind_a_miss_is_copied_rotated_and_computed_around(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req("r2", [self.HEAD_B, self.TAIL], req_pool_idx=1)
        second.init_next_round_input(cache)
        self.addCleanup(second.release_sub_context_rotated_slots, cache)
        self.addCleanup(second.release_sub_context_match_locks, cache)

        # Computed at 40, reused at 10; the last token is left for prefill.
        self.assertEqual(
            [(s, e) for s, e, _ in second.sub_context_layout], [(10, 49)]
        )
        src, dst, delta = cache.kv_rotator.calls[-1]
        self.assertEqual(src, list(range(140, 179)))
        self.assertEqual(delta, -30)
        self.assertEqual(second.sub_context_layout[0][2].tolist(), dst)
        self.assertEqual(second.sub_context_exact_len, 0)
        self.assertEqual(second.sub_context_rotated, 39)
        self.assertEqual(
            second.sub_context_fresh_positions(), list(range(0, 10)) + [49]
        )

    def test_the_plain_prefix_is_exact_and_not_scored(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])
        subctx_config.TOPK_RATIOS = (0.5,)
        second = make_req(
            "r2", [self.HEAD_A, self.MORE + self.TAIL], req_pool_idx=1
        )
        second.init_next_round_input(cache)
        self.addCleanup(second.release_sub_context_rotated_slots, cache)
        self.addCleanup(second.release_sub_context_match_locks, cache)

        self.assertEqual(
            [(s, e) for s, e, _ in second.sub_context_layout], [(0, 40), (80, 119)]
        )
        self.assertEqual(second.sub_context_layout[0][2].tolist(), list(range(100, 140)))
        self.assertEqual(second.sub_context_exact_len, 40)
        self.assertEqual(second.cache_protected_len, 40)
        self.assertEqual(second.sub_context_reused_positions(), list(range(80, 119)))
        self.assertEqual(second.sub_context_reused_ours(), [True] * 39)
        self.assertEqual(second.sub_context_topk_counts([0.5, 0.25]), [19, 9])

    def test_a_grown_conversation_is_a_plain_prefix_hit(self):
        cache, _, allocator = self._seeded([self.HEAD_A, self.TAIL])
        grown = make_req(
            "r2", [self.HEAD_A, self.TAIL + self.MORE], req_pool_idx=1
        )
        grown.init_next_round_input(cache)

        self.assertIsNone(grown.sub_context_layout)
        self.assertEqual(grown.prefix_indices.tolist(), list(range(100, 180)))
        self.assertEqual(grown.cache_protected_len, 80)
        self.assertEqual(cache.kv_rotator.calls, [])
        self.assertEqual(allocator.allocated, [])

    def test_a_run_at_its_own_position_on_another_chain_is_copied(self):
        cache, pool, allocator = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req("r2", [self.HEAD_C, self.TAIL], req_pool_idx=1)
        prefill_chained(cache, pool, second, first_slot=5000)
        decode_and_finish(cache, pool, second, [], first_slot=6000)

        self.assertEqual(cache.kv_rotator.calls, [], "nothing to rotate at delta zero")
        self.assertEqual(len(cache.kv_rotator.copies), 1)
        src, dst = cache.kv_rotator.copies[0]
        self.assertEqual(src, list(range(140, 179)))
        owners = tree_slot_owners(cache)
        self.assertEqual(len(owners), 80 + 80, "a slot is held by two nodes")
        self.assertEqual(set(allocator.freed) & set(owners), set())
        self.assertEqual(chain_slots(cache, self.HEAD_C + self.TAIL)[40:79], dst)
        self.assertEqual(cache.protected_size(), 0)

    def test_a_chunk_is_read_at_its_oldest_location_then_the_next(self):
        """Copies keep the first location ahead; once it is evicted the next one serves."""
        cache, pool, _ = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req("r2", [self.HEAD_B, self.TAIL], req_pool_idx=1)
        prefill_chained(cache, pool, second, first_slot=5000)
        decode_and_finish(cache, pool, second, [], first_slot=6000)

        cid = self._cid(self.TAIL)
        self.assertEqual(
            [start for _node, start in cache.sub_context_index.locations(cid)], [40, 10]
        )
        slots, _node, start = cache.chain_chunk_slots(cid)
        self.assertEqual((start, slots.tolist()), (40, list(range(140, 180))))

        # Reading the chunk refreshed the first chain; touch the second after it, so
        # the first is the least recently used leaf.
        second_slots = chain_slots(cache, self.HEAD_B + self.TAIL)
        cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(chain_slots(cache, self.HEAD_A + self.TAIL), [])
        slots, _node, start = cache.chain_chunk_slots(cid)
        self.assertEqual(start, 10)
        self.assertEqual(slots.tolist(), second_slots[10:50])
        self.assertEqual(len(cache.sub_context_index.locations(cid)), 1)

    def test_a_refused_request_drops_its_runs(self):
        """Not admitted: its locks and copies go back. With no room and nothing else
        running, later matches take the prefix only."""
        cache, _, allocator = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req(
            "r2", [self.HEAD_A, self.MORE + self.TAIL], req_pool_idx=1
        )
        second.init_next_round_input(cache)
        copies = second.sub_context_layout[1][2].tolist()
        self.assertGreater(cache.protected_size(), 0)

        self.assertIsNotNone(second.sub_context_tally)
        self.assertFalse(second.release_sub_context_scan(cache))
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(sorted(allocator.freed), copies)
        self.assertFalse(second.sub_context_index_declined)
        self.assertIsNone(second.sub_context_tally, "a refused match is not counted")

        second.init_next_round_input(cache)
        self.assertIsNotNone(second.sub_context_layout, "a plain refusal retries the scan")
        self.assertTrue(second.release_sub_context_scan(cache, no_room=True))
        self.assertTrue(second.sub_context_index_declined)
        self.assertFalse(second.release_sub_context_scan(cache, no_room=True))
        self.assertEqual(cache.sub_context_index.no_room, 1)

        second.init_next_round_input(cache)
        self.assertIsNone(second.sub_context_layout)
        self.assertEqual(second.prefix_indices.tolist(), list(range(100, 140)))
        self.assertEqual(cache.protected_size(), 0)

    def test_eviction_forgets_the_chunks_it_removes(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])
        cache.evict(EvictParams(num_tokens=10**6))
        self.assertEqual(len(cache.sub_context_index), 0)
        self.assertEqual(cache._chunks_at, {})

        second = make_req("r2", [self.HEAD_B, self.TAIL], req_pool_idx=1)
        second.init_next_round_input(cache)
        self.assertIsNone(second.sub_context_layout)

    def test_a_split_moves_a_chunk_to_the_node_now_holding_its_end(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])
        # Splits the 80-token node at 50: [0, 50) and [50, 80).
        chain_slots(cache, self.HEAD_A + self.TAIL[:10] + [1, 2, 3])

        node, start = cache.sub_context_index.location(self._cid(self.HEAD_A))
        self.assertEqual((node.canonical_position, len(node.key), start), (0, 50, 0))
        slots, _, _ = cache.chain_chunk_slots(self._cid(self.HEAD_A))
        self.assertEqual(slots.tolist(), list(range(100, 140)))
        slots, tail_node, start = cache.chain_chunk_slots(self._cid(self.TAIL))
        self.assertEqual((tail_node.canonical_position, start), (50, 40))
        self.assertEqual(slots.tolist(), list(range(140, 180)))

        # Evicting the lower half keeps the chunk that ends above it.
        cache.evict(EvictParams(num_tokens=30))
        self.assertIsNone(cache.sub_context_index.location(self._cid(self.TAIL)))
        self.assertIsNotNone(cache.chain_chunk_slots(self._cid(self.HEAD_A)))

    def test_the_reply_is_registered_with_the_last_block(self):
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL], output=self.MORE)
        reply = self._cid(self.TAIL + self.MORE)
        slots, _node, start = cache.chain_chunk_slots(reply)
        self.assertEqual(start, 40)
        self.assertEqual(slots.tolist(), list(range(140, 180)) + list(range(400, 440)))

        second = make_req(
            "r2", [self.HEAD_B, self.TAIL + self.MORE + [7, 7]],
            req_pool_idx=1,
        )
        second.init_next_round_input(cache)
        self.addCleanup(second.release_sub_context_rotated_slots, cache)
        self.addCleanup(second.release_sub_context_match_locks, cache)
        self.assertEqual([(s, e) for s, e, _ in second.sub_context_layout], [(10, 90)])
        self.assertEqual(cache.kv_rotator.calls[-1][2], -30)

    def test_a_layout_too_big_for_one_pass_keeps_the_prefix_only(self):
        """Known before copying: nothing is locked or copied."""
        cache, _, allocator = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req(
            "r2", [self.HEAD_A, self.MORE + self.TAIL], req_pool_idx=1
        )
        second._fits_one_pass = lambda to_compute: False
        second.init_next_round_input(cache)

        self.assertIsNone(second.sub_context_layout)
        self.assertEqual(second.sub_context_exact_len, 0)
        self.assertEqual(second.prefix_indices.tolist(), list(range(100, 140)))
        self.assertEqual(cache.kv_rotator.calls, [])
        self.assertEqual(allocator.allocated, [])
        self.assertIsNone(second.sub_context_match_nodes)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(second.sub_context_tally[4:], (1, 39))
        self.assertEqual(second.sub_context_moved, 39)

    def test_a_layout_left_too_big_by_failed_copies_keeps_the_prefix_only(self):
        """The check after copying: the copies made go back."""
        cache, _, allocator = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req(
            "r2", [self.HEAD_A, self.MORE + self.TAIL], req_pool_idx=1
        )
        second._sub_context_layout_fits_one_pass = lambda: False
        second.init_next_round_input(cache)

        self.assertIsNone(second.sub_context_layout)
        self.assertEqual(second.prefix_indices.tolist(), list(range(100, 140)))
        self.assertEqual(sorted(allocator.freed), sorted(cache.kv_rotator.calls[-1][1]))
        self.assertIsNone(second.sub_context_match_nodes)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(second.sub_context_tally[4:], (1, 39))
        self.assertEqual((second.sub_context_moved, second.sub_context_rotated), (39, 0))

    def test_a_run_out_of_rotation_range_is_never_locked(self):
        cache, pool, allocator = make_cache(rotator=FakeRotator(max_delta=5))
        wide_pool(pool)
        cache.sub_context_index = SubContextIndex(min_chunk_tokens=32)
        first = make_req("r1", [self.HEAD_A, self.TAIL])
        prefill_chained(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=400)

        second = make_req("r2", [self.HEAD_B, self.TAIL], req_pool_idx=1)
        second.init_next_round_input(cache)
        self.assertIsNone(second.sub_context_layout)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(allocator.allocated, [])
        self.assertEqual((second.sub_context_moved, second.sub_context_discarded), (39, 39))

    def test_a_request_finishing_at_prefill_is_stored_and_registered(self):
        """No cache_unfinished_req pass: the finish stores the chain and registers every
        block but the last, which is stored with the reply."""
        cache, pool, _ = self._cache()
        gen = [9, 9, 9]
        first = make_req("r1", [self.HEAD_A, self.TAIL, gen])
        first.init_next_round_input(cache)
        cache.inc_lock_ref(first.last_node)
        n = len(first.fill_ids)
        pool.req_to_token[0, :n] = torch.arange(100, 100 + n, dtype=torch.int64)
        first.sub_context_rotated_slots = None
        first.output_ids = [7]
        pool.req_to_token[0, n] = 400
        first.kv_committed_len = n + 1
        cache.cache_finished_req(first)

        self.assertEqual(
            chain_slots(cache, self.HEAD_A + self.TAIL + gen + [7]),
            list(range(100, 183)) + [400],
        )
        self.assertIsNotNone(cache.chain_chunk_slots(self._cid(self.TAIL)))
        self.assertEqual(cache.protected_size(), 0)

        second = make_req("r2", [self.HEAD_B, self.TAIL, gen], req_pool_idx=1)
        second.init_next_round_input(cache)
        self.addCleanup(second.release_sub_context_rotated_slots, cache)
        self.addCleanup(second.release_sub_context_match_locks, cache)
        self.assertEqual([(s, e) for s, e, _ in second.sub_context_layout], [(10, 50)])

    def test_a_request_is_counted_once(self):
        """Matched again after it ran (a retraction): the tallies do not grow."""
        cache, _, _ = self._seeded([self.HEAD_A, self.TAIL])
        index = cache.sub_context_index
        second = make_req("r2", [self.HEAD_B, self.TAIL], req_pool_idx=1)
        second.init_next_round_input(cache)
        self.addCleanup(second.release_sub_context_rotated_slots, cache)
        self.addCleanup(second.release_sub_context_match_locks, cache)
        before = index.prompt_tokens
        second.count_sub_context_match(index)
        self.assertEqual(index.prompt_tokens - before, 50)
        self.assertEqual(index.reused_tokens, 39)

        second.release_sub_context_match_locks(cache)
        second.release_sub_context_rotated_slots(cache)
        second.init_next_round_input(cache)
        second.count_sub_context_match(index)
        self.assertEqual(index.prompt_tokens - before, 50)
        self.assertIsNone(second.sub_context_tally)

    def test_a_cache_salt_scopes_the_scan(self):
        cache, pool, _ = self._cache()
        ids = self.HEAD_A + self.TAIL
        first = Req(
            rid="r1", origin_input_text="", origin_input_ids=ids,
            sampling_params=SamplingParams(max_new_tokens=8),
            sub_context_ends=[len(self.HEAD_A), len(ids)],
            extra_key="salt-a",
        )
        first.req_pool_idx = 0
        prefill_chained(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=400)
        self.assertEqual(chain_slots(cache, ids, "salt-a"), list(range(100, 180)))

        second = Req(
            rid="r2", origin_input_text="",
            origin_input_ids=self.HEAD_B + self.TAIL,
            sampling_params=SamplingParams(max_new_tokens=8),
            sub_context_ends=[len(self.HEAD_B), len(self.HEAD_B + self.TAIL)],
            extra_key="salt-b",
        )
        second.req_pool_idx = 1
        second.init_next_round_input(cache)
        self.assertIsNone(second.sub_context_layout)
        self.assertEqual(len(second.prefix_indices), 0)

    def test_every_lock_is_released_and_no_slot_is_lost(self):
        """The finish audit sees every slot the request held freed or in the tree."""
        from sglang.srt.mem_cache import radix_cache

        cache, pool, allocator = self._seeded([self.HEAD_A, self.TAIL])
        second = make_req(
            "r2", [self.HEAD_A, self.MORE + self.TAIL], req_pool_idx=1
        )
        prefill_chained(cache, pool, second, first_slot=5000)
        self.assertIsNone(second.sub_context_match_nodes)
        decode_and_finish(cache, pool, second, [7, 8], first_slot=6000)
        self.assertEqual(cache.protected_size(), 0)

        held = set(pool.req_to_token[1, :122].tolist())
        owners = tree_slot_owners(cache)
        self.assertEqual(held - set(owners) - set(allocator.freed), set(), "lost slots")
        self.assertEqual(held & set(owners) & set(allocator.freed), set(), "freed but held")


if __name__ == "__main__":
    unittest.main()
