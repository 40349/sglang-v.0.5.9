"""
Unit tests for the sub-context path of RadixCache.

A sub-context request has its prompt split into ordered blocks, each matched and
inserted under its own ``extra_key`` namespace. These drive the full lifecycle on
CPU tensors -- match, insert, finish -- and cover its invariants:

- read and write must agree on whether the cache can serve the split at all,
- a namespace must stay pinned to one absolute position in the prompt,
- a hit at another position is rotated to where it is reused, or dropped -- never
  stitched as it is, and never freed while a node still points at it,
- a finished request's generated tokens extend its last block's namespace.

Usage:
    python test_sub_context_unit.py
    python -m pytest test_sub_context_unit.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import unittest

import torch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

SYS_KEY = "system_prompt_key"
MSG_KEY = "messages_key"


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
        # Capacity is unbounded here, so "how many slots have come back" is the only
        # meaningful reading -- and it is what the finish-path leak audit needs, since
        # that only ever looks at the delta across one call.
        return len(self.freed)


class FakeRotator:
    """Stands in for the Triton kernel: records the deltas, moves no real KV.

    The kernel's arithmetic is covered on GPU by ``test_rotate_kv.py``; what these
    tests check is the bookkeeping around it -- which slots get allocated, what ends up
    in prefix_indices, and that everything is freed exactly once.
    """

    def __init__(self, max_delta: int = 10**6):
        self.max_delta = max_delta
        self.calls = []
        self.stages = []

    def can_rotate(self, delta: int) -> bool:
        return delta != 0 and abs(delta) <= self.max_delta

    def rotate_into(self, dst_loc, src_loc, delta, stage="subctx_rotate"):
        # `stage` only names the host timer the real rotator charges the call to.
        # Recorded so a test can tell a read-path rotation from a finish-path one.
        self.calls.append((src_loc.tolist(), dst_loc.tolist(), delta))
        self.stages.append(stage)


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


def make_req(rid: str, blocks, keys, req_pool_idx: int = 0) -> Req:
    ids = [tok for block in blocks for tok in block]
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=ids,
        sampling_params=SamplingParams(max_new_tokens=8),
        sub_context_ids=blocks,
        sub_context_extra_keys=keys,
    )
    req.req_pool_idx = req_pool_idx
    return req


def prefill(cache, pool, req: Req, first_slot: int) -> None:
    """Match, hand the request its KV slots, then insert per namespace."""
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


def prefill_chunk(cache, pool, req: Req, upto: int, first_slot: int) -> None:
    """One chunked-prefill pass covering origin_input_ids[:upto].

    Mirrors `prefill`, but stops short of the prompt so the next pass picks up where
    `cache_unfinished_req` left `prefix_indices` -- which is the whole point of the
    Stage 2 path, since the scheduler does not re-match a chunked request.
    """
    reused = len(req.prefix_indices)
    pool.req_to_token[req.req_pool_idx, :reused] = req.prefix_indices
    fresh = torch.arange(first_slot, first_slot + (upto - reused), dtype=torch.int64)
    pool.req_to_token[req.req_pool_idx, reused:upto] = fresh
    req.sub_context_rotated_slots = None  # as prepare_for_extend does
    req.fill_ids = list(req.origin_input_ids[:upto])
    cache.cache_unfinished_req(req, chunked=True)


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
    """Read and write path must agree on whether the split can be served."""

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
        req = make_req("r", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        req.init_next_round_input(cache)  # used to raise AttributeError
        self.assertEqual(len(req.prefix_indices), 0)

    def test_page_size_gt_1_uses_the_default_namespace_end_to_end(self):
        """No namespace is written, so none may be matched either."""
        cache, pool, _ = make_cache(page_size=4)
        req = make_req("r", [[1, 2, 3, 4], [5, 6, 7, 8]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, req, first_slot=100)
        self.assertIsNone(req.sub_context_last_nodes)
        # The prompt went into the default namespace, and a second request with the
        # same prompt hits it there.
        again = make_req("r2", [[1, 2, 3, 4], [5, 6, 7, 8]], [SYS_KEY, MSG_KEY])
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

    def test_block_insert_records_its_offset(self):
        cache, _, _ = make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([7, 8], MSG_KEY),
                value=torch.tensor([70, 80]),
                canonical_position=3,
            )
        )
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8], MSG_KEY)))
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 2), 3)

    def test_split_divides_the_position(self):
        cache, _, _ = make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([7, 8, 9], MSG_KEY),
                value=torch.tensor([70, 80, 90]),
                canonical_position=3,
            )
        )
        # Matching a shorter key splits the node; both halves keep a true position.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8], MSG_KEY)))
        self.assertEqual(m.last_device_node.canonical_position, 3)
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 2), 3)
        tail = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8, 9], MSG_KEY)))
        self.assertEqual(tail.last_device_node.canonical_position, 5)
        self.assertEqual(cache.matched_canonical_position(tail.last_device_node, 3), 3)

    def test_position_of_a_partial_match_is_the_chain_start(self):
        cache, pool, _ = make_cache()
        req = make_req("r1", [[1, 2, 3], [7, 8]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, req, first_slot=100)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7], MSG_KEY)))
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 1), 3)

    def test_no_match_has_no_position(self):
        cache, _, _ = make_cache()
        self.assertIsNone(cache.matched_canonical_position(None, 0))


class TestMovedBlocks(unittest.TestCase):
    """A block whose KV was computed elsewhere is dropped, not stitched."""

    def _seed_and_move(self):
        """Cache [7,8] at offset 3, then ask for it at offset 2."""
        cache, pool, allocator = make_cache()
        first = make_req("r1", [[1, 2, 3], [7, 8]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=200)
        second = make_req("r2", [[1, 2], [7, 8]], [SYS_KEY, MSG_KEY], req_pool_idx=1)
        return cache, pool, allocator, second

    def test_moved_hit_is_not_stitched(self):
        cache, _, _, second = self._seed_and_move()
        second.init_next_round_input(cache)
        # The head still hits at its own position and is still reused; only the
        # block whose KV was computed elsewhere is dropped.
        self.assertEqual(second.sub_context_match_lens, [2, 2])
        self.assertEqual(second.sub_context_match_positions, [0, 3])
        self.assertEqual(second.prefix_indices.tolist(), [100, 101])
        self.assertEqual(second.sub_context_moved, 2)
        self.assertEqual(second.sub_context_discarded, 2)
        second.release_sub_context_match_locks(cache)

    def test_moved_hit_stays_locked_for_a_later_rotation(self):
        """The dropped slots are the raw material Stage 1 rotates; keep them."""
        cache, _, _, second = self._seed_and_move()
        second.init_next_round_input(cache)
        self.assertIsNotNone(second.sub_context_match_indices[1])
        self.assertIsNotNone(second.sub_context_match_nodes[1])
        self.assertGreater(cache.protected_size(), 0)
        second.release_sub_context_match_locks(cache)

    def test_first_writer_keeps_the_tree_copy(self):
        cache, pool, _, second = self._seed_and_move()
        prefill(cache, pool, second, first_slot=300)
        # The namespace still holds the original copy at its original position.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8], MSG_KEY)))
        self.assertEqual(m.device_indices.tolist(), [103, 104])
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 2), 3)
        # The second request keeps its own slots for that block: not tree-owned,
        # so they are still pointed at by req_to_token and freed at finish.
        self.assertEqual(pool.req_to_token[1, 2:4].tolist(), [302, 303])
        self.assertEqual(second.cache_protected_len, 2)

    def test_moved_block_is_freed_once_at_finish(self):
        cache, pool, allocator, second = self._seed_and_move()
        prefill(cache, pool, second, first_slot=300)
        allocator.freed.clear()
        decode_and_finish(cache, pool, second, [9], first_slot=400)
        # Its own copy of [7,8] plus the generated token; the tree's copy untouched.
        self.assertEqual(sorted(allocator.freed), [302, 303, 400])
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey([7, 8], MSG_KEY))
            ).device_indices.tolist(),
            [103, 104],
        )

    def test_one_node_per_token_sequence(self):
        """No fragmentation: the moved request must not add a second copy."""
        cache, pool, _, second = self._seed_and_move()
        before = cache.total_size()
        prefill(cache, pool, second, first_slot=300)
        # [1,2] was already there as a prefix (it only splits a node) and [7,8] was
        # refused, so the tree still holds exactly one copy of each token sequence.
        self.assertEqual(cache.total_size(), before)


class TestRotatedBlocks(unittest.TestCase):
    """A displaced hit is copied to fresh slots and rotated, instead of dropped."""

    TOOLS_KEY = "tools_key"

    def _seed(self, rotator=None, capacity: int = 10**6):
        """Cache SYS[1,2,3]@0, TOOLS[7,8]@3, MSG[20,21]@5."""
        cache, pool, allocator = make_cache(rotator=rotator, capacity=capacity)
        first = make_req(
            "r1",
            [[1, 2, 3], [7, 8], [20, 21]],
            [SYS_KEY, self.TOOLS_KEY, MSG_KEY],
        )
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=200)
        return cache, pool, allocator

    def _shifted_req(self):
        """TOOLS[7,8] now sits at offset 2 instead of 3, so delta is -1."""
        return make_req(
            "r2",
            [[1, 2], [7, 8], [40, 41]],
            [SYS_KEY, self.TOOLS_KEY, MSG_KEY],
            req_pool_idx=1,
        )

    def test_displaced_hit_is_rotated_into_the_prefix(self):
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted_req()
        req.init_next_round_input(cache)

        # The head is reused where it is; the displaced block is rotated by
        # offset - canonical = 2 - 3 and stitched from its fresh copy.
        self.assertEqual(rotator.calls, [([103, 104], [900, 901], -1)])
        self.assertEqual(req.prefix_indices.tolist(), [100, 101, 900, 901])
        self.assertEqual(req.sub_context_rotated, 2)
        self.assertEqual(req.sub_context_moved, 0)
        req.release_sub_context_match_locks(cache)
        req.release_sub_context_rotated_slots(cache)

    def test_without_a_rotator_the_hit_is_still_dropped(self):
        cache, pool, allocator = self._seed(rotator=None)
        req = self._shifted_req()
        req.init_next_round_input(cache)

        self.assertEqual(req.prefix_indices.tolist(), [100, 101])
        self.assertEqual(req.sub_context_rotated, 0)
        self.assertEqual(req.sub_context_moved, 2)
        req.release_sub_context_match_locks(cache)

    def test_a_full_pool_falls_back_to_dropping(self):
        """An allocator that cannot hand out slots must not change the outcome."""
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator, capacity=0)
        req = self._shifted_req()
        req.init_next_round_input(cache)

        self.assertEqual(rotator.calls, [])
        self.assertEqual(req.prefix_indices.tolist(), [100, 101])
        self.assertEqual(req.sub_context_rotated, 0)
        self.assertEqual(req.sub_context_moved, 2)
        req.release_sub_context_match_locks(cache)

    def test_rotated_copy_is_never_inserted(self):
        """The tree keeps one rotation per token sequence: the first writer's."""
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted_req()
        prefill(cache, pool, req, first_slot=300)

        hit = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([7, 8], self.TOOLS_KEY))
        )
        self.assertEqual(hit.device_indices.tolist(), [103, 104])
        self.assertEqual(cache.matched_canonical_position(hit.last_device_node, 2), 3)
        # ... and the block after the un-inserted one still gets cached, which a
        # `break` on the position conflict would have prevented.
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey([40, 41], MSG_KEY))
            ).device_indices.tolist(),
            [304, 305],
        )
        self.assertEqual(req.sub_context_tree_owned, [True, False, True])

    def test_restitch_clears_stale_ownership(self):
        """A re-scheduled request must not inherit last attempt's tree ownership.

        Retraction frees the request's KV, so a block that was tree-owned then may be
        this request's to free now; a stale True would leak it at finish.
        """
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted_req()
        prefill(cache, pool, req, first_slot=300)
        self.assertEqual(req.sub_context_tree_owned, [True, False, True])

        req.init_next_round_input(cache)  # as a retracted request would re-stitch
        self.assertIsNone(req.sub_context_tree_owned)
        req.release_sub_context_match_locks(cache)
        req.release_sub_context_rotated_slots(cache)

    def test_rotated_copy_is_freed_exactly_once(self):
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted_req()
        prefill(cache, pool, req, first_slot=300)
        allocator.freed.clear()
        decode_and_finish(cache, pool, req, [9], first_slot=400)

        # Only the rotated copy of [7,8], which the re-file handed back as a duplicate
        # of what the namespace already held. The generated token 400 is NOT here: the
        # re-file closed the hole [7,8] left, so the reply could be cached. The tree
        # owns the head and the tail block, and nothing is freed twice.
        self.assertEqual(sorted(allocator.freed), [900, 901])
        self.assertEqual(len(allocator.freed), len(set(allocator.freed)))

    def test_append_leaves_a_token_to_compute(self):
        """A pass with nothing to compute is not a valid batch, so the append stops
        one token short of the prompt exactly as the stitch does."""
        import sglang.srt.utils.subctx_config as subctx_config

        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        original = subctx_config.ROTATE_ACROSS_RECOMPUTE
        subctx_config.ROTATE_ACROSS_RECOMPUTE = True
        try:
            # Six prompt tokens, and the rotatable block is the last one: taking it
            # whole would cover the lot.
            req = make_req(
                "r2",
                [[1, 2, 9, 10], [7, 8]],
                [SYS_KEY, self.TOOLS_KEY],
                req_pool_idx=1,
            )
            req.init_next_round_input(cache)
            self.assertEqual(req.sub_context_next_boundary, 4)

            prefill_chunk(cache, pool, req, upto=4, first_slot=300)

            # One of the block's two tokens is rotated in; the other is recomputed.
            self.assertEqual(rotator.calls, [([103], [900], 1)])
            self.assertEqual(req.sub_context_rotated, 1)
            # The token the cap refused is a genuine drop, and it is reported by this
            # pass -- the one that gave up on it -- not by the stitch.
            self.assertEqual(req.sub_context_moved, 1)
            self.assertEqual(req.sub_context_deferred_moved, 0)
            self.assertEqual(len(req.prefix_indices), 5)

            req.init_next_round_input()
            self.assertEqual(req.extend_input_len, 1)
        finally:
            subctx_config.ROTATE_ACROSS_RECOMPUTE = original

    def test_block_after_a_recompute_is_rotated_in(self):
        """Stage 2: block 0 partially hits, and block 1 is still reused after it.

        The stitch cannot take block 1 -- the tokens before it do not exist yet -- so
        the chunk is cut at block 1's offset, and once the gap is computed the block is
        rotated onto the end of the prefix instead of being recomputed.
        """
        import sglang.srt.utils.subctx_config as subctx_config

        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        original = subctx_config.ROTATE_ACROSS_RECOMPUTE
        subctx_config.ROTATE_ACROSS_RECOMPUTE = True
        try:
            req = make_req(
                "r2",
                [[1, 2, 9, 10], [7, 8], [22, 23]],
                [SYS_KEY, self.TOOLS_KEY, MSG_KEY],
                req_pool_idx=1,
            )
            req.init_next_round_input(cache)

            # [1,2] of the head hits, [9,10] must be computed, so the stitch stops at
            # 2 and the boundary is where TOOLS starts in THIS prompt.
            self.assertEqual(req.prefix_indices.tolist(), [100, 101])
            self.assertEqual(req.sub_context_next_boundary, 4)
            # Held back, not reported as a drop: the append is about to attempt it,
            # and a pass that reported the drop would have to un-report it later.
            self.assertEqual(req.sub_context_moved, 0)
            self.assertEqual(req.sub_context_deferred_moved, 2)

            # The chunk the scheduler would cut at that boundary.
            prefill_chunk(cache, pool, req, upto=4, first_slot=300)

            # TOOLS was rotated by 4 - 3 and appended, so the prefix now runs past it.
            self.assertEqual(rotator.calls, [([103, 104], [900, 901], 1)])
            self.assertEqual(
                req.prefix_indices.tolist(), [100, 101, 300, 301, 900, 901]
            )
            self.assertEqual(req.sub_context_rotated, 2)
            # The append took all of it, so nothing was dropped after all.
            self.assertEqual(req.sub_context_moved, 0)
            self.assertEqual(req.sub_context_deferred_moved, 0)
            # Only the head is tree-owned so far; the rotated block is this
            # request's, and MSG has not been reached by any chunk yet.
            self.assertEqual(req.cache_protected_len, 4)
            self.assertEqual(req.sub_context_tree_owned, [True, False, False])

            # The next pass picks up where this one left off: it has 2 tokens left.
            req.init_next_round_input()
            self.assertEqual(req.extend_input_len, 2)
        finally:
            subctx_config.ROTATE_ACROSS_RECOMPUTE = original


class TestSubContextLifecycle(unittest.TestCase):
    """Match -> per-namespace insert -> finish, and the reuse it enables."""

    def test_read_and_write_gates_agree(self):
        """The read path and the write paths must answer "is this request split?"
        identically, in every state that has ever pulled them apart.

        Both times this failed, the write gate carried a condition the read gate did
        not -- a length clause, then a `sub_context_last_nodes` check -- and the request
        was stitched out of the namespaces by one and filed under the DEFAULT namespace
        by the other. `serves_sub_contexts` is now the single gate; this holds the three
        call sites to it.
        """
        for label, page_size, mutate in (
            ("fresh request", 1, lambda r: None),
            # `fill_ids = origin_input_ids + output_ids`, so a retracted request comes
            # back longer than the prompt its blocks describe. That is what the write
            # gate's old length clause tripped on.
            ("retracted, fill_ids past the split prompt", 1,
             lambda r: r.output_ids.extend([61, 62, 63])),
            # No unfinished pass ran: the state a request finishing during prefill
            # arrives in.
            ("no unfinished pass ran", 1,
             lambda r: setattr(r, "sub_context_last_nodes", None)),
            # A cache that cannot serve the split must turn BOTH paths off, or matches
            # land in namespaces that nothing ever inserts into.
            ("cache does not support the split", 4, lambda r: None),
        ):
            with self.subTest(label):
                cache, pool, _ = make_cache(page_size=page_size)
                req = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
                mutate(req)
                req.init_next_round_input(cache)
                read_took_split = req.sub_context_match_lens is not None
                self.assertEqual(
                    read_took_split,
                    cache.serves_sub_contexts(req),
                    f"{label}: read path and write gate disagree",
                )

    def test_finishing_without_an_unfinished_pass_files_nothing_by_default_key(self):
        """A split request can reach finish without ever running an unfinished pass --
        it emitted its stop token during prefill, or was aborted while queued.

        `cache_finished_req` used to gate the sub-context branch on
        `sub_context_last_nodes`, which only that pass sets, so such a request fell
        through to the default branch and filed whatever `req_to_token` held under
        `req.extra_key` -- None. The stitch had just pointed that prefix at the
        namespace nodes' own slots, so the tree served them under two keys at once and
        evicting either handed a live slot back to the pool. Job 441's
        `dup_within_tree`, and a replay with ignore_eos cannot produce it because no
        request can finish at prefill.
        """
        cache, pool, allocator = make_cache()
        seed = make_req("r1", [[1, 2, 3], [7, 8]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, seed, first_slot=100)
        decode_and_finish(cache, pool, seed, [50], first_slot=200)

        req = make_req("r2", [[1, 2, 3], [7, 8]], [SYS_KEY, MSG_KEY], req_pool_idx=1)
        req.init_next_round_input(cache)
        n = len(req.origin_input_ids)
        reused = len(req.prefix_indices)
        self.assertGreater(reused, 0, "the stitch must have reused namespace slots")
        pool.req_to_token[req.req_pool_idx, :reused] = req.prefix_indices
        pool.req_to_token[req.req_pool_idx, reused:n] = torch.arange(
            300, 300 + (n - reused), dtype=torch.int64
        )
        req.sub_context_rotated_slots = None
        req.fill_ids = list(req.origin_input_ids)
        req.output_ids = []
        req.kv_committed_len = n
        self.assertIsNone(req.sub_context_last_nodes)

        freed_before = set(allocator.freed)
        cache.cache_finished_req(req)

        owners = {}
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is cache.root_node or node.value is None:
                continue
            for slot in node.value.tolist():
                owners.setdefault(slot, []).append(node.key.extra_key)

        self.assertNotIn(
            None,
            {k for keys in owners.values() for k in keys},
            "a split request must never file its slots under the default namespace",
        )
        self.assertEqual(
            {s: k for s, k in owners.items() if len(k) > 1}, {}, "no slot has two owners"
        )
        freed = set(allocator.freed) - freed_before
        self.assertEqual(freed & set(owners), set(), "no tree-owned slot was freed")
        self.assertEqual(freed, {300}, "only its own freshly computed slot goes back")

    def test_blocks_are_inserted_in_their_own_namespaces(self):
        cache, pool, _ = make_cache()
        req = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, req, first_slot=100)

        sys_hit = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3], SYS_KEY))
        )
        msg_hit = cache.match_prefix(MatchPrefixParams(key=RadixKey([4, 5], MSG_KEY)))
        self.assertEqual(len(sys_hit.device_indices), 3)
        self.assertEqual(len(msg_hit.device_indices), 2)
        # Same tokens, no namespace: a different tree, still empty.
        plain = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], None)))
        self.assertEqual(len(plain.device_indices), 0)

    def test_second_request_reuses_both_blocks(self):
        cache, pool, _ = make_cache()
        first = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [9], first_slot=200)

        second = make_req("r2", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY], req_pool_idx=1)
        second.init_next_round_input(cache)
        # input_len - 1 keeps at least one token to compute.
        self.assertEqual(len(second.prefix_indices), 4)
        self.assertEqual(second.sub_context_match_lens, [3, 2])
        second.release_sub_context_match_locks(cache)

    def test_generated_tokens_extend_the_last_namespace(self):
        """The reply becomes part of next turn's message block, so cache it."""
        cache, pool, allocator = make_cache()
        first = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [6, 7], first_slot=200)

        # The generated slots were handed to the tree, not freed.
        self.assertEqual(allocator.freed, [])
        extended = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([4, 5, 6, 7], MSG_KEY))
        )
        self.assertEqual(len(extended.device_indices), 4)
        self.assertEqual(extended.device_indices.tolist(), [103, 104, 200, 201])
        # The head namespace is untouched by the continuation.
        self.assertEqual(
            len(
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey([1, 2, 3], SYS_KEY))
                ).device_indices
            ),
            3,
        )
        # Nothing stays locked once the request is done.
        self.assertEqual(cache.protected_size(), 0)

    def test_next_turn_hits_the_generated_tokens(self):
        """Turn 2's message block starts with turn 1's reply -- it must hit."""
        cache, pool, _ = make_cache()
        first = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [6, 7], first_slot=200)

        # Next turn: same head, message block grew by the reply plus a new question.
        second = make_req(
            "r2", [[1, 2, 3], [4, 5, 6, 7, 8]], [SYS_KEY, MSG_KEY], req_pool_idx=1
        )
        second.init_next_round_input(cache)
        self.assertEqual(second.sub_context_match_lens, [3, 4])
        self.assertEqual(len(second.prefix_indices), 7)
        self.assertEqual(
            second.prefix_indices.tolist(), [100, 101, 102, 103, 104, 200, 201]
        )
        second.release_sub_context_match_locks(cache)

    def test_output_caching_can_be_turned_off(self):
        cache, pool, allocator = make_cache()
        import sglang.srt.mem_cache.radix_cache as radix_cache_module

        original = radix_cache_module.CACHE_SUBCONTEXT_OUTPUT
        radix_cache_module.CACHE_SUBCONTEXT_OUTPUT = False
        try:
            req = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
            prefill(cache, pool, req, first_slot=100)
            decode_and_finish(cache, pool, req, [6, 7], first_slot=200)
        finally:
            radix_cache_module.CACHE_SUBCONTEXT_OUTPUT = original

        self.assertEqual(allocator.freed, [200, 201])
        self.assertEqual(
            len(
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey([4, 5, 6, 7], MSG_KEY))
                ).device_indices
            ),
            2,
        )

    def test_duplicate_continuation_is_freed_not_leaked(self):
        """Two requests generating the same reply must not double-own slots."""
        cache, pool, allocator = make_cache()
        first = make_req("r1", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [6, 7], first_slot=200)

        second = make_req("r2", [[1, 2, 3], [4, 5]], [SYS_KEY, MSG_KEY], req_pool_idx=1)
        second.init_next_round_input(cache)
        second.release_sub_context_match_locks(cache)
        # Recompute the whole prompt on fresh slots, then generate the same reply.
        n = len(second.origin_input_ids)
        pool.req_to_token[1, :n] = torch.arange(300, 300 + n, dtype=torch.int64)
        second.prefix_indices = torch.empty((0,), dtype=torch.int64)
        second.cache_protected_len = 0
        second.fill_ids = list(second.origin_input_ids)
        cache.cache_unfinished_req(second)
        allocator.freed.clear()
        decode_and_finish(cache, pool, second, [6, 7], first_slot=400)

        # The duplicated continuation slots go back to the pool; the tree keeps the
        # first request's.
        self.assertEqual(allocator.freed, [400, 401])
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey([4, 5, 6, 7], MSG_KEY))
            ).device_indices.tolist(),
            [103, 104, 200, 201],
        )


class TestReverseRotateInsert(unittest.TestCase):
    """A block the namespace refused is rotated back to its position and filed there.

    The read path can rescue a displaced block for *this* request; the write path could
    not, so the block stayed a hole, the prompt was never fully tree-owned, and
    `_cache_sub_context_output` refused the reply. In an agent loop that costs a whole
    turn: the next round re-prefills every reply it was meant to have cached.
    """

    TOOLS_KEY = "tools_key"

    def _seed(self, rotator=None):
        """Cache SYS[1,2,3]@0, TOOLS[7,8]@3, MSG[20,21]@5."""
        cache, pool, allocator = make_cache(rotator=rotator)
        first = make_req(
            "r1", [[1, 2, 3], [7, 8], [20, 21]], [SYS_KEY, self.TOOLS_KEY, MSG_KEY]
        )
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=200)
        return cache, pool, allocator

    def _shifted(self):
        """TOOLS[7,8] sits at offset 2 here, so the tree holds it one place later."""
        return make_req(
            "r2",
            [[1, 2], [7, 8], [40, 41]],
            [SYS_KEY, self.TOOLS_KEY, MSG_KEY],
            req_pool_idx=1,
        )

    def test_declined_block_is_rotated_back_and_filed(self):
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted()
        prefill(cache, pool, req, first_slot=300)
        # The write path still refuses it during prefill: the request is decoding over
        # these slots, so nothing may move them yet.
        self.assertEqual(req.sub_context_tree_owned, [True, False, True])
        rotator.calls.clear()

        decode_and_finish(cache, pool, req, [9], first_slot=400)

        # `canonical - offset` = 3 - 2, the read path's rotation run backwards, and in
        # place: source and destination are the same slots.
        self.assertEqual(rotator.calls, [([900, 901], [900, 901], 1)])
        self.assertEqual(req.sub_context_reinserted, 2)
        self.assertEqual(cache.sub_context_reinserted_tokens, 2)

    def test_the_reply_is_cached_once_the_hole_is_closed(self):
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted()
        prefill(cache, pool, req, first_slot=300)
        decode_and_finish(cache, pool, req, [9], first_slot=400)

        # MSG is the last block, so the reply extends its namespace -- which the gate
        # only allows because the re-file left no block un-owned.
        extended = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([40, 41, 9], MSG_KEY))
        )
        self.assertEqual(extended.device_indices.tolist(), [304, 305, 400])

    def test_the_namespace_keeps_one_position(self):
        """The re-filed block must read back at the first writer's position."""
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted()
        prefill(cache, pool, req, first_slot=300)
        decode_and_finish(cache, pool, req, [9], first_slot=400)

        m = cache.match_prefix(MatchPrefixParams(key=RadixKey([7, 8], self.TOOLS_KEY)))
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 2), 3)
        # A full duplicate: the tree keeps its own copy and the re-file gave back the
        # rotated one it no longer needs.
        self.assertEqual(m.device_indices.tolist(), [103, 104])
        self.assertIn(900, allocator.freed)
        self.assertIn(901, allocator.freed)

    def test_a_new_tail_reaches_the_tree(self):
        """The payoff case: the request's block runs past what the namespace holds."""
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = make_req(
            "r2",
            [[1, 2], [7, 8, 30, 31], [40, 41]],
            [SYS_KEY, self.TOOLS_KEY, MSG_KEY],
            req_pool_idx=1,
        )
        prefill(cache, pool, req, first_slot=300)
        decode_and_finish(cache, pool, req, [], first_slot=400)

        # [7,8] was already there at 3; [30,31] is new and lands right after it.
        m = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([7, 8, 30, 31], self.TOOLS_KEY))
        )
        self.assertEqual(len(m.device_indices), 4)
        self.assertEqual(cache.matched_canonical_position(m.last_device_node, 4), 3)

    def _declined_block_with_shared_head(self):
        """A block the write path refused whose head is the TREE's own slots.

        Seen on H200 job 429: `tree_owned=[True, False]` with `owned_lens=[259, 2197]`
        -- the stitch took the messages block's head straight from the tree at a plain
        hit, and only afterwards did another writer move that namespace, so the write
        path declined the block. From finish's point of view the block is "not tree
        owned", yet part of it is, and that is the whole trap.

        Returns the pieces both regressions need, with the request stitched, written to
        req_to_token, and already through `cache_unfinished_req`.
        """
        rotator = FakeRotator()
        cache, pool, allocator = make_cache(rotator=rotator)
        first = make_req("r1", [[1, 2, 3], [90, 91, 92, 93]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=200)

        # Same offset, so the stitch reuses the tree's slots for the block's head.
        second = make_req(
            "r2", [[1, 2, 3], [90, 91, 92, 93]], [SYS_KEY, MSG_KEY], req_pool_idx=1
        )
        second.init_next_round_input(cache)
        n = len(second.origin_input_ids)
        reused = len(second.prefix_indices)
        pool.req_to_token[1, :reused] = second.prefix_indices
        pool.req_to_token[1, reused:n] = torch.arange(
            300, 300 + n - reused, dtype=torch.int64
        )
        second.sub_context_rotated_slots = None
        second.fill_ids = list(second.origin_input_ids)
        tree_slots = pool.req_to_token[1, 3:6].tolist()

        # Another writer moves the namespace between the match and the insert, which is
        # what makes the block get declined even though its head is the tree's.
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([90, 91, 92, 93], MSG_KEY))
        ).last_device_node
        node.canonical_position = 99
        cache.cache_unfinished_req(second)
        self.assertEqual(second.sub_context_tree_owned, [True, False])
        return cache, pool, allocator, rotator, second, tree_slots

    def test_a_tree_reused_head_is_never_handed_back(self):
        """Freeing a declined block from its start returns the tree's own slots.

        Its node still points at them, so the allocator reissues them while the tree
        goes on serving them as cache -- which is how the KV cache ends up claiming
        more tokens than the pool holds.
        """
        cache, pool, allocator, _rot, second, tree_slots = (
            self._declined_block_with_shared_head()
        )
        allocator.freed.clear()
        decode_and_finish(cache, pool, second, [9], first_slot=400)
        self.assertEqual(
            [s for s in tree_slots if s in allocator.freed],
            [],
            "handed the tree's own slots back to the pool",
        )

    def test_a_tree_held_slot_is_kept_wherever_it_sits(self):
        """The overlap with the tree is not a prefix, so a leading run cannot find it.

        Job 432: `double_pos` began one slot AFTER the block's offset -- the first slot
        differed and the ~2000 behind it did not. Counting the leading run of agreement
        gave 0, and the whole matched prefix went back to the pool while the tree went
        on serving it; three different requests handed back the same slots (212, 213,
        214 ...) that way, and the pool then issued live KV.

        The head is diverged here directly rather than through whatever produced it on
        the H200 -- a recomputed first token, a node another writer replaced. What has
        to hold is the response to the shape, not the shape's provenance.
        """
        cache, pool, allocator, _rot, second, tree_slots = (
            self._declined_block_with_shared_head()
        )
        pool.req_to_token[1, 3] = 999  # first slot of the block no longer the tree's
        still_the_trees = pool.req_to_token[1, 4:6].tolist()

        allocator.freed.clear()
        decode_and_finish(cache, pool, second, [9], first_slot=400)
        self.assertEqual(
            [s for s in still_the_trees if s in allocator.freed],
            [],
            "a leading-run test missed slots the tree holds behind a diverged head",
        )
        self.assertIn(999, allocator.freed, "this request's own slot was not freed")

    def test_a_shared_head_is_never_rotated_in_place(self):
        """In-place rotation is only ever safe on slots this request allocated.

        The re-file rotates a block where it lies, which is sound for KV this request
        computed and catastrophic for KV it merely borrowed: the shared node would come
        away rotated for somewhere else while still advertising its old canonical
        position, so every later hit on it reads K rotated for nowhere. Unlike a double
        free this leaves the accounting perfect -- nothing but output quality shows it,
        which is why it gets its own test rather than riding on the free assertion.
        """
        cache, pool, allocator, rotator, second, tree_slots = (
            self._declined_block_with_shared_head()
        )
        rotator.calls.clear()
        decode_and_finish(cache, pool, second, [9], first_slot=400)

        touched = [
            (src, delta)
            for src, _dst, delta in rotator.calls
            if delta and set(src) & set(tree_slots)
        ]
        self.assertEqual(touched, [], "rotated KV the tree still owns")
        # And the block stays out of the tree rather than being filed half-rotated.
        self.assertEqual(second.sub_context_reinserted, 0)

    def test_an_unrotatable_delta_keeps_the_old_behaviour(self):
        rotator = FakeRotator()
        cache, pool, allocator = self._seed(rotator)
        req = self._shifted()
        prefill(cache, pool, req, first_slot=300)  # the read path rotates into 900,901
        cache.kv_rotator = FakeRotator(max_delta=0)  # ...but the re-file cannot
        allocator.freed.clear()
        decode_and_finish(cache, pool, req, [9], first_slot=400)

        self.assertEqual(req.sub_context_reinserted, 0)
        # The block stays this request's to free, and the reply is dropped as before.
        self.assertEqual(sorted(allocator.freed), [400, 900, 901])
        self.assertEqual(
            len(
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey([40, 41, 9], MSG_KEY))
                ).device_indices
            ),
            2,  # the block, not the reply
        )

    def test_without_a_rotator_nothing_is_refiled(self):
        """The plain ON arm must behave exactly as it did before this existed."""
        cache, pool, allocator = self._seed(rotator=None)
        req = self._shifted()
        prefill(cache, pool, req, first_slot=300)
        allocator.freed.clear()
        decode_and_finish(cache, pool, req, [9], first_slot=400)

        self.assertEqual(req.sub_context_reinserted, 0)
        self.assertEqual(cache.sub_context_reinserted_tokens, 0)
        # Nothing was rotated on the read path either, so [7,8] was recomputed at 302.
        self.assertEqual(sorted(allocator.freed), [302, 303, 400])


class TestSubContextRequestNormalization(unittest.TestCase):
    """`text` and `sub_contexts` may not describe two different prompts."""

    def _obj(self, **kwargs):
        return GenerateReqInput(
            sub_contexts=[
                {"content": "you are a bot.", "extra_key": SYS_KEY},
                {"content": "hello", "extra_key": MSG_KEY},
            ],
            **kwargs,
        )

    def test_text_is_derived_from_the_blocks(self):
        obj = self._obj()
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.text, "you are a bot.hello")

    def test_matching_text_is_kept_with_the_split(self):
        obj = self._obj(text="you are a bot.hello")
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.text, "you are a bot.hello")
        self.assertIsNotNone(obj.sub_contexts)

    def test_conflicting_text_is_rejected(self):
        obj = self._obj(text="a completely different prompt")
        with self.assertRaises(ValueError):
            obj.normalize_batch_and_arguments()

    def test_input_ids_wins_and_drops_the_split(self):
        obj = self._obj(input_ids=[1, 2, 3])
        obj.normalize_batch_and_arguments()
        self.assertIsNone(obj.sub_contexts)


class TestSubContextTruncation(unittest.TestCase):
    """Auto-truncation must not leave the block list describing a longer prompt."""

    def setUp(self):
        from sglang.srt.managers.tokenizer_manager import _clip_sub_contexts_to_input

        self.clip = _clip_sub_contexts_to_input

    def test_untouched_prompt_is_passed_through(self):
        ids, keys = self.clip([[1, 2], [3, 4]], [SYS_KEY, MSG_KEY], 4, "r")
        self.assertEqual(ids, [[1, 2], [3, 4]])
        self.assertEqual(keys, [SYS_KEY, MSG_KEY])

    def test_trailing_block_is_clipped(self):
        ids, keys = self.clip([[1, 2], [3, 4]], [SYS_KEY, MSG_KEY], 3, "r")
        self.assertEqual(ids, [[1, 2], [3]])
        self.assertEqual(keys, [SYS_KEY, MSG_KEY])
        self.assertEqual(sum(len(s) for s in ids), 3)

    def test_blocks_past_the_cut_are_dropped(self):
        ids, keys = self.clip([[1, 2], [3, 4]], [SYS_KEY, MSG_KEY], 2, "r")
        self.assertEqual(ids, [[1, 2]])
        self.assertEqual(keys, [SYS_KEY])

    def test_split_is_dropped_when_nothing_is_left(self):
        ids, keys = self.clip([[1, 2], [3, 4]], [SYS_KEY, MSG_KEY], 0, "r")
        self.assertIsNone(ids)
        self.assertIsNone(keys)

    def test_blocks_shorter_than_the_prompt_drop_the_split(self):
        ids, keys = self.clip([[1, 2]], [SYS_KEY], 5, "r")
        self.assertIsNone(ids)
        self.assertIsNone(keys)


if __name__ == "__main__":
    unittest.main()


class TestRetractedRequest(unittest.TestCase):
    """A retracted request re-enters prefill with its generated tokens in fill_ids.

    `init_next_round_input` sets ``fill_ids = origin_input_ids + output_ids``
    (`schedule_batch.py:1029`), so once a request has produced anything its fill_ids are
    longer than the prompt the split describes. The read path stitches per namespace
    regardless -- its gate never looks at that length -- so the request comes back
    holding the namespaces' own slots.
    """

    def _retracted(self, cache, pool):
        first = make_req("r1", [[1, 2, 3], [40, 41]], [SYS_KEY, MSG_KEY])
        prefill(cache, pool, first, first_slot=100)
        decode_and_finish(cache, pool, first, [], first_slot=200)

        second = make_req(
            "r2", [[1, 2, 3], [40, 41]], [SYS_KEY, MSG_KEY], req_pool_idx=1
        )
        second.output_ids = [9]  # retracted after generating one token
        second.init_next_round_input(cache)
        n = len(second.origin_input_ids)
        reused = len(second.prefix_indices)
        pool.req_to_token[1, :reused] = second.prefix_indices
        pool.req_to_token[1, reused : n + 1] = torch.arange(
            300, 300 + n + 1 - reused, dtype=torch.int64
        )
        second.sub_context_rotated_slots = None
        self.assertGreater(len(second.fill_ids), n, "not the retracted shape")
        self.assertGreater(reused, 0, "the stitch reused nothing; nothing to protect")
        return second

    def _namespace_slots(self, cache):
        slots = set()
        for key, ids in ((SYS_KEY, [1, 2, 3]), (MSG_KEY, [40, 41])):
            slots.update(
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(ids, key))
                ).device_indices.tolist()
            )
        return slots

    def test_a_namespace_slot_never_gets_a_second_owner(self):
        """The write path must follow the read path into the namespaces.

        Writing this request to the default namespace instead files the KV the
        namespaces already own under a second node. Nothing is freed at that moment, so
        no audit fires -- but from then on either owner can free it while the other goes
        on serving it, which is how the pool starts handing out live KV.
        """
        cache, pool, allocator = make_cache()
        second = self._retracted(cache, pool)
        cache.cache_unfinished_req(second)

        default = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(second.fill_ids), None))
        ).device_indices.tolist()
        self.assertEqual(
            sorted(self._namespace_slots(cache) & set(default)),
            [],
            "namespace slots filed under the default namespace as well",
        )

    def test_it_still_reaches_the_audited_finish_path(self):
        """...and therefore finishes through the audited branch rather than around it.

        `cache_finished_req` picks its branch on `sub_context_last_nodes`, which only
        the per-namespace insert sets. A request that wrote to the default namespace
        finishes through the ordinary path, where the conservation check never runs.
        """
        cache, pool, allocator = make_cache()
        second = self._retracted(cache, pool)
        cache.cache_unfinished_req(second)
        self.assertIsNotNone(second.sub_context_last_nodes)
