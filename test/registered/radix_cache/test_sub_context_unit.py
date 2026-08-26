"""
Unit tests for the sub-context path of RadixCache.

A sub-context request has its prompt split into ordered blocks, each matched and
inserted under its own ``extra_key`` namespace. These tests drive the full
lifecycle on CPU tensors -- match, per-namespace insert, finish -- and cover the
invariants that path depends on:

- read and write must agree on whether the cache can serve the split at all,
- a namespace must stay pinned to one absolute position in the prompt,
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

    def __init__(self):
        self.device = torch.device("cpu")
        self.freed = []

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def available_size(self):
        return 0


def make_cache(page_size: int = 1, is_eagle: bool = False, disable: bool = False):
    pool = FakeReqToTokenPool()
    allocator = FakeAllocator()
    cache = RadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            is_eagle=is_eagle,
        )
    )
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


class TestSubContextLifecycle(unittest.TestCase):
    """Match -> per-namespace insert -> finish, and the reuse it enables."""

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
