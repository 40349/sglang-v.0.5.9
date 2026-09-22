"""
Unit tests for selective recompute of reused sub-context tokens.

Reuse is cheap because a reused token is never recomputed. That is also why its KV
still carries the context it was cached under: rotation moves a block to a new
position, it does not move it to a new neighbourhood. Selective recompute buys a
fraction of that back by running the whole prompt through the first layers -- purely to
obtain a key to compare against the cache -- scoring the reused positions, and keeping
only the highest scorers alongside the tokens that were going to be computed anyway.

The properties below are the ones whose failure is silent:

- the selection is the fresh tokens plus the top scorers, ascending, and ends on the
  prompt's last position -- the logits processor takes the last row of each request,
- a reused row's KV is never written to the cache row it is reusing, which belongs to
  the radix tree and is shared with every other request holding that block,
- a selected position gets **every** layer filled, not just the ones recomputed after
  the cut, because decode reads all of them,
- at ratio 0 the selection is exactly the fresh set, which is what makes the arm
  bit-identical to reuse-without-recompute.

Usage:
    python test_subctx_blend.py
    python -m pytest test_subctx_blend.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.subctx_blend import build_plan
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import subctx_config

NUM_LAYERS = 4
NUM_HEADS = 2
HEAD_DIM = 8
NUM_SLOTS = 1024
CHECK_LAYER = 1


class FakeReqToTokenPool:
    def __init__(self, size: int = 4, max_context_len: int = 64):
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class FakeKVPool:
    """A per-layer K/V buffer and the one write the backend makes into it.

    ``set_kv_buffer`` is a scatter by slot id, which is all the real pool does once the
    dtype handling is stripped out.
    """

    def __init__(self):
        self.k_buffer = [
            torch.zeros(NUM_SLOTS, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
        ]
        self.v_buffer = [
            torch.zeros(NUM_SLOTS, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
        ]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def set_kv_buffer(self, layer, loc, k, v):
        self.k_buffer[layer.layer_id][loc] = k
        self.v_buffer[layer.layer_id][loc] = v


class FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class FakeForwardBatch:
    def __init__(self, kv_pool, req_to_token_pool):
        self.token_to_kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool


class FakeBatch:
    def __init__(self, reqs, fresh_positions, req_to_token_pool):
        self.reqs = reqs
        self.subctx_fresh_positions = fresh_positions
        self.req_to_token_pool = req_to_token_pool


def make_req(prompt_len: int, layout, req_pool_idx: int = 0, ours=None) -> Req:
    """A request whose reuse lands at ``layout``: (start, end, slots) runs.

    ``ours`` says, per run, whether those rows are a rotated copy made for this
    request (True) or the tree's own rows (False). With rotation on, which it must be
    for the index, most runs are copies.
    """
    ids = list(range(1, prompt_len + 1))
    req = Req(
        rid=f"r{req_pool_idx}",
        origin_input_text="",
        origin_input_ids=ids,
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    req.fill_ids = list(ids)
    req.req_pool_idx = req_pool_idx
    req.sub_context_layout = [
        (start, end, torch.arange(slot0, slot0 + end - start, dtype=torch.int64))
        for start, end, slot0 in layout
    ]
    req.sub_context_layout_ours = (
        list(ours) if ours is not None else [False] * len(layout)
    )
    return req


def build(reqs, first_slot: int = 400, check_layer: int = CHECK_LAYER):
    """Lay out a batch the way ``alloc_for_extend`` would, and hand back the plan."""
    pool = FakeReqToTokenPool()
    fresh_positions = [r.sub_context_fresh_positions() for r in reqs]
    topk_counts = [r.sub_context_topk_count() for r in reqs]

    num_fresh = sum(len(p) for p in fresh_positions)
    out_cache_loc = torch.arange(
        first_slot, first_slot + num_fresh, dtype=torch.int64
    )
    topk_slots = torch.arange(
        first_slot + num_fresh,
        first_slot + num_fresh + sum(topk_counts),
        dtype=torch.int64,
    )

    # What `write_cache_indices_sparse` puts in the table: reused runs at their
    # positions, the computed tokens scattered into the gaps.
    cursor = 0
    for i, req in enumerate(reqs):
        for start, end, slots in req.sub_context_layout:
            pool.req_to_token[req.req_pool_idx, start:end] = slots
        positions = torch.tensor(fresh_positions[i], dtype=torch.int64)
        pool.req_to_token[req.req_pool_idx, positions] = out_cache_loc[
            cursor : cursor + len(positions)
        ]
        cursor += len(positions)

    batch = FakeBatch(reqs, fresh_positions, pool)
    plan = build_plan(
        batch,
        out_cache_loc,
        [r.req_pool_idx for r in reqs],
        topk_slots,
        check_layer,
    )
    return plan, pool, out_cache_loc, topk_slots


def keys_with_deviation(plan, loud_positions, kv_pool, request=0):
    """Fresh keys that match the cache everywhere except at ``loud_positions``."""
    total_rows = sum(plan.probe_lens)
    cached = kv_pool.get_key_buffer(plan.check_layer)
    fresh = torch.zeros(total_rows, NUM_HEADS, HEAD_DIM)
    fresh[plan.reused_rows] = cached[plan.reused_slots]
    for rank, position in enumerate(loud_positions):
        row = plan.row_offsets[request] + position
        # Descending, so the order the positions are given in is the order they should
        # come back in -- a test that passes on a tie tells us nothing.
        fresh[row] += float(len(loud_positions) - rank)
    return fresh


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.ratio = subctx_config.TOPK_RATIO
        self.index = subctx_config.INDEX_SUBCONTEXTS
        subctx_config.INDEX_SUBCONTEXTS = True

    def tearDown(self):
        subctx_config.TOPK_RATIO = self.ratio
        subctx_config.INDEX_SUBCONTEXTS = self.index

    def test_picks_the_positions_whose_keys_moved_most(self):
        subctx_config.TOPK_RATIO = 0.25
        # 32 tokens; [4, 20) is reused, so 16 reused tokens and a quota of 4.
        req = make_req(32, [(4, 20, 500)])
        plan, _pool, _fresh_loc, _topk = build([req])
        self.assertEqual(plan.topk_counts, [4])
        self.assertEqual(plan.sel_lens, [16 + 4])

        kv_pool = FakeKVPool()
        loud = [7, 12, 5, 18]
        k = keys_with_deviation(plan, loud, kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))

        chosen = (plan.reused_rows[plan.sel_topk_rows] - plan.row_offsets[0]).tolist()
        self.assertEqual(chosen, loud)

    def test_selection_is_ascending_and_keeps_the_last_token_last(self):
        subctx_config.TOPK_RATIO = 0.5
        req = make_req(24, [(2, 18, 500)])
        plan, _pool, _fresh_loc, _topk = build([req])

        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [3, 9, 4, 15, 11, 16, 2, 7], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))

        positions = plan.sel_positions.tolist()
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(positions), plan.sel_lens[0])
        # The logits processor takes cumsum(extend_seq_lens) - 1 as each request's last
        # row, so the last prompt position has to be the last row.
        self.assertEqual(positions[-1], 23)
        self.assertEqual(plan.sel_rows.tolist(), positions)

    def test_selection_is_the_union_of_fresh_and_chosen(self):
        subctx_config.TOPK_RATIO = 0.25
        req = make_req(32, [(4, 20, 500)])
        plan, _pool, _fresh_loc, _topk = build([req])

        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [6, 11, 17, 19], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))

        fresh = set(req.sub_context_fresh_positions())
        selected = set(plan.sel_positions.tolist())
        self.assertTrue(fresh.issubset(selected))
        self.assertEqual(selected - fresh, {6, 11, 17, 19})

    def test_ratio_zero_selects_exactly_the_fresh_tokens(self):
        subctx_config.TOPK_RATIO = 0.0
        req = make_req(32, [(4, 20, 500)])
        plan, _pool, _fresh_loc, _topk = build([req])
        self.assertEqual(plan.topk_counts, [0])

        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [6, 11], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))

        self.assertEqual(
            plan.sel_positions.tolist(), req.sub_context_fresh_positions()
        )
        self.assertEqual(plan.sel_topk_rows.numel(), 0)

    def test_each_request_gets_its_own_quota(self):
        subctx_config.TOPK_RATIO = 0.25
        # Eight reused tokens for one, sixteen for the other: quotas 2 and 4.
        first = make_req(16, [(2, 10, 500)], req_pool_idx=0)
        second = make_req(32, [(4, 20, 600)], req_pool_idx=1)
        plan, _pool, _fresh_loc, _topk = build([first, second])
        self.assertEqual(plan.topk_counts, [2, 4])

        kv_pool = FakeKVPool()
        cached = kv_pool.get_key_buffer(plan.check_layer)
        k = torch.zeros(sum(plan.probe_lens), NUM_HEADS, HEAD_DIM)
        k[plan.reused_rows] = cached[plan.reused_slots]
        # Every deviation is in the second request. Its quota must not spill into the
        # first, and the first must still fill its own from a flat score. Distinct
        # magnitudes, so the four that should win are the four that do rather than
        # whichever four a tie happened to order first.
        for rank, position in enumerate((5, 6, 7, 8, 9, 10)):
            k[plan.row_offsets[1] + position] += 6.0 - rank
        plan.select(k, cached)

        self.assertEqual(plan.sel_positions.numel(), sum(plan.sel_lens))
        chosen = plan.reused_rows[plan.sel_topk_rows]
        first_chosen = chosen[chosen < plan.row_offsets[1]]
        second_chosen = chosen[chosen >= plan.row_offsets[1]] - plan.row_offsets[1]
        self.assertEqual(first_chosen.numel(), 2)
        self.assertEqual(sorted(second_chosen.tolist()), [5, 6, 7, 8])


class TestProbeWrites(unittest.TestCase):
    def setUp(self):
        self.ratio = subctx_config.TOPK_RATIO
        self.index = subctx_config.INDEX_SUBCONTEXTS
        subctx_config.INDEX_SUBCONTEXTS = True
        subctx_config.TOPK_RATIO = 0.25

    def tearDown(self):
        subctx_config.TOPK_RATIO = self.ratio
        subctx_config.INDEX_SUBCONTEXTS = self.index

    def test_reused_rows_are_written_to_the_discard_slot(self):
        req = make_req(32, [(4, 20, 500)])
        plan, _pool, fresh_loc, _topk = build([req])

        loc = plan.probe_cache_loc
        self.assertEqual(loc.numel(), 32)
        # Slot 0 is the allocator's padded dummy: never handed out, so a write there
        # cannot reach another request's KV.
        self.assertTrue(bool((loc[plan.reused_rows] == 0).all()))
        fresh_positions = torch.tensor(req.sub_context_fresh_positions())
        self.assertEqual(loc[fresh_positions].tolist(), fresh_loc.tolist())

    def test_every_computed_token_keeps_its_own_slot(self):
        first = make_req(16, [(2, 10, 500)], req_pool_idx=0)
        second = make_req(32, [(4, 20, 600)], req_pool_idx=1)
        plan, _pool, fresh_loc, _topk = build([first, second])

        written = plan.probe_cache_loc[plan.probe_cache_loc != 0]
        self.assertEqual(sorted(written.tolist()), sorted(fresh_loc.tolist()))


class TestCommit(unittest.TestCase):
    def setUp(self):
        self.ratio = subctx_config.TOPK_RATIO
        self.index = subctx_config.INDEX_SUBCONTEXTS
        subctx_config.INDEX_SUBCONTEXTS = True
        subctx_config.TOPK_RATIO = 0.25

    def tearDown(self):
        subctx_config.TOPK_RATIO = self.ratio
        subctx_config.INDEX_SUBCONTEXTS = self.index

    def commit_once(self, loud=(6, 11, 17, 19)):
        req = make_req(32, [(4, 20, 500)])
        plan, pool, _fresh_loc, topk_slots = build([req])

        kv_pool = FakeKVPool()
        # Give every cache row a fingerprint, so a slot that was never filled is
        # distinguishable from one that was.
        for layer in range(NUM_LAYERS):
            kv_pool.k_buffer[layer] = (
                torch.arange(NUM_SLOTS, dtype=torch.float32)
                .view(-1, 1, 1)
                .expand(NUM_SLOTS, NUM_HEADS, HEAD_DIM)
                .contiguous()
                + layer * 1000.0
            )
            kv_pool.v_buffer[layer] = kv_pool.k_buffer[layer].clone()

        k = keys_with_deviation(plan, list(loud), kv_pool)
        v = k.clone()
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))

        fb = FakeForwardBatch(kv_pool, pool)
        plan.commit(k, v, fb, FakeLayer(plan.check_layer))
        return plan, pool, kv_pool, topk_slots, k

    def test_layers_before_the_scored_one_are_seeded_from_the_cache(self):
        plan, _pool, kv_pool, topk_slots, _k = self.commit_once()
        source = plan.reused_slots[plan.sel_topk_rows]
        for layer in range(plan.check_layer):
            # Not merely non-zero: the value has to be the one the reused row held, or
            # decode reads a different token's key at that layer.
            self.assertTrue(
                torch.equal(
                    kv_pool.k_buffer[layer][topk_slots],
                    kv_pool.k_buffer[layer][source],
                )
            )
            self.assertTrue(
                torch.equal(
                    kv_pool.v_buffer[layer][topk_slots],
                    kv_pool.v_buffer[layer][source],
                )
            )

    def test_the_scored_layer_holds_the_recomputed_key(self):
        plan, _pool, kv_pool, topk_slots, k = self.commit_once()
        rows = plan.reused_rows[plan.sel_topk_rows]
        self.assertTrue(
            torch.equal(kv_pool.k_buffer[plan.check_layer][topk_slots], k[rows])
        )

    def test_later_layers_are_left_for_the_second_range(self):
        plan, _pool, kv_pool, topk_slots, _k = self.commit_once()
        # Untouched here, and filled by the layers that run after the cut. Seeding them
        # too would only be wasted copying.
        for layer in range(plan.check_layer + 1, NUM_LAYERS):
            expected = (
                topk_slots.to(torch.float32).view(-1, 1, 1).expand_as(
                    kv_pool.k_buffer[layer][topk_slots]
                )
                + layer * 1000.0
            )
            self.assertTrue(
                torch.equal(kv_pool.k_buffer[layer][topk_slots], expected)
            )

    def test_selected_positions_stop_pointing_at_the_tree(self):
        plan, pool, _kv_pool, topk_slots, _k = self.commit_once()
        row = pool.req_to_token[0]
        chosen = (plan.reused_rows[plan.sel_topk_rows] - plan.row_offsets[0]).tolist()
        self.assertEqual(
            sorted(row[chosen].tolist()), sorted(topk_slots.tolist())
        )

        # And the reused positions that were not selected still do -- recomputing four
        # tokens must not detach the other twelve from the block they came from.
        untouched = [p for p in range(4, 20) if p not in chosen]
        self.assertEqual(row[untouched].tolist(), [500 + p - 4 for p in untouched])

    def test_displaced_copies_of_our_own_are_handed_back(self):
        # The row a selected position leaves is only reachable through req_to_token, and
        # the selection has just overwritten that entry. A rotated copy is this
        # request's alone, so unless it is reported here nothing ever frees it -- which
        # shows up minutes later as a pool leak of exactly ratio x reused tokens, far
        # from the code that caused it.
        req = make_req(32, [(4, 20, 500)], ours=[True])
        plan, pool, _fresh_loc, _topk = build([req])
        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [6, 11, 17, 19], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))
        plan.commit(k, k.clone(), FakeForwardBatch(kv_pool, pool), FakeLayer(1))

        # The four rows those positions used to point at, and nothing else.
        self.assertEqual(
            sorted(plan.orphaned_slots.tolist()),
            sorted(500 + p - 4 for p in (6, 11, 17, 19)),
        )
        row = pool.req_to_token[0]
        self.assertTrue(set(plan.orphaned_slots.tolist()).isdisjoint(row.tolist()))

    def test_displaced_tree_rows_are_left_alone(self):
        # Same displacement, but the rows belong to the tree: other requests are still
        # served from them, and handing them back would corrupt those.
        req = make_req(32, [(4, 20, 500)], ours=[False])
        plan, pool, _fresh_loc, _topk = build([req])
        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [6, 11, 17, 19], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))
        plan.commit(k, k.clone(), FakeForwardBatch(kv_pool, pool), FakeLayer(1))

        self.assertEqual(plan.orphaned_slots.numel(), 0)

    def test_ownership_is_per_run_not_per_request(self):
        # A prompt can reuse one block where it was cached and another somewhere else.
        # Only the second is a copy, so only its displaced rows come back.
        req = make_req(48, [(4, 20, 500), (24, 40, 600)], ours=[False, True])
        plan, pool, _fresh_loc, _topk = build([req])
        self.assertEqual(plan.topk_counts, [8])
        kv_pool = FakeKVPool()
        k = keys_with_deviation(plan, [5, 6, 7, 8, 25, 26, 27, 28], kv_pool)
        plan.select(k, kv_pool.get_key_buffer(plan.check_layer))
        plan.commit(k, k.clone(), FakeForwardBatch(kv_pool, pool), FakeLayer(1))

        self.assertEqual(
            sorted(plan.orphaned_slots.tolist()),
            sorted(600 + p - 24 for p in (25, 26, 27, 28)),
        )

    def test_the_table_is_what_the_second_range_reads_slots_from(self):
        plan, pool, _kv_pool, topk_slots, _k = self.commit_once()
        # `_gather_sel_cache_loc` reads req_to_token rather than reassembling the two
        # allocations, so every selected position must resolve there.
        slots = pool.req_to_token[0][plan.sel_positions]
        self.assertEqual(slots.numel(), plan.sel_lens[0])
        self.assertTrue(bool((slots != 0).all()))
        self.assertEqual(len(set(slots.tolist())), slots.numel())


if __name__ == "__main__":
    unittest.main()
