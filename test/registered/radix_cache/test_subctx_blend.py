"""Unit tests for selective recompute of reused sub-context tokens.

Checked, on the plan alone:

- every probe row writes to its own slot: a fresh row to its new slot, a reused row to
  the request's copy,
- each scored layer keeps the top scorers of the reused rows still running, nested in
  what the previous one kept, per request, with the fresh rows always kept,
- a scored run that is not the request's own copy is refused.

Checked through ``ModelRunner.forward_subctx_blend`` on a small attention model:

- at ratio 1.0 the output and every KV row match a full prefill, scored at layer 1 or 0,
- a reused token keeps its cached KV on the layers after it was dropped, and has its KV
  rewritten on every layer it ran,
- with two scored layers, the second picks by its own deviation among the first's picks.

Usage:
    python test_subctx_blend.py
    python -m pytest test_subctx_blend.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.subctx_blend import build_plan
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams

NUM_LAYERS = 5
NUM_HEADS = 2
HEAD_DIM = 8
HIDDEN = NUM_HEADS * HEAD_DIM
NUM_SLOTS = 1024
VOCAB = 256
MAX_POS = 128
DTYPE = torch.float64


class FakeReqToTokenPool:
    def __init__(self, size: int = 4, max_context_len: int = MAX_POS):
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int64)


class FakeKVPool:
    """Per-layer K/V buffers; slots never written hold NaN."""

    def __init__(self):
        self.k_buffer = [
            torch.full((NUM_SLOTS, NUM_HEADS, HEAD_DIM), math.nan, dtype=DTYPE)
            for _ in range(NUM_LAYERS)
        ]
        self.v_buffer = [b.clone() for b in self.k_buffer]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def set_kv_buffer(self, layer_id, loc, k, v):
        self.k_buffer[layer_id][loc] = k
        self.v_buffer[layer_id][loc] = v


class ToyModel:
    """Attention-only decoder: each layer's K/V depend on the token's own hidden state,
    and its attention reads every key at or before the query's position through
    ``req_to_token``. Layer 0's K/V depend only on the token and its position.
    """

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        def rand(*shape, scale=1.0):
            return torch.randn(*shape, generator=g, dtype=DTYPE) * scale

        self.embed = rand(VOCAB, HIDDEN)
        self.pos = rand(MAX_POS, HIDDEN, scale=0.5)
        w = HIDDEN**-0.5
        self.wq = [rand(HIDDEN, HIDDEN, scale=w) for _ in range(NUM_LAYERS)]
        self.wk = [rand(HIDDEN, HIDDEN, scale=w) for _ in range(NUM_LAYERS)]
        self.wv = [rand(HIDDEN, HIDDEN, scale=w) for _ in range(NUM_LAYERS)]
        self.wo = [rand(HIDDEN, HIDDEN, scale=w) for _ in range(NUM_LAYERS)]

    def layer(self, i, positions, hidden, fb, residual):
        residual = hidden if residual is None else hidden + residual
        q = (residual @ self.wq[i]).view(-1, NUM_HEADS, HEAD_DIM)
        k = (residual @ self.wk[i]).view(-1, NUM_HEADS, HEAD_DIM)
        v = (residual @ self.wv[i]).view(-1, NUM_HEADS, HEAD_DIM)
        pool = fb.token_to_kv_pool
        pool.set_kv_buffer(i, fb.out_cache_loc, k, v)

        # Row counts and query positions as the attention metadata last saw them.
        meta = fb.attn_metadata
        out = torch.empty_like(q)
        row = 0
        for b, n in enumerate(meta.lens):
            slots = fb.req_to_token_pool.req_to_token[
                fb.req_pool_indices_cpu[b], : fb.seq_lens_cpu[b]
            ]
            keys, values = pool.k_buffer[i][slots], pool.v_buffer[i][slots]
            qp = meta.q_positions[row : row + n]
            att = torch.einsum("qhd,khd->hqk", q[row : row + n], keys) / HEAD_DIM**0.5
            visible = torch.arange(len(slots))[None, :] <= qp[:, None]
            att = att.masked_fill(~visible[None], -math.inf).softmax(-1)
            out[row : row + n] = torch.einsum("hqk,khd->qhd", att, values)
            row += n
        return torch.tanh(out.reshape(-1, HIDDEN) @ self.wo[i]), residual

    def forward_split_prefill(self, input_ids, positions, fb, interval, **_kwargs):
        start, end = interval
        if start == 0:
            fb.hidden_states = self.embed[input_ids] + self.pos[positions]
        for i in range(start, end):
            fb.hidden_states, fb.residual = self.layer(
                i, positions, fb.hidden_states, fb, fb.residual
            )
        if end < NUM_LAYERS:
            return None
        return fb.hidden_states + fb.residual


def init_attn_metadata(fb):
    """What the triton backend keeps from ``init_forward_metadata``: copies of the row
    counts and the query positions."""
    fb.attn_metadata = SimpleNamespace(
        lens=list(fb.extend_seq_lens_cpu), q_positions=fb.positions.clone()
    )


class FakeRunner:
    forward_subctx_blend = ModelRunner.forward_subctx_blend
    _set_extend_lens = ModelRunner._set_extend_lens

    def __init__(self, model):
        self.model = model
        self.model_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
        self.attn_backend = SimpleNamespace(init_forward_metadata=init_attn_metadata)


def make_req(
    prompt_len, layout, req_pool_idx=0, exact=0, ours=None, ids=None
) -> Req:
    """A request whose reuse lands at ``layout``: (start, end, first slot) runs, the
    first ``exact`` positions a plain prefix match. ``ours``, per run: whether its slots
    are this request's copy (default: every run past the prefix is)."""
    ids = list(ids) if ids is not None else list(range(1, prompt_len + 1))
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
        list(ours) if ours is not None else [start >= exact for start, _e, _s in layout]
    )
    req.sub_context_exact_len = exact
    return req


def make_plain_req(prompt_len, prefix_slot0, prefix_len, req_pool_idx, ids=None) -> Req:
    """An ordinary request in the batch: a plain prefix, then the tokens it computes."""
    req = make_req(prompt_len, [], req_pool_idx=req_pool_idx, ids=ids)
    req.sub_context_layout = None
    req.sub_context_layout_ours = None
    req.prefix_indices = torch.arange(prefix_slot0, prefix_slot0 + prefix_len)
    return req


def lay_out(reqs, first_slot=600):
    """What ``alloc_for_extend`` leaves: fresh slots, and req_to_token filled in."""
    pool = FakeReqToTokenPool()
    fresh_positions = [r.sub_context_fresh_positions() for r in reqs]
    num_fresh = sum(len(p) for p in fresh_positions)
    out_cache_loc = torch.arange(first_slot, first_slot + num_fresh, dtype=torch.int64)
    cursor = 0
    for i, req in enumerate(reqs):
        if req.sub_context_layout is None:
            prefix = req.prefix_indices
            pool.req_to_token[req.req_pool_idx, : len(prefix)] = prefix
        for start, end, slots in req.sub_context_layout or []:
            pool.req_to_token[req.req_pool_idx, start:end] = slots
        positions = torch.tensor(fresh_positions[i], dtype=torch.int64)
        pool.req_to_token[req.req_pool_idx, positions] = out_cache_loc[
            cursor : cursor + len(positions)
        ]
        cursor += len(positions)
    batch = SimpleNamespace(
        reqs=reqs, subctx_fresh_positions=fresh_positions, req_to_token_pool=pool
    )
    return batch, out_cache_loc


def reference(model, ids):
    """Full prefill of ``ids`` from position 0: per-layer K and V, and the output."""
    n = len(ids)
    pool = FakeKVPool()
    r2t = FakeReqToTokenPool()
    r2t.req_to_token[0, :n] = torch.arange(n)
    fb = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=r2t,
        out_cache_loc=torch.arange(n),
        positions=torch.arange(n),
        extend_seq_lens_cpu=[n],
        seq_lens_cpu=[n],
        req_pool_indices_cpu=[0],
        residual=None,
    )
    init_attn_metadata(fb)
    out = model.forward_split_prefill(
        torch.tensor(ids), torch.arange(n), fb, (0, NUM_LAYERS)
    )
    keys = torch.stack([pool.k_buffer[i][:n] for i in range(NUM_LAYERS)])
    values = torch.stack([pool.v_buffer[i][:n] for i in range(NUM_LAYERS)])
    return keys, values, out


def stale(position, layer):
    """The deviation a reused token's cached KV carries on ``layer`` (none on layer 0)."""
    if layer == 0:
        return 0.0
    return 0.05 * (1 + (position * 7 + layer * 3) % 53)


class Scenario:
    """One batch through ``forward_subctx_blend``, against each request's full prefill.

    Exact prefixes hold the full prefill's KV; reused runs hold it plus ``deviation``
    from layer 1 on; fresh slots start as NaN.
    """

    def __init__(self, reqs, ratios, check_layer, deviation=stale, seed=0):
        self.model = ToyModel(seed)
        self.reqs = reqs
        self.batch, out_cache_loc = lay_out(reqs)
        self.kv = FakeKVPool()
        self.refs = [reference(self.model, r.fill_ids) for r in reqs]
        self.stale_k = {}
        for req, (keys, values, _out) in zip(reqs, self.refs):
            row = self.batch.req_to_token_pool.req_to_token[req.req_pool_idx]
            if req.sub_context_layout is None:
                held = range(len(req.prefix_indices))
            else:
                held = [p for s, e, _ in req.sub_context_layout for p in range(s, e)]
            exact = req.sub_context_probe_start()
            for p in held:
                slot = int(row[p])
                for layer in range(NUM_LAYERS):
                    shift = 0.0 if p < exact else deviation(p, layer)
                    self.kv.k_buffer[layer][slot] = keys[layer, p] + shift
                    self.kv.v_buffer[layer][slot] = values[layer, p] + shift
                    self.stale_k[(req.rid, p, layer)] = self.kv.k_buffer[layer][
                        slot
                    ].clone()

        self.plan = build_plan(self.batch, out_cache_loc, list(ratios), check_layer)
        starts = [r.sub_context_probe_start() for r in reqs]
        ids = [t for r, s in zip(reqs, starts) for t in r.fill_ids[s:]]
        positions = [p for r, s in zip(reqs, starts) for p in range(s, len(r.fill_ids))]
        self.fb = SimpleNamespace(
            input_ids=torch.tensor(ids),
            positions=torch.tensor(positions),
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=self.kv,
            req_to_token_pool=self.batch.req_to_token_pool,
            extend_seq_lens=torch.zeros(len(reqs), dtype=torch.int32),
            extend_seq_lens_cpu=None,
            seq_lens_cpu=[len(r.fill_ids) for r in reqs],
            req_pool_indices_cpu=[r.req_pool_idx for r in reqs],
            hidden_states=None,
            residual=None,
            subctx_blend_plan=self.plan,
        )

        # Which reused positions ran each layer, recorded as the plan advances.
        self.running = {}
        select = self.plan.select

        def recording_select(cached, fresh):
            self._record()
            return select(cached, fresh)

        self.plan.select = recording_select
        self.out = FakeRunner(self.model).forward_subctx_blend(self.fb)
        self._record()

    def _record(self):
        plan = self.plan
        rows = plan.reused_rows[plan.reused].tolist()
        stage = plan.stage
        layers = (
            range(0, plan.check_layers[0] + 1)
            if stage == 0
            else range(
                plan.check_layers[stage - 1] + 1,
                (plan.check_layers[stage] + 1)
                if stage < len(plan.check_layers)
                else NUM_LAYERS,
            )
        )
        for layer in layers:
            self.running[layer] = set(self.position_of(r) for r in rows)

    def position_of(self, row):
        """``(rid, position)`` of a probe row."""
        plan = self.plan
        i = max(j for j, off in enumerate(plan.row_offsets) if off <= row)
        return self.reqs[i].rid, row - plan.row_offsets[i] + plan.probe_starts[i]

    def slot(self, req, position):
        return int(self.batch.req_to_token_pool.req_to_token[req.req_pool_idx, position])


class TestPlan(unittest.TestCase):
    def test_every_probe_row_writes_to_its_own_slot(self):
        # [4, 20) reused from slot 500 on; [0, 4) and [20, 32) fresh.
        req = make_req(32, [(4, 20, 500)])
        batch, fresh_loc = lay_out([req])
        plan = build_plan(batch, fresh_loc, [0.25], 1)

        loc = plan.probe_cache_loc
        self.assertEqual(loc[plan.reused_rows].tolist(), list(range(500, 516)))
        fresh = torch.tensor(req.sub_context_fresh_positions())
        self.assertEqual(loc[fresh].tolist(), fresh_loc.tolist())
        self.assertEqual(sorted(loc.tolist()), sorted(set(loc.tolist())))

    def test_counts_are_per_request_and_per_scored_layer(self):
        first = make_req(16, [(2, 10, 500)], req_pool_idx=0)
        second = make_req(32, [(4, 20, 600)], req_pool_idx=1)
        batch, fresh_loc = lay_out([first, second], first_slot=700)
        plan = build_plan(batch, fresh_loc, [0.5, 0.25], 1)
        self.assertEqual(plan.check_layers, [1, 2])
        self.assertEqual(plan.keep_counts, [[4, 8], [2, 4]])
        self.assertEqual(plan.fresh_counts, [8, 16])
        plan.start()
        self.assertEqual(plan.lens(), [16, 32])

    def test_selection_keeps_the_top_scorers_nested_and_the_fresh_rows(self):
        first = make_req(16, [(2, 10, 500)], req_pool_idx=0)
        second = make_req(32, [(4, 20, 600)], req_pool_idx=1)
        batch, fresh_loc = lay_out([first, second], first_slot=700)
        plan = build_plan(batch, fresh_loc, [0.5, 0.25], 1)
        plan.start()
        n = plan.reused_rows.numel()

        cached = torch.zeros(n, NUM_HEADS, HEAD_DIM)
        fresh = torch.zeros(n, NUM_HEADS, HEAD_DIM)
        # Scores 0..23 by index, so the top of each request is its last reused rows.
        fresh += torch.arange(n, dtype=torch.float32)[:, None, None]
        keep = plan.select(cached, fresh)
        first_kept = [r for r in plan.reused_rows[plan.reused].tolist() if r < 16]
        second_kept = [
            r - 16 for r in plan.reused_rows[plan.reused].tolist() if r >= 16
        ]
        self.assertEqual(first_kept, [6, 7, 8, 9])
        self.assertEqual(second_kept, list(range(12, 20)))
        self.assertEqual(plan.lens(), [12, 24])
        self.assertEqual(keep.tolist(), plan.rows.tolist())

        # Second layer: scores falling with the index among the survivors.
        m = plan.reused.numel()
        fresh = torch.zeros(m, NUM_HEADS, HEAD_DIM)
        fresh += (m - torch.arange(m, dtype=torch.float32))[:, None, None]
        before = plan.rows.clone()
        keep = plan.select(torch.zeros_like(fresh), fresh)
        kept = plan.reused_rows[plan.reused].tolist()
        self.assertEqual([r for r in kept if r < 16], [6, 7])
        self.assertEqual([r - 16 for r in kept if r >= 16], [12, 13, 14, 15])
        self.assertEqual(before[keep].tolist(), plan.rows.tolist())
        fresh_rows = set(plan.fresh_rows.tolist())
        self.assertTrue(fresh_rows.issubset(set(plan.rows.tolist())))
        self.assertEqual(plan.rows.tolist(), sorted(plan.rows.tolist()))

    def test_a_run_that_is_not_our_copy_is_refused(self):
        req = make_req(32, [(4, 20, 500)], ours=[False])
        batch, fresh_loc = lay_out([req])
        with self.assertRaises(AssertionError):
            build_plan(batch, fresh_loc, [0.25], 1)

    def test_the_probe_starts_past_the_exact_prefix(self):
        # [0, 12) exact, [20, 36) a scanned run; a plain request beside it.
        chained = make_req(40, [(0, 12, 100), (20, 36, 500)], exact=12)
        plain = make_plain_req(30, 700, 25, req_pool_idx=1)
        batch, fresh_loc = lay_out([chained, plain], first_slot=800)
        plan = build_plan(batch, fresh_loc, [0.25], 1)
        self.assertEqual(plan.probe_starts, [12, 25])
        self.assertEqual(plan.probe_lens, [28, 5])
        self.assertEqual(plan.keep_counts, [[4, 0]])
        self.assertEqual(plan.reused_rows.tolist(), list(range(8, 24)))


class TestForward(unittest.TestCase):
    def assert_matches_full_prefill(self, sc):
        for i, req in enumerate(sc.reqs):
            keys, values, out = sc.refs[i]
            start = req.sub_context_probe_start()
            for p in range(start, len(req.fill_ids)):
                slot = sc.slot(req, p)
                for layer in range(NUM_LAYERS):
                    torch.testing.assert_close(
                        sc.kv.k_buffer[layer][slot], keys[layer, p], rtol=0, atol=1e-10
                    )
                    torch.testing.assert_close(
                        sc.kv.v_buffer[layer][slot], values[layer, p], rtol=0, atol=1e-10
                    )
        self.assertEqual(sc.fb.extend_seq_lens_cpu, sc.plan.lens())
        row = 0
        for i, req in enumerate(sc.reqs):
            n = sc.plan.lens()[i]
            start = req.sub_context_probe_start()
            torch.testing.assert_close(
                sc.out[row : row + n],
                sc.refs[i][2][start : start + n],
                rtol=0,
                atol=1e-10,
            )
            row += n

    def test_ratio_one_at_layer_one_is_a_full_prefill(self):
        req = make_req(40, [(0, 6, 100), (10, 24, 500), (28, 36, 530)], exact=6)
        sc = Scenario([req], [1.0], 1)
        self.assertEqual(sc.plan.lens(), [34])
        self.assert_matches_full_prefill(sc)

    def test_ratio_one_at_layer_zero_is_a_full_prefill(self):
        req = make_req(40, [(10, 24, 500), (28, 36, 530)])
        sc = Scenario([req], [1.0], 0)
        self.assert_matches_full_prefill(sc)

    def test_ratio_one_twice_is_a_full_prefill_with_a_plain_request_beside(self):
        chained = make_req(40, [(0, 6, 100), (10, 24, 500)], exact=6)
        plain = make_plain_req(30, 300, 25, req_pool_idx=1, ids=range(101, 131))
        sc = Scenario([chained, plain], [1.0, 1.0], 1)
        self.assert_matches_full_prefill(sc)

    def test_dropped_tokens_keep_their_cached_kv_and_running_ones_are_rewritten(self):
        req = make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        sc = Scenario([req], [0.5, 0.25], 1)
        reused = [p for s, e, _ in req.sub_context_layout[1:] for p in range(s, e)]
        for p in reused:
            slot = sc.slot(req, p)
            for layer in range(NUM_LAYERS):
                held = sc.kv.k_buffer[layer][slot]
                cached = sc.stale_k[(req.rid, p, layer)]
                if (req.rid, p) in sc.running[layer]:
                    if layer > 0:
                        self.assertFalse(torch.equal(held, cached), (p, layer))
                else:
                    self.assertTrue(torch.equal(held, cached), (p, layer))
        # Every fresh slot is written on every layer.
        for p in req.sub_context_fresh_positions():
            for layer in range(NUM_LAYERS):
                self.assertFalse(
                    bool(torch.isnan(sc.kv.k_buffer[layer][sc.slot(req, p)]).any())
                )
        self.assertFalse(bool(torch.isnan(sc.out).any()))

    def test_each_scored_layer_picks_by_its_own_deviation(self):
        req = make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        sc = Scenario([req], [0.5, 0.25], 1)
        reused = [p for s, e, _ in req.sub_context_layout[1:] for p in range(s, e)]
        first = sorted(reused, key=lambda p: -stale(p, 1))[:15]
        second = sorted(first, key=lambda p: -stale(p, 2))[:7]
        self.assertEqual({p for _rid, p in sc.running[2]}, set(first))
        self.assertEqual({p for _rid, p in sc.running[3]}, set(second))
        self.assertEqual({p for _rid, p in sc.running[4]}, set(second))
        # The layers both scored layers ran on all probe rows match the full prefill.
        keys, _values, _out = sc.refs[0]
        for p in first:
            for layer in (0, 1, 2):
                torch.testing.assert_close(
                    sc.kv.k_buffer[layer][sc.slot(req, p)],
                    keys[layer, p],
                    rtol=0,
                    atol=1e-10,
                )

    def test_the_last_row_of_each_request_is_its_last_position(self):
        chained = make_req(40, [(0, 6, 100), (10, 24, 500)], exact=6)
        plain = make_plain_req(30, 300, 25, req_pool_idx=1, ids=range(101, 131))
        sc = Scenario([chained, plain], [0.5, 0.2], 1)
        cursor = 0
        for i, req in enumerate(sc.reqs):
            cursor += sc.plan.lens()[i]
            self.assertEqual(int(sc.fb.positions[cursor - 1]), len(req.fill_ids) - 1)
        self.assertEqual(sc.fb.positions.numel(), sum(sc.plan.lens()))


if __name__ == "__main__":
    unittest.main()
