"""Unit tests for RelayCaching on reused sub-context tokens (``subctx_relay``).

Checked on the selection rules alone, with hand-computed expectations:

- the mean threshold is strict and per group; a zero group keeps none, tau 0 keeps all,
- value deviation keeps the rows above tau times the request's mean plus the preselected
  ones; a request with none above keeps none; a request whose V did not move defers,
- each span keeps its suffix and its top-attended rows within ceil(len * p), the suffix
  counted inside the budget, candidates capped at 256,
- contiguous scored runs form one span.

Checked through ``ModelRunner.forward_subctx_relay`` on the small attention model of
``test_subctx_blend``:

- selecting every reused row through the last layer is a full prefill, from layer 0 or
  with exact stored hidden states loaded at L_start,
- a reused row with a stored hidden state keeps its cached KV below L_start, an unselected
  one above L_det, a selected one above L_end,
- a request whose cached V matches at L_det selects one layer later,
- the profile records every layer and rewrites every reused row.

Checked on a GPU: the influence kernel against its torch reference.

Usage:
    python test_subctx_relay.py
    python -m pytest test_subctx_relay.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")

import json
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_subctx_blend as blend  # noqa: E402

from sglang.srt.mem_cache import subctx_relay  # noqa: E402
from sglang.srt.mem_cache.subctx_relay import (  # noqa: E402
    RelayState,
    accumulate_influence,
    build_relay_plan,
    deviation_select,
    influence_preselect,
    influence_reference,
    mean_threshold_mask,
)
from sglang.srt.model_executor.model_runner import ModelRunner  # noqa: E402
from sglang.srt.utils import subctx_config  # noqa: E402

NUM_LAYERS = blend.NUM_LAYERS
DTYPE = blend.DTYPE


def rotated(vectors: torch.Tensor, angles) -> torch.Tensor:
    """``vectors`` ([n, 1, 2]) each turned by its angle, so cos(new, old) = cos(angle)."""
    a = torch.tensor(angles, dtype=vectors.dtype)
    x, y = vectors[:, 0, 0], vectors[:, 0, 1]
    return torch.stack([x * a.cos() - y * a.sin(), x * a.sin() + y * a.cos()], 1)[:, None]


def with_deviations(devs):
    """Cached and fresh V of one head whose ``1 - cos`` are ``devs``."""
    cached = torch.tensor([[[1.0, 0.0]]] * len(devs), dtype=torch.float64)
    return cached, rotated(cached, [math.acos(1 - d) for d in devs])


class TestSelection(unittest.TestCase):
    def test_mean_threshold_is_strict_per_group_and_has_both_ends(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 10.0, 6.0, 6.0, 0.0, 0.0])
        group = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        mask = mean_threshold_mask(scores, group, 3, 1.5)
        # Group 0: mean 4, cutoff 6 -> only 10. Group 1: 6 is not above 9. Group 2: zero.
        self.assertEqual(mask.tolist(), [False, False, False, True, False, False, False, False])
        self.assertTrue(bool(mean_threshold_mask(scores, group, 3, 0.0).all()))
        # Exactly at the cutoff is not above it: mean 2, tau 1.5.
        self.assertEqual(
            mean_threshold_mask(torch.tensor([1.0, 3.0, 2.0]), torch.zeros(3, dtype=torch.long), 1, 1.5).tolist(),
            [False, False, False],
        )

    def test_deviation_keeps_the_outliers_and_the_preselected(self):
        cached, fresh = with_deviations([0.1, 0.1, 0.1, 0.7])
        pre = torch.tensor([True, False, False, False])
        keep, deferred = deviation_select(
            cached, fresh, torch.zeros(4, dtype=torch.long), 1, 1.5, pre
        )
        # Mean 0.25, cutoff 0.375: row 3, joined by the preselected row 0.
        self.assertEqual(keep.tolist(), [True, False, False, True])
        self.assertEqual(deferred.tolist(), [False])

    def test_no_outlier_drops_the_preselected_too(self):
        cached, fresh = with_deviations([0.2, 0.2, 0.2])
        pre = torch.tensor([True, False, True])
        keep, deferred = deviation_select(
            cached, fresh, torch.zeros(3, dtype=torch.long), 1, 1.5, pre
        )
        self.assertEqual(keep.tolist(), [False, False, False])
        self.assertEqual(deferred.tolist(), [False])

    def test_an_unmoved_request_keeps_all_and_defers_beside_a_moved_one(self):
        c0, f0 = with_deviations([0.1, 0.1, 0.1, 0.7])
        c1 = torch.tensor([[[0.0, 1.0]]] * 3, dtype=torch.float64)
        cached, fresh = torch.cat([c0, c1]), torch.cat([f0, c1.clone()])
        req = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        keep, deferred = deviation_select(cached, fresh, req, 2, 1.5, torch.zeros(7, dtype=torch.bool))
        self.assertEqual(keep.tolist(), [False, False, False, True, True, True, True])
        self.assertEqual(deferred.tolist(), [False, True])

    def test_bf16_defers_a_deviation_fp32_would_select_on(self):
        # Mean deviation ~5e-4, a few rows far above it: the bf16 mean cosine rounds to
        # 1.0 (official behaviour), fp32 selects the outliers.
        g = torch.Generator().manual_seed(0)
        old = torch.randn(400, 4, 128, generator=g)
        scale = 0.02 + 0.06 * torch.rand(400, 1, 1, generator=g) ** 6
        new = old + scale * torch.randn(400, 4, 128, generator=g)
        req = torch.zeros(400, dtype=torch.long)
        none = torch.zeros(400, dtype=torch.bool)
        keep, deferred = deviation_select(old.bfloat16(), new.bfloat16(), req, 1, 1.5, none)
        self.assertEqual(deferred.tolist(), [True])
        self.assertTrue(bool(keep.all()))
        keep, deferred = deviation_select(old, new, req, 1, 1.5, none)
        self.assertEqual(deferred.tolist(), [False])
        self.assertTrue(0 < int(keep.sum()) < 400)
        # bf16 per-token scores take few distinct values.
        self.assertLess(subctx_relay.value_deviation(old.bfloat16(), new.bfloat16()).unique().numel(), 20)

    def preselect(self, scores_per_span, tau, suffix):
        lens = torch.tensor([len(s) for s in scores_per_span])
        span_id = torch.repeat_interleave(torch.arange(len(lens)), lens)
        first = torch.cumsum(lens, 0) - lens
        off = torch.arange(int(lens.sum())) - first[span_id]
        scores = torch.cat([torch.tensor(s, dtype=torch.float32) for s in scores_per_span])
        picked = influence_preselect(scores, span_id, off, lens, tau, suffix)
        return [set((picked[first[i] : first[i] + lens[i]]).nonzero().flatten().tolist()) for i in range(len(lens))]

    def test_the_suffix_counts_inside_the_influence_budget(self):
        scores = [0, 0, 5, 0, 0, 1, 0, 0, 0, 0]
        # Mean 0.6, cutoff 0.87: 2 of 10 pass, budget ceil(10 * 0.2) = 2, all suffix.
        self.assertEqual(self.preselect([scores], 1.45, 2), [{8, 9}])
        self.assertEqual(self.preselect([scores], 1.45, 0), [{2, 5}])
        self.assertEqual(self.preselect([scores], 1.45, 1), [{2, 9}])

    def test_candidates_are_the_top_256_of_a_long_span(self):
        # Top 256 are 44..299, mean 171.5, cutoff 248.675: 51 pass, p = 51/256,
        # budget ceil(300 p) = 60, the last 10 inside it, 50 by rank: 250..299.
        picked = self.preselect([list(range(300))], 1.45, 10)[0]
        self.assertEqual(picked, set(range(250, 300)))

    def test_each_span_has_its_own_budget_and_a_zero_span_keeps_its_suffix(self):
        picked = self.preselect([[0, 0, 5, 0, 0, 1, 0, 0, 0, 0], [0.0] * 6], 1.45, 0)
        self.assertEqual(picked, [{2, 5}, set()])
        # Budget 2 against a suffix of 3: the suffix alone.
        picked = self.preselect([[0, 0, 5, 0, 0, 1, 0, 0, 0, 0], [0.0] * 6], 1.45, 3)
        self.assertEqual(picked, [{7, 8, 9}, {3, 4, 5}])

    def test_contiguous_runs_form_one_span(self):
        req = blend.make_req(48, [(0, 4, 100), (8, 20, 500), (20, 30, 520), (34, 40, 540)], exact=4)
        self.assertEqual(req.sub_context_reused_span_lens(), [22, 6])
        batch, fresh_loc = blend.lay_out([req])
        plan = build_relay_plan(batch, fresh_loc, (1, 2, 3))
        self.assertEqual(plan.span_len.tolist(), [22, 6])
        self.assertEqual(plan.span_off.tolist(), list(range(22)) + list(range(6)))
        self.assertEqual(plan.req_of_reused.tolist(), [0] * 28)


class RelayKVPool(blend.FakeKVPool):
    def __init__(self, state):
        super().__init__()
        self.relay_state = state

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]


class RelayRunner(blend.FakeRunner):
    forward_subctx_relay = ModelRunner.forward_subctx_relay


def reference_streams(model, ids):
    """Each position's residual stream at the input of every layer of a full prefill."""
    n = len(ids)
    pool = blend.FakeKVPool()
    r2t = blend.FakeReqToTokenPool()
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
    blend.init_attn_metadata(fb)
    streams = []
    hidden = model.embed[torch.tensor(ids)] + model.pos[torch.arange(n)]
    residual = None
    for layer in range(NUM_LAYERS):
        streams.append(hidden.clone() if residual is None else hidden + residual)
        hidden, residual = model.layer(layer, torch.arange(n), hidden, fb, residual)
    return streams


class RelayScenario:
    """One batch through ``forward_subctx_relay`` against each request's full prefill.

    Exact prefixes hold the full prefill's KV; reused rows hold it plus ``deviation``;
    ``decoded(position)`` marks the reused rows with a stored hidden state, stored exact.
    """

    def __init__(
        self,
        reqs,
        layers,
        tau_dev=1.5,
        tau_inf=1.45,
        suffix=0,
        deviation=blend.stale,
        decoded=lambda p: False,
        influence=lambda p: 0.0,
        profile_path=None,
        logprob_starts=None,
    ):
        self.model = blend.ToyModel(0)
        self.reqs = reqs
        self.layers = layers
        self.batch, out_cache_loc = blend.lay_out(reqs)
        start = layers[0] if layers is not None else 0
        self.state = RelayState(
            blend.NUM_SLOTS, 4, blend.HIDDEN, DTYPE, "cpu", start_layer=start
        )
        self.kv = RelayKVPool(self.state)
        self.refs = [blend.reference(self.model, r.fill_ids) for r in reqs]
        self.cached = {}
        for req, (keys, values, _out) in zip(reqs, self.refs):
            streams = reference_streams(self.model, req.fill_ids)
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
                    self.cached[(req.rid, p, layer)] = (
                        self.kv.k_buffer[layer][slot].clone(),
                        self.kv.v_buffer[layer][slot].clone(),
                    )
                if p >= exact:
                    self.state.decoded[slot] = bool(decoded(p))
                    self.state.influence[slot] = float(influence(p))
                    if self.state.hidden is not None:
                        self.state.hidden[slot] = streams[start][p]

        self.plan = build_relay_plan(self.batch, out_cache_loc, layers)
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
            extend_seq_lens_cpu=[len(r.sub_context_fresh_positions()) for r in reqs],
            seq_lens_cpu=[len(r.fill_ids) for r in reqs],
            req_pool_indices_cpu=[r.req_pool_idx for r in reqs],
            hidden_states=None,
            residual=None,
            subctx_blend_plan=self.plan,
            return_logprob=logprob_starts is not None,
            extend_logprob_start_lens_cpu=logprob_starts,
        )
        saved = (
            subctx_config.RELAY_TAU_DEV,
            subctx_config.RELAY_TAU_INF,
            subctx_config.RELAY_SUFFIX,
            subctx_config.RELAY_PROFILE,
        )
        subctx_config.RELAY_TAU_DEV = tau_dev
        subctx_config.RELAY_TAU_INF = tau_inf
        subctx_config.RELAY_SUFFIX = suffix
        subctx_config.RELAY_PROFILE = profile_path or ""

        # Which positions ran each layer, from the K/V the layers wrote.
        self.ran = {layer: set() for layer in range(NUM_LAYERS)}
        model_layer = self.model.layer

        def recording_layer(i, positions, hidden, fb, residual):
            for p in positions.tolist():
                self.ran[i].add(p)
            return model_layer(i, positions, hidden, fb, residual)

        self.model.layer = recording_layer
        try:
            self.out = RelayRunner(self.model).forward_subctx_relay(self.fb)
        finally:
            (
                subctx_config.RELAY_TAU_DEV,
                subctx_config.RELAY_TAU_INF,
                subctx_config.RELAY_SUFFIX,
                subctx_config.RELAY_PROFILE,
            ) = saved

    def slot(self, req, position):
        return int(self.batch.req_to_token_pool.req_to_token[req.req_pool_idx, position])


def reused_positions(req):
    return [p for s, e, _ in req.sub_context_layout if s >= req.sub_context_exact_len for p in range(s, e)]


class TestForward(unittest.TestCase):
    def assert_full_prefill(self, sc):
        for i, req in enumerate(sc.reqs):
            keys, values, out = sc.refs[i]
            for p in range(req.sub_context_probe_start(), len(req.fill_ids)):
                for layer in range(NUM_LAYERS):
                    slot = sc.slot(req, p)
                    torch.testing.assert_close(sc.kv.k_buffer[layer][slot], keys[layer, p], rtol=0, atol=1e-10)
                    torch.testing.assert_close(sc.kv.v_buffer[layer][slot], values[layer, p], rtol=0, atol=1e-10)
        row = 0
        for i, req in enumerate(sc.reqs):
            n = sc.fb.extend_seq_lens_cpu[i]
            start = req.sub_context_probe_start()
            # The last rows are fresh: compare each request's last row.
            torch.testing.assert_close(sc.out[row + n - 1], sc.refs[i][2][len(req.fill_ids) - 1], rtol=0, atol=1e-10)
            row += n

    def test_selecting_everything_from_layer_zero_is_a_full_prefill(self):
        req = blend.make_req(40, [(0, 6, 100), (10, 24, 500), (28, 36, 530)], exact=6)
        plain = blend.make_plain_req(30, 300, 25, req_pool_idx=1, ids=range(101, 131))
        sc = RelayScenario([req, plain], (0, 1, NUM_LAYERS - 1), tau_dev=0.0)
        self.assert_full_prefill(sc)

    def test_exact_stored_hidden_states_load_into_a_full_prefill(self):
        req = blend.make_req(40, [(0, 6, 100), (10, 24, 500), (28, 36, 530)], exact=6)
        sc = RelayScenario(
            [req],
            (2, 3, NUM_LAYERS - 1),
            tau_dev=0.0,
            deviation=lambda p, layer: 0.0 if layer < 2 else blend.stale(p, layer),
            decoded=lambda p: p % 3 != 0,
        )
        self.assert_full_prefill(sc)
        warm = {p for p in reused_positions(req) if p % 3 != 0}
        self.assertTrue(warm)
        for layer in (0, 1):
            self.assertFalse(warm & sc.ran[layer], layer)
        for layer in range(2, NUM_LAYERS):
            self.assertTrue(warm <= sc.ran[layer], layer)

    def test_each_reused_row_keeps_its_cached_kv_where_it_did_not_run(self):
        req = blend.make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        start, det, end = 1, 2, 3
        sc = RelayScenario(
            [req], (start, det, end), tau_dev=1.2, suffix=2, decoded=lambda p: p % 2 == 0
        )
        reused = reused_positions(req)
        selected = sc.ran[det + 1] & set(reused)
        self.assertTrue(0 < len(selected) < len(reused))
        # The last 2 of each span are selected: spans [8, 24) and [30, 44).
        self.assertTrue({22, 23, 42, 43} <= selected)
        for p in reused:
            slot = sc.slot(req, p)
            for layer in range(NUM_LAYERS):
                cached_k, cached_v = sc.cached[(req.rid, p, layer)]
                ran = (
                    (layer < start and p % 2 == 1)
                    or start <= layer <= det
                    or (det < layer <= end and p in selected)
                )
                self.assertEqual(p in sc.ran[layer], ran, (p, layer))
                held = sc.kv.v_buffer[layer][slot]
                if ran and layer > 0:
                    self.assertFalse(torch.equal(held, cached_v), (p, layer))
                elif not ran:
                    self.assertTrue(torch.equal(held, cached_v), (p, layer))
        # Fresh rows run every layer and the last row is the last position.
        for p in req.sub_context_fresh_positions():
            for layer in range(NUM_LAYERS):
                self.assertIn(p, sc.ran[layer])
        self.assertEqual(int(sc.fb.positions[-1]), len(req.fill_ids) - 1)
        self.assertEqual(sc.fb.extend_seq_lens_cpu, [len(req.sub_context_fresh_positions())])

    def test_the_selected_rows_are_the_deviation_outliers_and_the_top_attended(self):
        req = blend.make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        det = 2
        sc = RelayScenario(
            [req],
            (1, det, NUM_LAYERS - 1),
            tau_dev=1.2,
            suffix=0,
            influence=lambda p: 10.0 if p in (9, 31) else 0.0,
        )
        reused = reused_positions(req)
        _keys, values, _out = sc.refs[0]
        fresh = torch.stack([values[det, p] for p in reused])
        cached = torch.stack([sc.cached[(req.rid, p, det)][1] for p in reused])
        score = subctx_relay.value_deviation(cached, fresh)
        outliers = {p for p, s in zip(reused, score.tolist()) if s > float(score.mean()) * 1.2}
        self.assertTrue(outliers)
        self.assertEqual(sc.ran[det + 1] & set(reused), outliers | {9, 31})

    def test_an_unmoved_request_selects_one_layer_later(self):
        req = blend.make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        det, end = 2, 4
        sc = RelayScenario(
            [req],
            (1, det, end),
            tau_dev=1.2,
            deviation=lambda p, layer: 0.0 if layer <= det else blend.stale(p, layer),
        )
        reused = set(reused_positions(req))
        self.assertEqual(sc.ran[det + 1] & reused, reused)
        later = sc.ran[det + 2] & reused
        self.assertTrue(0 < len(later) < len(reused))

    def test_logprob_starts_follow_the_rows_left_in_the_last_stage(self):
        # A sparse request beside a plain one that returns logprobs for its last 3
        # computed tokens; L_end is the last layer, so the sparse one ends with
        # fresh + selected rows.
        chained = blend.make_req(40, [(0, 6, 100), (10, 24, 500)], exact=6)
        plain = blend.make_plain_req(30, 300, 25, req_pool_idx=1, ids=range(101, 131))
        fresh = [len(chained.sub_context_fresh_positions()), 5]
        sc = RelayScenario(
            [chained, plain],
            (1, 2, NUM_LAYERS - 1),
            tau_dev=1.2,
            logprob_starts=[fresh[0], 5 - 3],
        )
        lens = sc.fb.extend_seq_lens_cpu
        self.assertGreater(lens[0], fresh[0])
        self.assertEqual(sc.fb.extend_logprob_start_lens_cpu, [lens[0], lens[1] - 3])

    def test_the_tallies_count_rows_that_ran_to_the_last_layer(self):
        req = blend.make_req(48, [(0, 4, 100), (8, 24, 500), (30, 44, 530)], exact=4)
        reused = set(reused_positions(req))
        # L_det is the last layer: every reused row runs through it.
        sc = RelayScenario([req], (1, NUM_LAYERS - 1, NUM_LAYERS - 1), tau_dev=1.2)
        self.assertEqual(sc.state.requests, 1)
        self.assertEqual(sc.state.reused_rows, len(reused))
        self.assertEqual(sc.state.selected_rows, len(reused))
        # L_end is the last layer: the rows that ran past L_det.
        det = 2
        sc = RelayScenario([req], (1, det, NUM_LAYERS - 1), tau_dev=1.2)
        self.assertEqual(sc.state.selected_rows, len(sc.ran[det + 1] & reused))
        self.assertTrue(0 < sc.state.selected_rows < len(reused))

    def test_the_profile_records_every_layer_and_recomputes_every_row(self):
        req = blend.make_req(40, [(0, 6, 100), (10, 24, 500), (28, 36, 530)], exact=6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile.jsonl")
            sc = RelayScenario([req], None, decoded=lambda p: p < 20, profile_path=path)
            with open(path) as fh:
                lines = [json.loads(line) for line in fh]
        self.assert_full_prefill(sc)
        # One line per span: [10, 24) with 10 decoded rows, [28, 36) with none.
        self.assertEqual([(r["span"], r["n_reused"], r["n_decoded"]) for r in lines], [(0, 14, 10), (1, 8, 0)])
        self.assertIsNone(lines[1]["dec"])
        rec = lines[0]
        for group in ("all", "dec"):
            self.assertEqual(len(rec[group]["s_tok"]), NUM_LAYERS)
            self.assertEqual(len(rec[group]["s_code"]), NUM_LAYERS)
            self.assertEqual(len(rec[group]["rho"]), NUM_LAYERS - 1)
        # Layer 0's cached V carries no deviation (blend.stale).
        self.assertAlmostEqual(rec["all"]["s_tok"][0], 1.0, places=9)
        self.assertLess(rec["all"]["s_tok"][1], 1.0)
        # Each span is measured on its own rows: span 0's s_code from its rows only.
        keys, values, _out = sc.refs[0]
        layer = 2
        rows = list(range(10, 24))
        fresh = torch.stack([values[layer, p] for p in rows])
        cached = torch.stack([sc.cached[(req.rid, p, layer)][1] for p in rows])
        want = float(torch.nn.functional.cosine_similarity(fresh, cached, dim=0, eps=1e-8).mean())
        self.assertAlmostEqual(rec["all"]["s_code"][layer], want, places=9)

    def test_spearman_matches_a_hand_count(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(subctx_relay._spearman(a, a * 3), 1.0)
        self.assertAlmostEqual(subctx_relay._spearman(a, -a), -1.0)
        # Ranks [0..4] vs [1, 0, 2, 4, 3]: sum d^2 = 4, rho = 1 - 6 * 4 / (5 * 24) = 0.8.
        b = torch.tensor([2.0, 1.0, 3.0, 5.0, 4.0])
        self.assertAlmostEqual(subctx_relay._spearman(a, b), 0.8)
        self.assertEqual(subctx_relay._spearman(a, torch.ones(5)), 1.0)
        # Ties share their mean rank: [1.5, 1.5, 3, 4] vs [1, 2.5, 2.5, 4] -> 3.75 / 4.5.
        ties_a = torch.tensor([1.0, 1.0, 2.0, 3.0])
        ties_b = torch.tensor([1.0, 2.0, 2.0, 3.0])
        self.assertAlmostEqual(subctx_relay._spearman(ties_a, ties_b), 3.75 / 4.5)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestInfluenceKernel(unittest.TestCase):
    def test_matches_the_reference_with_gqa_and_empty_rows(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for q_heads, kv_heads, head_dim in ((8, 2, 128), (28, 4, 128), (4, 4, 64)):
            slots, pool_size, ctx = 4096, 6, 2048
            k_buffer = torch.randn(slots, kv_heads, head_dim, generator=g, device="cuda").bfloat16()
            req_to_token = torch.randint(1, slots, (pool_size, ctx), generator=g, device="cuda", dtype=torch.int32)
            # Rows: output keys; none (padding); none; output keys; a reused pool entry.
            req_pool_indices = torch.tensor([3, 0, 5, 1, 2], device="cuda")
            seq_lens = torch.tensor([700, 1, 33, 1500, 600], device="cuda")
            out_start = torch.full((pool_size,), 1 << 62, dtype=torch.int64, device="cuda")
            out_start[3], out_start[0], out_start[5], out_start[1], out_start[2] = 500, 5, 33, 10, 100
            out_cache_loc = req_to_token[req_pool_indices, seq_lens - 1].long()
            out_cache_loc[4] += 1
            q = torch.randn(5, q_heads, head_dim, generator=g, device="cuda").bfloat16()
            args = (k_buffer, req_to_token, req_pool_indices, seq_lens, out_cache_loc, out_start)
            got = torch.zeros(slots, device="cuda")
            want = torch.zeros(slots, device="cuda")
            accumulate_influence(q, *args, got, head_dim**-0.5)
            influence_reference(q, *args, want, head_dim**-0.5)
            torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-4)
            # Each of the two live requests adds q_heads in total; the reused entry nothing.
            self.assertAlmostEqual(float(got.sum()), 2 * q_heads, delta=1e-2)


if __name__ == "__main__":
    unittest.main()
