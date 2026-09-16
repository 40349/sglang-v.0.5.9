"""
Unit tests for the sub-context index.

The index answers one question -- *which registered sub-contexts occur in this query,
and where* -- and everything downstream trusts that answer. These tests hold it to the
three properties the rest of the feature is built on:

- a chunk's identity is its token ids (and ``extra_key``) and **nothing else**, because
  finding the same run of tokens at a new position is the entire point; where it lands
  is the rotation's problem, not the index's,
- scanning finds every occurrence, checked against brute force on random inputs, since
  the fingerprint that drives it is a filter and a filter that silently drops a match
  costs reuse with no symptom,
- the subset chosen for reuse covers as many tokens as any non-overlapping subset
  could, checked against exhaustive enumeration.

Usage:
    python test_subctx_index.py
    python -m pytest test_subctx_index.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")

import itertools
import random
import unittest

import numpy as np

from sglang.srt.mem_cache.subctx_index import (
    ANCHOR_TOKENS,
    Match,
    SubContextIndex,
    anchor_fingerprints,
    chunk_id,
    cut_points,
)


def _brute_force(query, chunks):
    """Every occurrence of every chunk, found the slow obvious way."""
    found = []
    for cid, chunk in chunks.items():
        for start in range(len(query) - len(chunk) + 1):
            if query[start : start + len(chunk)] == chunk:
                found.append(Match(start, start + len(chunk), cid))
    return sorted(found)


class TestChunkId(unittest.TestCase):
    """What is, and is not, part of a chunk's address."""

    def setUp(self):
        self.rng = random.Random(20260915)
        self.ids = [self.rng.randrange(150000) for _ in range(100)]

    def test_position_is_not_part_of_the_identity(self):
        """The property the whole index rests on.

        A chunk cached at position 120 in one prompt has to be *the same chunk* when it
        turns up at position 100 in the next, or the scan finds nothing and every
        prompt is prefilled from scratch. Folding the offset in would be the bug.
        """
        head = [self.rng.randrange(150000) for _ in range(37)]
        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        cid = index.register(self.ids)

        at_zero = index.scan(self.ids)
        displaced = index.scan(head + self.ids)

        self.assertEqual([(m.start, m.chunk_id) for m in at_zero], [(0, cid)])
        self.assertEqual([(m.start, m.chunk_id) for m in displaced], [(37, cid)])

    def test_extra_key_separates_tenants(self):
        """``extra_key`` carries cache_salt *and* lora_id (see ``Req.__init__``).

        A LoRA adapter's keys and values are not the base model's, so the same tokens
        under a different adapter must be a different chunk. The pre-index code keyed
        blocks by role and dropped ``extra_key`` entirely, which shared them.
        """
        self.assertNotEqual(chunk_id(self.ids), chunk_id(self.ids, "salt:a"))
        self.assertNotEqual(chunk_id(self.ids, "salt:a"), chunk_id(self.ids, "salt:b"))

        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        index.register(self.ids, "tenant-a")
        self.assertEqual(index.scan(self.ids, "tenant-b"), [])
        self.assertEqual(len(index.scan(self.ids, "tenant-a")), 1)

    def test_ids_are_hashed_at_a_fixed_width(self):
        """A varint encoding would let ``[1, 23]`` and ``[12, 3]`` share bytes."""
        self.assertNotEqual(chunk_id([1, 23]), chunk_id([12, 3]))
        self.assertNotEqual(chunk_id([300, 1]), chunk_id([1, 300]))

    def test_a_collision_does_not_reuse_the_wrong_kv(self):
        """Why a 64-bit address is safe here.

        ``chunk_id`` only picks the namespace; the radix tree still walks the token ids
        inside it, and the scan verifies them too. So two different runs of tokens
        forced onto one id find their own KV, not each other's -- a collision costs a
        wasted comparison, never a wrong reuse.
        """
        first = [self.rng.randrange(150000) for _ in range(80)]
        second = [self.rng.randrange(150000) for _ in range(80)]
        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        index.register(first)
        index.register(second)

        # Force both onto the one fingerprint the query will produce, which is what a
        # collision does; the token-by-token check downstream is all that separates
        # them.
        collided = int(
            anchor_fingerprints(np.array(second[:ANCHOR_TOKENS], dtype=np.int32))[0]
        )
        scope = index._scopes[None]
        scope.by_anchor.clear()
        scope.by_anchor[collided] = [chunk_id(first), chunk_id(second)]
        scope.drop_anchor()

        hits = index.scan([0] * 5 + second)
        self.assertEqual([(m.start, m.chunk_id) for m in hits], [(5, chunk_id(second))])


class TestFingerprints(unittest.TestCase):
    """The filter that turns a per-position question into one array pass."""

    def test_a_window_fingerprints_the_same_wherever_it_sits(self):
        rng = random.Random(11)
        window = [rng.randrange(150000) for _ in range(ANCHOR_TOKENS)]
        early = anchor_fingerprints(np.array([9] * 77 + window, dtype=np.int32))
        late = anchor_fingerprints(np.array([4] * 5 + window + [8] * 900, dtype=np.int32))
        self.assertEqual(early[77], late[5])

    def test_the_two_ends_of_a_window_carry_different_weight(self):
        """A repeating weight would make the fingerprint blind to a swap."""
        rng = random.Random(12)
        window = [rng.randrange(150000) for _ in range(ANCHOR_TOKENS)]
        swapped = list(window)
        swapped[0], swapped[ANCHOR_TOKENS - 1] = swapped[ANCHOR_TOKENS - 1], swapped[0]
        self.assertNotEqual(
            anchor_fingerprints(np.array(window, dtype=np.int32))[0],
            anchor_fingerprints(np.array(swapped, dtype=np.int32))[0],
        )

    def test_a_sequence_shorter_than_one_window_has_no_fingerprints(self):
        short = np.arange(ANCHOR_TOKENS - 1, dtype=np.int32)
        self.assertEqual(anchor_fingerprints(short).shape[0], 0)
        exact = np.arange(ANCHOR_TOKENS, dtype=np.int32)
        self.assertEqual(anchor_fingerprints(exact).shape[0], 1)


class TestScan(unittest.TestCase):
    """Finding occurrences, held against brute force."""

    def test_matches_brute_force_on_random_inputs(self):
        """A small alphabet so overlaps and nesting actually occur.

        The fingerprint is only a filter, and a filter that drops a real match costs
        reuse without ever failing -- so the check has to be exhaustive equality, not a
        spot test of a case someone thought of.
        """
        rng = random.Random(20260915)
        for trial in range(300):
            alphabet = list(range(6))
            index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
            chunks = {}
            for _ in range(rng.randrange(1, 7)):
                length = rng.randrange(ANCHOR_TOKENS, ANCHOR_TOKENS + 20)
                chunk = [rng.choice(alphabet) for _ in range(length)]
                cid = index.register(chunk)
                if cid is not None:
                    chunks[cid] = chunk

            query = [rng.choice(alphabet) for _ in range(rng.randrange(0, 250))]
            for chunk in list(chunks.values()):
                if rng.random() < 0.7 and query:
                    at = rng.randrange(0, len(query))
                    query = query[:at] + chunk + query[at:]

            with self.subTest(trial=trial):
                self.assertEqual(sorted(index.scan(query)), _brute_force(query, chunks))

    def test_a_run_too_short_to_be_worth_reusing_is_refused(self):
        index = SubContextIndex(min_chunk_tokens=64)
        self.assertIsNone(index.register(list(range(63))))
        self.assertIsNotNone(index.register(list(range(64))))

    def test_registering_the_same_run_twice_is_one_chunk(self):
        """The common case: two requests computed it independently."""
        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        ids = list(range(80))
        self.assertEqual(index.register(ids), index.register(ids))
        self.assertEqual(len(index), 1)

    def test_unregister_makes_it_unfindable_and_tolerates_a_miss(self):
        """Eviction and registration race; neither side is authoritative."""
        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        ids = list(range(100, 180))
        cid = index.register(ids)
        index.unregister(cid)
        self.assertEqual(index.scan([0] * 5 + ids + [0] * 5), [])
        index.unregister(cid)  # again
        index.unregister("sc:never-existed")

    def test_registrations_interleaved_with_scans_stay_findable(self):
        """The prefilter is grown in place as chunks arrive, and rebuilt when it is
        outgrown or an entry is dropped. Either path losing a chunk would show up as
        reuse quietly falling off, so walk both while scanning between every step."""
        rng = random.Random(4242)
        index = SubContextIndex(min_chunk_tokens=ANCHOR_TOKENS)
        live = {}
        for step in range(200):
            if live and rng.random() < 0.25:
                cid = rng.choice(list(live))
                index.unregister(cid)
                del live[cid]
            else:
                chunk = [rng.randrange(1000) for _ in range(ANCHOR_TOKENS + 8)]
                cid = index.register(chunk)
                if cid is not None:
                    live[cid] = chunk
            if live:
                cid = rng.choice(list(live))
                query = [0] * 3 + live[cid] + [0] * 3
                with self.subTest(step=step):
                    self.assertIn(cid, {m.chunk_id for m in index.scan(query)})


class TestSelect(unittest.TestCase):
    """Choosing which of the overlapping occurrences to actually reuse."""

    def test_covers_as_many_tokens_as_any_disjoint_subset(self):
        """Against exhaustive enumeration.

        Covered tokens is the quantity to maximise because it is exactly the prefill
        skipped. Greedy-by-length and greedy-by-leftmost both lose to it, so the check
        is the optimum rather than a plausible-looking answer.
        """
        rng = random.Random(777)
        for trial in range(400):
            matches = [
                Match(start, start + rng.randrange(1, 15), f"sc:{k}")
                for k in range(rng.randrange(0, 8))
                for start in (rng.randrange(0, 40),)
            ]
            chosen = SubContextIndex.select(matches)
            ordered = sorted(chosen, key=lambda m: m.start)

            best = 0
            for size in range(len(matches) + 1):
                for combo in itertools.combinations(matches, size):
                    run = sorted(combo, key=lambda m: m.start)
                    if all(run[i].end <= run[i + 1].start for i in range(len(run) - 1)):
                        best = max(best, sum(m.length for m in run))

            with self.subTest(trial=trial):
                self.assertTrue(
                    all(
                        ordered[i].end <= ordered[i + 1].start
                        for i in range(len(ordered) - 1)
                    ),
                    f"overlapping selection {ordered}",
                )
                self.assertEqual(sum(m.length for m in chosen), best)

    def test_prefers_one_long_chunk_over_two_short_ones_and_the_reverse(self):
        """Both directions, because either greedy rule gets one of them wrong."""
        long_one = Match(0, 10, "sc:long")
        short_pair = [Match(0, 3, "sc:a"), Match(4, 7, "sc:b")]
        self.assertEqual(SubContextIndex.select([long_one] + short_pair), [long_one])

        wide = [Match(0, 9, "sc:a"), Match(10, 19, "sc:b")]
        overlapping = Match(5, 16, "sc:mid")
        self.assertEqual(SubContextIndex.select(wide + [overlapping]), wide)


class TestCutPoints(unittest.TestCase):
    """Boundaries decided by content, and the bounds that bend that rule."""

    def setUp(self):
        self.rng = random.Random(20260915)

    def toks(self, n):
        return [self.rng.randrange(1000, 150000) for _ in range(n)]

    def chunks(self, ids, *a, **kw):
        cuts = cut_points(ids, *a, **kw)
        bounds = [0, *cuts, len(ids)]
        return [ids[x:y] for x, y in zip(bounds, bounds[1:])]

    def test_the_pieces_reassemble_into_the_input(self):
        """`concat(segments) == prompt_ids` is what the prefill path assumes."""
        ids = self.toks(20000)
        for target in (128, 256, 512):
            flat = [tok for c in self.chunks(ids, target) for tok in c]
            self.assertEqual(flat, ids)

    def test_a_run_is_cut_the_same_way_wherever_it_sits(self):
        """The property the whole scheme rests on.

        Only the interior is claimed: the bounds make the first and last boundary of an
        occurrence depend on where the previous one fell, so those can differ. A
        content-defined cut in the middle may not.
        """
        doc = self.toks(3000)
        alone = {c for c in cut_points(doc, 256) if 600 < c < len(doc) - 600}
        self.assertTrue(alone, "test needs a doc long enough to have interior cuts")
        for _ in range(20):
            pre = self.toks(self.rng.randrange(0, 4000))
            full = pre + doc + self.toks(self.rng.randrange(0, 4000))
            moved = {
                c - len(pre)
                for c in cut_points(full, 256)
                if len(pre) < c < len(pre) + len(doc)
            }
            self.assertEqual(
                {c for c in moved if 600 < c < len(doc) - 600},
                alone,
                "the same tokens were cut differently at a different offset",
            )

    def test_a_quoted_fragment_still_yields_the_chunks_it_yielded_whole(self):
        """Quoting part of a document is the case roles cannot serve at all."""
        doc = self.toks(4000)
        whole = {chunk_id(c) for c in self.chunks(doc, 256) if len(c) >= 64}
        part = doc[900:3100]
        shared = whole & {chunk_id(c) for c in self.chunks(part, 256) if len(c) >= 64}
        self.assertGreater(len(shared), 1)

    def test_bounds_are_respected(self):
        ids = self.toks(50000)
        lens = [len(c) for c in self.chunks(ids, 256, 64, 1024)]
        self.assertGreaterEqual(min(lens), 64)
        self.assertLessEqual(max(lens), 1024)

    def test_average_chunk_length_tracks_the_target(self):
        ids = self.toks(60000)
        for target in (128, 256, 512):
            lens = [len(c) for c in self.chunks(ids, target, 8, 64 * target)]
            mean = sum(lens) / len(lens)
            self.assertGreater(mean, target * 0.5)
            self.assertLess(mean, target * 2.0)

    def test_nothing_to_cut_is_no_cuts_rather_than_a_useless_one(self):
        self.assertEqual(cut_points(self.toks(100), 256, 64), [])
        self.assertEqual(cut_points([], 256), [])
        self.assertEqual(cut_points(self.toks(5000), 1), [])

    def test_a_short_tail_stays_on_the_chunk_before_it(self):
        """A stub too short to register would be a namespace no scan may reuse."""
        for _ in range(10):
            ids = self.toks(self.rng.randrange(3000, 9000))
            self.assertGreaterEqual(len(self.chunks(ids, 256, 64, 1024)[-1]), 64)


if __name__ == "__main__":
    unittest.main()
