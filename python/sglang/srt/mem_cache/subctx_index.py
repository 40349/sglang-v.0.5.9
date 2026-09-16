"""Find every registered sub-context inside a request's token ids.

A sub-context is addressed by a hash of its own token ids (and the request's
``extra_key``), never by what role it played in the prompt it came from: the same run of
tokens is the same chunk wherever it appears, so a block cached at position 120 in one
request is found at position 100 in the next. Where it lands is not part of its
identity -- moving it there is what ``rotate_kv.py`` does, using the
``canonical_position`` the radix tree records alongside the KV.

Scanning walks the query position by position. At every position we fingerprint the
next ``ANCHOR_TOKENS`` tokens, look the fingerprint up in the index, and report each
candidate for as far as its tokens and the query's then agree -- the whole chunk where
the whole chunk occurs, a head of it where only a head does.

Only a fixed-length head is fingerprinted, so a chunk costs one table entry rather than
one trie node per token, registration is a dict insert with no rebuild, and the
per-position fingerprints for a whole query are one vectorised pass. The exact
verification that follows means a fingerprint collision costs a wasted comparison,
never a wrong reuse.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
import xxhash

# Shortest run of tokens worth reusing.
MIN_CHUNK_TOKENS = int(os.environ.get("SGLANG_SUBCTX_MIN_CHUNK", "64"))

# How many leading tokens of a chunk the index fingerprints. Only ever a filter: a
# fingerprint hit is confirmed token by token before it is reported, so this trades
# bucket size against scan work and nothing else.
ANCHOR_TOKENS = 32

# The window fingerprint is a polynomial in this base over Z/2**64: token j of a
# window contributes ``token * BASE**(W-1-j)``, so the two positions of a window never
# share a weight. Odd, therefore invertible mod 2**64, which lets every window be read
# off one prefix sum below.
_BASE = np.uint64(0x9E3779B97F4A7C15)
_BASE_INV = np.uint64(pow(int(_BASE), -1, 1 << 64))

# ``_POW[i] == _BASE**i`` and ``_POW_INV[i] == _BASE_INV**i``, grown on demand and kept
# across calls: every scan needs them up to the query length, and they never change.
_POW = np.ones(1, dtype=np.uint64)
_POW_INV = np.ones(1, dtype=np.uint64)


def _powers(n: int) -> None:
    """Ensure ``_POW`` and ``_POW_INV`` hold at least ``n`` entries."""
    global _POW, _POW_INV
    if _POW.shape[0] >= n:
        return
    n = max(n, 2 * _POW.shape[0], 4096)
    with np.errstate(over="ignore"):
        for name, base in (("_POW", _BASE), ("_POW_INV", _BASE_INV)):
            out = np.empty(n, dtype=np.uint64)
            out[0] = 1
            np.cumprod(np.full(n - 1, base, dtype=np.uint64), out=out[1:])
            globals()[name] = out


# The scan asks "is this position's fingerprint one we know" once per token, so the
# answer has to be a couple of vectorised array ops rather than a search. A Bloom
# filter gives that: two bit probes per position, no false negatives, and the few false
# positives cost one dictionary lookup that misses. Bits per anchor buys accuracy --
# at 32 with two probes the false-positive rate is under half a percent, so a 6k-token
# query hands ~25 spurious positions to Python instead of 6144 real lookups.
_BLOOM_BITS_PER_ANCHOR = 32
_BLOOM_PROBES = (np.uint64(0), np.uint64(32))
_U6 = np.uint64(6)
_U63 = np.uint64(63)
_U1 = np.uint64(1)


def _bloom_hits(bits: np.ndarray, mask: np.uint64, prints: np.ndarray) -> np.ndarray:
    """Positions of ``prints`` whose fingerprint may be registered."""
    hit = None
    for shift in _BLOOM_PROBES:
        b = (prints >> shift) & mask
        probe = (bits[b >> _U6] >> (b & _U63)) & _U1
        hit = probe if hit is None else (hit & probe)
    return np.flatnonzero(hit)


def _common_prefix(ids: np.ndarray, chunk: np.ndarray) -> int:
    """Number of tokens ``ids`` and ``chunk`` share counting from their starts."""
    n = min(ids.shape[0], chunk.shape[0])
    if n == 0:
        return 0
    diff = np.flatnonzero(ids[:n] != chunk[:n])
    return int(diff[0]) if diff.shape[0] else n


# ``None`` is a real scope (no cache_salt, no adapter), so absence needs its own value.
_MISSING = object()


class Match(NamedTuple):
    """``token_ids[start:end]`` is ``chunk_id``'s chunk, or a leading run of it.

    ``end`` stops where the two sequences diverge, so it spans the whole registered
    chunk when the whole chunk occurs at ``start`` and less when only its head does.
    ``SubContextIndex.chunk_length`` tells the two apart.
    """

    start: int
    end: int
    chunk_id: str

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_id(token_ids: Sequence[int], extra_key: Optional[str] = None) -> str:
    """Address of the KV computed for ``token_ids`` under ``extra_key``.

    Only the token ids and ``extra_key`` go in, never a position: two requests holding
    this run of tokens at different offsets land on the same chunk.

    ``extra_key`` carries ``cache_salt`` and, concatenated onto it, ``lora_id`` (see
    ``Req.__init__``). Both change the KV a token sequence produces.

    The ids are hashed at a fixed width, so ``[1, 23]`` and ``[12, 3]`` cannot produce
    the same bytes.
    """
    h = xxhash.xxh3_64()
    h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
    if extra_key:
        h.update(b"\x00")
        h.update(extra_key.encode())
    return f"sc:{h.hexdigest()}"


def _mix(h: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, mixing the high bits of a weighted sum back down."""
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def anchor_fingerprints(ids: np.ndarray) -> np.ndarray:
    """Fingerprint of every ``ANCHOR_TOKENS``-long window of ``ids``.

    Entry ``i`` covers ``ids[i : i + ANCHOR_TOKENS]``, so the result has
    ``len(ids) - ANCHOR_TOKENS + 1`` entries (empty when the sequence is shorter than
    one window). Position-independent by construction: the same window of tokens
    fingerprints the same wherever it sits.
    """
    n = ids.shape[0]
    if n < ANCHOR_TOKENS:
        return np.empty(0, dtype=np.uint64)
    windows = n - ANCHOR_TOKENS + 1
    _powers(n + 1)

    # Every window is a slice of one prefix sum. Scaling token j by ``_BASE_INV**j``
    # before accumulating makes the sum over a window differ from the polynomial it
    # should be by the single factor ``_BASE**(s + W - 1)``, put back below. uint64
    # arithmetic wraps, which is the modular arithmetic this wants.
    with np.errstate(over="ignore"):
        scaled = ids.astype(np.uint64) * _POW_INV[:n]
        running = np.empty(n + 1, dtype=np.uint64)
        running[0] = 0
        np.cumsum(scaled, out=running[1:])
        span = running[ANCHOR_TOKENS:] - running[:windows]
        return _mix(span * _POW[ANCHOR_TOKENS - 1 : ANCHOR_TOKENS - 1 + windows])



def cut_points(
    ids: Sequence[int],
    target_tokens: int,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = 0,
) -> List[int]:
    """Offsets at which to cut ``ids`` into chunks, decided by content.

    A cut is made before position ``i`` when the fingerprint of ``ids[i:i+ANCHOR_TOKENS]``
    has its low ``log2(target_tokens)`` bits clear, so a boundary depends on the tokens
    *at* it and nothing else: the same run of tokens is cut in the same places wherever
    it appears and whatever precedes it, and a document quoted at a new offset, or
    quoted in part, still yields the chunks it yielded before.

    ``min_tokens`` suppresses a cut too close to the last one and ``max_tokens`` forces
    one the content did not produce. Both make a boundary depend on where the previous
    boundary fell, so an occurrence at a new offset can lose the chunks either side of a
    forced cut while keeping the ones between.

    ``target_tokens`` is rounded down to a power of two. A tail shorter than
    ``min_tokens`` is left on the chunk before it.
    """
    n = len(ids)
    if target_tokens < 2 or n < 2 * min_tokens:
        return []
    if max_tokens <= 0:
        max_tokens = 4 * target_tokens
    max_tokens = max(max_tokens, 2 * min_tokens)

    arr = np.asarray(ids, dtype=np.int32)
    prints = anchor_fingerprints(arr)
    if prints.shape[0] == 0:
        return []

    mask = np.uint64((1 << (int(target_tokens).bit_length() - 1)) - 1)
    cand = np.flatnonzero((prints & mask) == np.uint64(0))

    cuts: List[int] = []
    last = 0
    ci = 0
    n_cand = cand.shape[0]
    while True:
        while ci < n_cand and cand[ci] < last + min_tokens:
            ci += 1
        nxt = int(cand[ci]) if ci < n_cand else n
        if nxt - last > max_tokens:
            nxt = last + max_tokens
        if n - nxt < min_tokens:
            break
        cuts.append(nxt)
        last = nxt
    return cuts


class _Scope:
    """The chunks registered under one ``extra_key``.

    Scopes never share: a ``cache_salt`` or a LoRA adapter separates two tenants' KV
    and what a scan can find.
    """

    __slots__ = ("chunks", "by_anchor", "_bits", "_mask")

    def __init__(self) -> None:
        self.chunks: Dict[str, np.ndarray] = {}
        self.by_anchor: Dict[int, List[str]] = {}
        # The prefilter over `by_anchor`'s keys. None means "build it on next use".
        self._bits: Optional[np.ndarray] = None
        self._mask = np.uint64(0)

    def prefilter(self):
        if self._bits is None:
            count = len(self.by_anchor)
            n_bits = 1 << max(9, (max(count, 1) * _BLOOM_BITS_PER_ANCHOR).bit_length())
            self._mask = np.uint64(n_bits - 1)
            self._bits = np.zeros(n_bits >> 6, dtype=np.uint64)
            if count:
                keys = np.fromiter(self.by_anchor, dtype=np.uint64, count=count)
                for shift in _BLOOM_PROBES:
                    b = (keys >> shift) & self._mask
                    np.bitwise_or.at(self._bits, b >> _U6, _U1 << (b & _U63))
        return self._bits, self._mask

    def add_anchor(self, anchor: int) -> None:
        """Record a newly occupied anchor.

        Set straight into the live filter rather than rebuilding it. Outgrowing the
        current size drops the filter; the next scan builds a bigger one.
        """
        if self._bits is None:
            return
        if len(self.by_anchor) * _BLOOM_BITS_PER_ANCHOR > (self._bits.shape[0] << 6):
            self._bits = None
            return
        h = np.uint64(anchor)
        for shift in _BLOOM_PROBES:
            b = (h >> shift) & self._mask
            self._bits[b >> _U6] |= _U1 << (b & _U63)

    def drop_anchor(self) -> None:
        """A Bloom filter cannot forget one entry, so an eviction rebuilds it."""
        self._bits = None


class SubContextIndex:
    """Maps a run of tokens to the sub-context it is, so a scan can find it anywhere.

    This answers only *what is in this query and where*. What KV a chunk owns, whether
    it is still resident, and at which position it was computed all stay with the radix
    tree, which is looked up afterwards with the chunk id as its ``extra_key``. A chunk
    evicted from the tree but still registered here returns nothing from that lookup
    and the caller recomputes; the two are allowed to drift.
    """

    def __init__(self, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        # A chunk shorter than one anchor window could never be fingerprinted.
        self.min_chunk_tokens = max(min_chunk_tokens, ANCHOR_TOKENS)
        self._scopes: Dict[Optional[str], _Scope] = {}
        # Which scope each chunk lives in. The tree only ever hands back a chunk id --
        # its namespace -- so forgetting an evicted chunk needs the way back.
        self._scope_of: Dict[str, Optional[str]] = {}
        self.registered = 0
        self.scanned_queries = 0
        self.scanned_tokens = 0
        # Dry-run tallies. `beyond_stitch_tokens` is the one that decides whether the
        # rest of the work pays: tokens a scan located, that are still cached, and that
        # sit past where a prefix-only reuse had to stop.
        self.found_tokens = 0
        self.resident_tokens = 0
        self.beyond_stitch_tokens = 0
        self.displaced_tokens = 0
        self.stitched_tokens = 0
        # Live tallies, filled when the scan is actually driving the prefill.
        self.reused_tokens = 0
        self.rotated_tokens = 0
        # Requests that scanned, found reuse, and then had to give it back because what
        # was left to compute did not fit one prefill pass. Reported because it is the
        # difference between "the index found nothing" and "the index found plenty and
        # the chunker could not take it", which look identical in a reuse figure.
        self.fell_back = 0

    def report(self) -> str:
        """One line on what the index is doing, in whichever mode it is running."""
        head = (
            f"sub-context index: {len(self)} chunks, {self.scanned_queries} queries, "
            f"{self.scanned_tokens} prompt tokens"
        )
        if self.reused_tokens or self.fell_back:
            share = 100.0 * self.reused_tokens / max(self.scanned_tokens, 1)
            return (
                f"{head} | reused {self.reused_tokens} ({share:.0f}% of prompt), "
                f"{self.rotated_tokens} of them rotated into place"
                f" | {self.fell_back} requests too big for one pass"
            )
        # Dry run: what the scan *would* have won over the prefix-only path.
        return (
            f"{head} | stitched {self.stitched_tokens} | scan found "
            f"{self.found_tokens} ({self.resident_tokens} still cached, "
            f"{self.displaced_tokens} of those at another position) | "
            f"reachable only by placing blocks freely: {self.beyond_stitch_tokens}"
        )

    def __len__(self) -> int:
        return sum(len(s.chunks) for s in self._scopes.values())

    def register(
        self, token_ids: Sequence[int], extra_key: Optional[str] = None
    ) -> Optional[str]:
        """Make ``token_ids`` findable. Returns its chunk id, or None if too short.

        Idempotent: registering the same run twice does not duplicate it.
        """
        ids = np.asarray(token_ids, dtype=np.int32)
        if ids.shape[0] < self.min_chunk_tokens:
            return None

        cid = chunk_id(ids, extra_key)
        scope = self._scopes.get(extra_key)
        if scope is None:
            scope = self._scopes[extra_key] = _Scope()
        if cid in scope.chunks:
            return cid

        scope.chunks[cid] = ids
        self._scope_of[cid] = extra_key
        anchor = int(anchor_fingerprints(ids[:ANCHOR_TOKENS])[0])
        bucket = scope.by_anchor.get(anchor)
        if bucket is None:
            scope.by_anchor[anchor] = [cid]
            scope.add_anchor(anchor)
        else:
            # Two chunks sharing their first ANCHOR_TOKENS tokens; both are verified
            # in full.
            bucket.append(cid)
        self.registered += 1
        return cid

    def unregister(self, cid: str) -> None:
        """Forget a chunk. Called when the tree evicts it.

        Unknown ids are ignored. A chunk left registered after its KV is gone costs one
        lookup that misses; one evicted before it is registered is found next time.
        """
        extra_key = self._scope_of.pop(cid, _MISSING)
        if extra_key is _MISSING:
            return
        scope = self._scopes.get(extra_key)
        ids = None if scope is None else scope.chunks.pop(cid, None)
        if ids is None:
            return
        anchor = int(anchor_fingerprints(ids[:ANCHOR_TOKENS])[0])
        bucket = scope.by_anchor.get(anchor)
        if bucket is None:
            return
        if len(bucket) == 1 and bucket[0] == cid:
            del scope.by_anchor[anchor]
            scope.drop_anchor()
        elif cid in bucket:
            bucket.remove(cid)

    def chunk_length(self, cid: str) -> int:
        """Tokens registered under ``cid``, or 0 if nothing is."""
        scope = self._scopes.get(self._scope_of.get(cid, _MISSING))
        chunk = None if scope is None else scope.chunks.get(cid)
        return 0 if chunk is None else int(chunk.shape[0])

    def scan(
        self, token_ids: Sequence[int], extra_key: Optional[str] = None
    ) -> List[Match]:
        """Every registered chunk that occurs in ``token_ids``, at every position.

        A chunk is reported wherever its first ``ANCHOR_TOKENS`` tokens occur, running
        as far as its tokens and the query's agree; runs shorter than
        ``min_chunk_tokens`` are dropped. A head reaching only part way into the chunk
        is still reported, and reuses that many of the chunk's slots.

        Exhaustive and unordered by preference: overlapping and nested occurrences are
        all reported. ``select`` decides which of them to reuse.
        """
        scope = self._scopes.get(extra_key)
        if scope is None or not scope.chunks:
            return []

        ids = np.asarray(token_ids, dtype=np.int32)
        n = ids.shape[0]
        self.scanned_queries += 1
        self.scanned_tokens += n

        prints = anchor_fingerprints(ids)
        if prints.shape[0] == 0:
            return []

        # The prefilter answers "could anything start here" for every position at
        # once; survivors are checked against the real table, then token by token.
        bits, mask = scope.prefilter()
        matches: List[Match] = []
        for start in _bloom_hits(bits, mask, prints).tolist():
            for cid in scope.by_anchor.get(int(prints[start]), ()):
                take = _common_prefix(ids[start:], scope.chunks[cid])
                if take >= self.min_chunk_tokens:
                    matches.append(Match(start, start + take, cid))
        return matches

    @staticmethod
    def select(matches: List[Match]) -> List[Match]:
        """Pick a non-overlapping subset covering as many tokens as possible.

        Occurrences overlap and nest, but a position has one KV slot, so at most one
        chunk can own it. Covered tokens is exactly the prefill skipped.

        Sorted by end, each match is either taken -- with the best result that ends at
        or before its start -- or skipped for the best result so far; one pass decides
        it, and the choices are read back off the same array.
        """
        if len(matches) <= 1:
            return list(matches)

        ordered = sorted(matches, key=lambda m: (m.end, m.start))
        ends = [m.end for m in ordered]

        # best[i] is the most tokens coverable using only ordered[:i]; take[i] says
        # whether ordered[i - 1] is in that solution, and prev[i] where it resumes.
        best = [0] * (len(ordered) + 1)
        take = [False] * (len(ordered) + 1)
        prev = [0] * (len(ordered) + 1)
        for i, m in enumerate(ordered, start=1):
            # The last match ending at or before this one starts; bisect over `ends`,
            # which is sorted, so the compatible prefix is found without a scan.
            lo, hi = 0, i - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if ends[mid] <= m.start:
                    lo = mid + 1
                else:
                    hi = mid
            with_m = m.length + best[lo]
            if with_m > best[i - 1]:
                best[i], take[i], prev[i] = with_m, True, lo
            else:
                best[i] = best[i - 1]

        chosen: List[Match] = []
        i = len(ordered)
        while i > 0:
            if take[i]:
                chosen.append(ordered[i - 1])
                i = prev[i]
            else:
                i -= 1
        chosen.reverse()
        return chosen
