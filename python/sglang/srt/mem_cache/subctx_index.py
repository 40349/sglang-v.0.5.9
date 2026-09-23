"""Find every registered sub-context inside a request's token ids.

A chunk is addressed by a hash of its token ids and the request's ``extra_key``, not by
position, so the same tokens are found wherever they appear.

A scan fingerprints the ``ANCHOR_TOKENS``-long window at every position, looks each
fingerprint up among the chunks' first windows, and confirms a candidate by comparing
its whole token sequence.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
import xxhash

# Shortest run of tokens worth reusing.
MIN_CHUNK_TOKENS = int(os.environ.get("SGLANG_SUBCTX_MIN_CHUNK", "64"))

# Leading tokens of a chunk the index fingerprints.
ANCHOR_TOKENS = 32

# Window fingerprint: ``sum(token_j * BASE**(W-1-j))`` mod 2**64, then mixed. BASE is
# odd, hence invertible, so every window is read off one prefix sum.
_BASE = np.uint64(0x9E3779B97F4A7C15)
_BASE_INV = np.uint64(pow(int(_BASE), -1, 1 << 64))

# ``_POW[i] == _BASE**i`` and ``_POW_INV[i] == _BASE_INV**i``, grown on demand.
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


# Bloom filter over the registered anchors: two bit probes per position, vectorised.
# False positives cost one dict lookup that misses.
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


# ``None`` is a real scope (no cache_salt, no adapter), so absence needs its own value.
_MISSING = object()


class Match(NamedTuple):
    """``token_ids[start:end]`` is the chunk registered as ``chunk_id``."""

    start: int
    end: int
    chunk_id: str

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_id(token_ids: Sequence[int], extra_key: Optional[str] = None) -> str:
    """Namespace for the KV of ``token_ids`` under ``extra_key`` (cache_salt + lora_id).

    Position-independent. The ids are hashed as fixed-width int32.
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

    Entry ``i`` covers ``ids[i : i + ANCHOR_TOKENS]`` (empty if ``ids`` is shorter than
    one window). The same window fingerprints the same at any position.
    """
    n = ids.shape[0]
    if n < ANCHOR_TOKENS:
        return np.empty(0, dtype=np.uint64)
    windows = n - ANCHOR_TOKENS + 1
    _powers(n + 1)

    # Prefix sum of ``token_j * BASE_INV**j``; window s is multiplied back by
    # ``BASE**(s + W - 1)``. uint64 wraps mod 2**64.
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
    has its low ``log2(target_tokens)`` bits clear, so the same tokens are cut the same
    way wherever they appear. ``min_tokens`` suppresses a cut too close to the previous
    one and ``max_tokens`` (default ``4 * target_tokens``) forces one; both depend on
    where the previous cut fell.

    ``target_tokens`` is rounded down to a power of two. A tail shorter than
    ``min_tokens`` is left on the chunk before it.
    """
    min_tokens = max(min_tokens, 1)
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
    """The chunks registered under one ``extra_key``; a scan sees only its own scope."""

    __slots__ = ("chunks", "by_anchor", "_bits", "_mask")

    def __init__(self) -> None:
        self.chunks: Dict[str, np.ndarray] = {}
        self.by_anchor: Dict[int, List[str]] = {}
        # Bloom filter over `by_anchor`'s keys; None means rebuild on next use.
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
        """Add an anchor to the live filter, or drop the filter once it is too small."""
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
    """Registered chunks, and where they occur in a query.

    Holds token ids only. The KV, its residency and its position stay in the radix
    tree, looked up with the chunk id as ``extra_key``; the two may drift.
    """

    def __init__(self, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        # At least one anchor window.
        self.min_chunk_tokens = max(min_chunk_tokens, ANCHOR_TOKENS)
        self._scopes: Dict[Optional[str], _Scope] = {}
        # chunk id -> scope, for `unregister`.
        self._scope_of: Dict[str, Optional[str]] = {}
        self.registered = 0
        self.scanned_queries = 0
        self.scanned_tokens = 0
        # Dry-run tallies. `beyond_stitch_tokens`: found, still cached, and past the
        # stitched prefix.
        self.found_tokens = 0
        self.resident_tokens = 0
        self.beyond_stitch_tokens = 0
        self.displaced_tokens = 0
        self.stitched_tokens = 0
        # Live tallies.
        self.reused_tokens = 0
        self.rotated_tokens = 0
        # Requests whose layout did not fit one prefill pass and took the stitch.
        self.fell_back = 0

    def report(self) -> str:
        """One-line summary of the tallies."""
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
        # Dry run.
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
            # Same first window as another chunk.
            bucket.append(cid)
        self.registered += 1
        return cid

    def unregister(self, cid: str) -> None:
        """Forget a chunk (called when the tree evicts it). Unknown ids are ignored."""
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

    def scan(
        self, token_ids: Sequence[int], extra_key: Optional[str] = None
    ) -> List[Match]:
        """Every occurrence of a registered chunk in ``token_ids``, overlaps included."""
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

        # Bloom filter, then the anchor table, then a full token comparison.
        bits, mask = scope.prefilter()
        matches: List[Match] = []
        for start in _bloom_hits(bits, mask, prints).tolist():
            for cid in scope.by_anchor.get(int(prints[start]), ()):
                chunk = scope.chunks[cid]
                end = start + chunk.shape[0]
                if end <= n and np.array_equal(ids[start:end], chunk):
                    matches.append(Match(start, end, cid))
        return matches

    @staticmethod
    def select(matches: List[Match]) -> List[Match]:
        """Pick non-overlapping matches covering as many tokens as possible.

        Matches sorted by end; each is either taken, on top of the best cover ending at
        or before its start, or skipped. Ties keep the earlier match.
        """
        if len(matches) <= 1:
            return list(matches)

        ordered = sorted(matches, key=lambda m: (m.end, m.start))
        ends = [m.end for m in ordered]

        # best[i]: most tokens coverable with ordered[:i]; take[i]: ordered[i - 1] is
        # in it; prev[i]: where the solution continues.
        best = [0] * (len(ordered) + 1)
        take = [False] * (len(ordered) + 1)
        prev = [0] * (len(ordered) + 1)
        for i, m in enumerate(ordered, start=1):
            # Binary search: lo = number of matches ending at or before m.start.
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
