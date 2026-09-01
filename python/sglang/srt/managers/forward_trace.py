"""Per-forward-pass GPU timing, for A/B-ing sub-context KV reuse.

Set ``SGLANG_FORWARD_TRACE`` to an output path: every forward pass is bracketed by
CUDA events and appends one JSON row, so the two arms are comparable on the same
binary (``SGLANG_DISABLE_SUBCONTEXT=1`` for the baseline).

Elapsed time is read only once the end event has completed (``Event.query()``),
never by synchronising: under overlap scheduling the forward runs on a side stream,
so a blocking ``elapsed_time()`` would stall the pipeline being measured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.device_timer import DeviceTimer

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class ForwardTracer:
    """Appends one row per forward pass: mode, token counts and GPU milliseconds."""

    def __init__(self, path: str, tag: str = ""):
        self._path = path
        self._file = open(path, "a", buffering=1)  # line buffered, survives a kill -9
        self._timer = DeviceTimer(reporter=self._write)
        self._ct = 0
        self._t0 = time.perf_counter()
        self._file.write(
            json.dumps({"type": "run_start", "tag": tag, "wall": time.time()}) + "\n"
        )
        logger.info("ForwardTracer: writing forward-pass GPU timings to %s", path)

    @contextmanager
    def wrap(self, batch: "ScheduleBatch"):
        self._ct += 1
        with self._timer.wrap(metadata=self._describe(batch)):
            yield

    def _describe(self, batch: "ScheduleBatch") -> dict:
        mode = batch.forward_mode.name.lower()
        reqs = batch.reqs or []
        bs = len(reqs)

        if batch.forward_mode.is_extend():
            # Tokens pushed through the model this pass vs. served from the radix
            # cache -- the quantity sub-context reuse is meant to move.
            new_tokens = batch.extend_num_tokens or 0
            cached_tokens = sum(len(req.prefix_indices) for req in reqs)
            # Matching can find MORE than the contiguity rule stitches: after a
            # non-final segment misses, every later hit is dropped though matched and
            # locked, and prefix_indices keeps no trace of it. Taken from where
            # `_stitch_sub_contexts` recorded it and drained on read, so one stitch is
            # counted once however many chunks prefill splits into -- deriving it from
            # sum(sub_context_match_lens) here would go negative on continuations.
            discarded_tokens = 0
            # The part of the drop caused by a *position* mismatch rather than
            # contiguity, and still dropped: the share a rotation could have won back
            # but did not (rotation off, delta out of range, or the pool was full).
            moved_tokens = 0
            # Displaced hits that WERE won back, by copying the block to fresh slots
            # with its K rotated to the position it is reused at. These are part of
            # cached_tokens, so moved + rotated is the whole displaced population.
            rotated_tokens = 0
            # Tokens of a block the namespace had refused, rotated back to the position
            # it holds and filed there at finish -- which is what lets the reply be
            # cached at all. Cache-level, not per-request: the re-file happens after the
            # request's last forward pass, so this pass reports work another request
            # finished. The totals are right; a single row's attribution is not.
            reinserted_tokens = 0
            cache = getattr(batch, "tree_cache", None)
            if getattr(cache, "sub_context_reinserted_tokens", 0):
                reinserted_tokens = cache.sub_context_reinserted_tokens
                cache.sub_context_reinserted_tokens = 0
            for req in reqs:
                d = getattr(req, "sub_context_discarded", 0) or 0
                if d:
                    req.sub_context_discarded = 0
                    discarded_tokens += d
                m = getattr(req, "sub_context_moved", 0) or 0
                if m:
                    req.sub_context_moved = 0
                    moved_tokens += m
                r = getattr(req, "sub_context_rotated", 0) or 0
                if r:
                    req.sub_context_rotated = 0
                    rotated_tokens += r
            matched_tokens = cached_tokens + discarded_tokens
            # How many of these requests took the split path at all. A run with none
            # did not observe zero drops; without this, "0 vs 0" reads as evidence.
            sub_reqs = sum(1 for req in reqs if getattr(req, "has_sub_contexts", False))
        else:
            new_tokens = bs  # one token per sequence per decode step
            cached_tokens = 0
            matched_tokens = 0
            discarded_tokens = 0
            moved_tokens = 0
            rotated_tokens = 0
            reinserted_tokens = 0
            sub_reqs = 0

        return {
            "ct": self._ct,
            "mode": mode,
            "bs": bs,
            "new_tokens": new_tokens,
            "cached_tokens": cached_tokens,
            "matched_tokens": matched_tokens,
            "discarded_tokens": discarded_tokens,
            "moved_tokens": moved_tokens,
            "rotated_tokens": rotated_tokens,
            "reinserted_tokens": reinserted_tokens,
            "sub_reqs": sub_reqs,
            "t_rel": round(time.perf_counter() - self._t0, 6),
        }

    def _write(self, t: float, **metadata):
        metadata["gpu_ms"] = round(t * 1000.0, 4)
        self._file.write(json.dumps(metadata) + "\n")

    @staticmethod
    def maybe_create(tag: str = "") -> Optional["ForwardTracer"]:
        path = os.environ.get("SGLANG_FORWARD_TRACE", "")
        if not path or not torch.cuda.is_available():
            return None
        try:
            return ForwardTracer(path, tag=tag)
        except OSError as e:
            logger.warning("ForwardTracer: cannot open %s (%s), tracing off", path, e)
            return None
