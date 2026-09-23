"""Per-forward-pass GPU timing for the sub-context A/B.

Set ``SGLANG_FORWARD_TRACE`` to an output path: every forward pass is bracketed by
CUDA events and appends one JSON row. Elapsed time is read once the end event has
completed (``Event.query()``), never by synchronising.
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
            # Tokens computed this pass, and tokens served from the cache. Both count
            # each request once however many chunks its prefill takes.
            new_tokens = batch.extend_num_tokens or 0
            cached_tokens = 0
            # Matched in the tree but not reused (drained from the request).
            discarded_tokens = 0
            # The part of `discarded` that was matched at another position.
            moved_tokens = 0
            # Displaced hits rotated into place; part of `cached_tokens`.
            rotated_tokens = 0
            # Declined blocks rotated back and filed at finish. Cache-level: it is
            # reported by whichever pass comes next, not by the request that did it.
            reinserted_tokens = 0
            cache = getattr(batch, "tree_cache", None)
            if getattr(cache, "sub_context_reinserted_tokens", 0):
                reinserted_tokens = cache.sub_context_reinserted_tokens
                cache.sub_context_reinserted_tokens = 0
            for req in reqs:
                # Charge only the increase of the client-facing `cached_tokens`.
                c = getattr(req, "cached_tokens", 0) or 0
                charged = getattr(req, "traced_cached_tokens", 0) or 0
                if c > charged:
                    cached_tokens += c - charged
                    req.traced_cached_tokens = c
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
            # Requests in this pass that took the split path.
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
