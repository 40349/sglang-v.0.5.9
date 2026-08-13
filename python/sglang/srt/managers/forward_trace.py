"""Per-forward-pass GPU timing, for A/B-ing sub-context KV reuse.

Enabled by setting ``SGLANG_FORWARD_TRACE`` to an output path. Every forward pass
is bracketed by a pair of CUDA events and one JSON row is appended per pass, so a
run can be compared against a run with the sub-context split turned off
(``SGLANG_DISABLE_SUBCONTEXT=1``) without rebuilding or swapping checkouts.

The elapsed time is only read once the end event has actually completed
(``Event.query()``), never by synchronising on it. That matters here: the
scheduler runs the forward on a side stream under overlap scheduling, so a
blocking ``elapsed_time()`` would stall the very pipeline being measured and
inflate the numbers it reports.
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
            # Tokens actually pushed through the model this pass, vs. tokens served
            # from the radix cache (the quantity sub-context reuse is meant to move).
            new_tokens = batch.extend_num_tokens or 0
            cached_tokens = sum(len(req.prefix_indices) for req in reqs)
            # Per-namespace matching can find MORE than the contiguity rule lets it
            # stitch: once a non-final segment misses or only partially hits, every
            # later segment's hit is dropped even though it was matched and locked.
            # prefix_indices holds only what was stitched, so that waste is invisible
            # unless the matched totals are recorded next to it.
            matched_tokens = 0
            for req in reqs:
                lens = getattr(req, "sub_context_match_lens", None)
                matched_tokens += sum(lens) if lens else len(req.prefix_indices)
        else:
            new_tokens = bs  # one token per sequence per decode step
            cached_tokens = 0
            matched_tokens = 0

        return {
            "ct": self._ct,
            "mode": mode,
            "bs": bs,
            "new_tokens": new_tokens,
            "cached_tokens": cached_tokens,
            "matched_tokens": matched_tokens,
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
