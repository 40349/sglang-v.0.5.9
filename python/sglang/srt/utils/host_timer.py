"""Host-side (CPU) stage timing, for costing the sub-context mechanism itself.

CUDA events measure what the GPU does. The work sub-context reuse *adds* is not
on the GPU at all: re-rendering the chat template to find block boundaries, one
``match_prefix`` per namespace instead of one per request, one insert per
namespace instead of one. All of that is host time, and none of it shows up in a
forward-pass trace.

Enabled by setting ``SGLANG_STAGE_TRACE`` to an output path. Each process writes
to ``<path>.<proc>`` because the stages live in different processes: the template
split happens in the HTTP/tokenizer process, matching and insertion in the
scheduler.

Stages are named identically in both configs and placed at the branch point, so
the same name covers the stock path and the sub-context path. Running once with
``SGLANG_DISABLE_SUBCONTEXT=1`` and once without, then subtracting per stage,
gives the added cost directly.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# A benchmark server usually dies by SIGKILL, which atexit never sees, so the
# snapshot on disk has to stay close to current. Bounded by both time and call
# count: a short run that ends before the interval elapses still lands.
_DUMP_INTERVAL_S = 1.0
_DUMP_EVERY_N = 20


class HostTimer:
    def __init__(self, path: str, proc: str):
        self._path = f"{path}.{proc}"
        self._proc = proc
        # stage -> [count, total_ns, max_ns]
        self._stages: Dict[str, List[int]] = {}
        self._lock = threading.Lock()
        self._last_dump = time.perf_counter()
        self._since_dump = 0
        atexit.register(self.dump)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, self._make_handler(sig, prev))
            except (ValueError, OSError):
                pass  # not the main thread, or signal unavailable

    def _make_handler(self, sig, prev):
        def handler(signum, frame):
            self.dump()
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)

        return handler
        logger.info("HostTimer: writing stage timings to %s", self._path)

    def add(self, stage: str, ns: int) -> None:
        with self._lock:
            slot = self._stages.get(stage)
            if slot is None:
                self._stages[stage] = [1, ns, ns]
            else:
                slot[0] += 1
                slot[1] += ns
                if ns > slot[2]:
                    slot[2] = ns
            self._since_dump += 1
            due = (
                self._since_dump >= _DUMP_EVERY_N
                or time.perf_counter() - self._last_dump > _DUMP_INTERVAL_S
            )
        if due:
            self.dump()

    def dump(self) -> None:
        with self._lock:
            if not self._stages:
                return
            snapshot = {
                stage: {
                    "count": c,
                    "total_ms": round(total / 1e6, 4),
                    "mean_us": round(total / c / 1e3, 3),
                    "max_us": round(mx / 1e3, 3),
                }
                for stage, (c, total, mx) in sorted(self._stages.items())
            }
            self._last_dump = time.perf_counter()
            self._since_dump = 0
        try:
            with open(self._path, "w") as f:
                json.dump({"proc": self._proc, "stages": snapshot}, f, indent=2)
        except OSError as e:
            logger.warning("HostTimer: cannot write %s (%s)", self._path, e)


_timer: Optional[HostTimer] = None
_init_done = False


def init_host_timer(proc: str) -> None:
    """Arm the timer for this process. No-op unless SGLANG_STAGE_TRACE is set."""
    global _timer, _init_done
    if _init_done:
        return
    _init_done = True
    path = os.environ.get("SGLANG_STAGE_TRACE", "")
    if path:
        _timer = HostTimer(path, proc)


@contextmanager
def record(stage: str):
    if _timer is None:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        _timer.add(stage, time.perf_counter_ns() - t0)


def timed(stage: str):
    """Decorator form. Preferred at the hook sites: it is a single inserted line
    above a ``def``, so the identical probe can be applied to an upstream sglang
    whose function bodies differ from this fork's."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _timer is None:
                return fn(*args, **kwargs)
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                _timer.add(stage, time.perf_counter_ns() - t0)

        return wrapper

    return deco
