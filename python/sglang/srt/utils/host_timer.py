"""Host-side (CPU) stage timing, for costing the sub-context mechanism itself.

What the split *adds* is not on the GPU: re-rendering the template to find block
boundaries, one ``match_prefix`` and one insert per namespace instead of per
request. None of it shows up in a forward-pass trace.

Set ``SGLANG_STAGE_TRACE`` to an output path. Each process writes ``<path>.<proc>``
-- the template split happens in the HTTP/tokenizer process, matching and insertion
in the scheduler. Stages are named identically in both arms and placed at the branch
point, so subtracting per stage across an A/B gives the added cost directly.

Counters accumulate from process start, which includes the replay client's warm-up.
Once that client creates ``<SGLANG_STAGE_TRACE>.mark`` a second accumulator opens and
is reported as ``measured``, matching the forward trace's ``measure_start`` window.
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
# snapshot must stay current. Bounded by time and call count, so a short run lands.
_DUMP_INTERVAL_S = 1.0
_DUMP_EVERY_N = 20


class HostTimer:
    def __init__(self, path: str, proc: str):
        self._path = f"{path}.{proc}"
        self._proc = proc
        # stage -> [count, total_ns, max_ns]
        self._stages: Dict[str, List[int]] = {}
        # Same shape, restarted when the client marks the end of warm-up. See
        # _check_mark.
        self._measured: Optional[Dict[str, List[int]]] = None
        self._mark_path: Optional[str] = f"{path}.mark"
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

    def _check_mark(self) -> None:
        """Open a second accumulator once the replay client marks warm-up over.

        The forward trace gets its ``measure_start`` for free -- client and
        server append to the same file. The stages cannot: they live in two
        server processes while the marker is created by a third, so there is no
        in-process signal to hook. A stat() on an agreed path is the cheapest
        thing that crosses that boundary. It runs outside the timed region
        (``add`` is called after the elapsed time has been taken) and stops
        entirely once the marker has been seen.

        Counting into a fresh dict rather than subtracting a baseline keeps
        max_us honest: a maximum cannot be un-summed.
        """
        if not os.path.exists(self._mark_path):
            return
        with self._lock:
            if self._measured is None:
                self._measured = {}
        self._mark_path = None  # seen; never stat again

    def add(self, stage: str, ns: int) -> None:
        if self._mark_path is not None:
            self._check_mark()
        with self._lock:
            for table in (self._stages, self._measured):
                if table is None:
                    continue
                slot = table.get(stage)
                if slot is None:
                    table[stage] = [1, ns, ns]
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

    @staticmethod
    def _snapshot(table: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        return {
            stage: {
                "count": c,
                "total_ms": round(total / 1e6, 4),
                "mean_us": round(total / c / 1e3, 3),
                "max_us": round(mx / 1e3, 3),
            }
            for stage, (c, total, mx) in sorted(table.items())
        }

    def dump(self) -> None:
        with self._lock:
            if not self._stages:
                return
            doc = {"proc": self._proc, "stages": self._snapshot(self._stages)}
            # Present only once the warm-up marker has been seen. Readers should
            # prefer it and say so when it is missing, rather than silently
            # reporting a window that includes warm-up.
            if self._measured is not None:
                doc["measured"] = self._snapshot(self._measured)
            self._last_dump = time.perf_counter()
            self._since_dump = 0
        try:
            with open(self._path, "w") as f:
                json.dump(doc, f, indent=2)
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
