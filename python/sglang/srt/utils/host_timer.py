"""Host-side (CPU) stage timing for the sub-context mechanism.

Set ``SGLANG_STAGE_TRACE`` to an output prefix. Each process writes one JSON object to
``<prefix>.<proc>.json`` (``http`` for the template split, ``scheduler`` for match and
insert). Stage names are the same in both A/B arms, so per-stage differences are the
added cost.

Counters accumulate from process start. Once the replay client creates
``<prefix>.mark``, a second accumulator starts and is reported as ``measured``.
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

# Rewrite the snapshot at least this often (a SIGKILL skips atexit).
_DUMP_INTERVAL_S = 1.0
_DUMP_EVERY_N = 20


class HostTimer:
    def __init__(self, path: str, proc: str):
        self._path = f"{path}.{proc}.json"
        self._proc = proc
        # stage -> [count, total_ns, max_ns]
        self._stages: Dict[str, List[int]] = {}
        # Same shape, counted from the client's warm-up mark on.
        self._measured: Optional[Dict[str, List[int]]] = None
        self._mark_path: Optional[str] = f"{path}.mark"
        # Reentrant: the signal handler calls dump() on the thread that may hold it.
        self._lock = threading.RLock()
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
        """Start the ``measured`` accumulator once the marker file exists.

        Stats the marker on each ``add`` until it is seen, then never again.
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
            # Absent until the warm-up marker is seen.
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


def armed() -> bool:
    """Whether this process is recording."""
    return _timer is not None


def add(stage: str, ns: int) -> None:
    """Record a duration measured elsewhere (e.g. a CUDA event pair)."""
    if _timer is not None:
        _timer.add(stage, ns)


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
    """Decorator form of ``record``. ``instrument_sglang.py`` inserts it upstream."""

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
