#!/usr/bin/env python3
"""Inject the same timing probes into another sglang installation.

The baseline arm is a pristine upstream sglang in its own conda env; its stage
timings are only subtractable from this fork's with byte-identical probes, so this
copies the timing modules over and inserts the hook lines rather than hand-editing.

Usage, from inside the baseline env:

    conda activate sglang_orig
    python /home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9/instrument_sglang.py
    python .../instrument_sglang.py --check     # report status, change nothing
    python .../instrument_sglang.py --revert    # undo (restores .orig backups)

Every edit is idempotent and backed up to <file>.orig on first touch. Probes go
above a `def`, never inside a body: upstream's bodies differ, only signatures line up.

`subctx_split` reports as absent on upstream -- there is no _compute_sub_context_ids
there, which is the point: its baseline cost is zero.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "python" / "sglang" / "srt"

# Modules copied in wholesale.
NEW_MODULES = [
    ("utils/host_timer.py", "utils/host_timer.py"),
    ("managers/forward_trace.py", "managers/forward_trace.py"),
]

IMPORT_LINE = "from sglang.srt.utils import host_timer"

# (file, regex matching the def line, decorator to insert above it, required)
DECORATORS = [
    ("managers/schedule_batch.py", r"^    def init_next_round_input\(",
     '    @host_timer.timed("match")', True),
    ("mem_cache/radix_cache.py", r"^    def cache_finished_req\(",
     '    @host_timer.timed("cache_finished")', True),
    ("mem_cache/radix_cache.py", r"^    def cache_unfinished_req\(",
     '    @host_timer.timed("cache_unfinished")', True),
    ("entrypoints/openai/serving_chat.py", r"^    def _apply_jinja_template\(",
     '    @host_timer.timed("tpl_render")', True),
    # Absent upstream by design; present in the fork.
    ("entrypoints/openai/serving_chat.py", r"^    def _compute_sub_context_ids\(",
     '    @host_timer.timed("subctx_split")', False),
]

# (file, anchor regex, line inserted AFTER the anchor's line, indent-matched)
INIT_CALLS = [
    ("entrypoints/openai/serving_chat.py", r"^        super\(\)\.__init__\(tokenizer_manager\)",
     '        host_timer.init_host_timer("http")'),
    ("managers/scheduler_metrics_mixin.py", r"^        self\.stats = SchedulerStats\(\)",
     '        host_timer.init_host_timer("scheduler")\n'
     '        self.forward_tracer = ForwardTracer.maybe_create(\n'
     '            tag=self.server_args.served_model_name\n'
     '        ) if self.attn_tp_rank == 0 else None'),
]

TRACER_IMPORT = "from sglang.srt.managers.forward_trace import ForwardTracer"

# The forward-pass CUDA-event hook. Exact-match replacement, so a mismatch is
# loud rather than silently producing a baseline with no GPU trace.
FORWARD_HOOK_OLD = """    @contextmanager
    def record_forward_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return
"""
FORWARD_HOOK_NEW = """    @contextmanager
    def record_forward_metrics(self: Scheduler, batch: ScheduleBatch):
        # Both are @contextmanager, so they compose with `with`; `yield from` would
        # try to iterate a _GeneratorContextManager and blow up on the first batch.
        with self._record_forward_trace(batch), self._record_forward_metrics_inner(batch):
            yield

    @contextmanager
    def _record_forward_trace(self: Scheduler, batch: ScheduleBatch):
        if getattr(self, "forward_tracer", None) is None:
            yield
            return
        with self.forward_tracer.wrap(batch):
            yield

    @contextmanager
    def _record_forward_metrics_inner(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return
"""


def find_target() -> Path:
    import sglang  # noqa: F401  (resolved through the active env)

    root = Path(sglang.__file__).resolve().parent / "srt"
    if not root.is_dir():
        sys.exit(f"no srt/ under {root.parent}")
    return root


MARKER = "host_timer"


def write_unlinked(path: Path, text: str) -> None:
    """Write a file without writing *through* a hardlink.

    uv (and pip) install by hardlinking from a shared global cache, so a plain
    in-place write edits the cached copy too -- silently instrumenting every
    future install made from that cache, in every env, with no way to reinstall
    out of it. Replace the directory entry instead of truncating the shared
    inode, so the cache keeps the pristine content.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    shutil.copystat(path, tmp)
    tmp.replace(path)


def backup(path: Path) -> None:
    """Snapshot a file before first edit -- but never snapshot an edited one.

    Backing up an already-instrumented file silently poisons the backup, and then
    --revert 'restores' the very thing it was meant to undo. Refuse instead, and
    say how to get a clean tree back.
    """
    orig = path.with_suffix(path.suffix + ".orig")
    if orig.exists():
        if MARKER in orig.read_text():
            sys.exit(
                f"POISONED BACKUP: {orig.name} is itself instrumented, so --revert\n"
                f"cannot recover the original. Restore the package first:\n"
                f"    uv pip install --force-reinstall --no-deps sglang==0.5.9\n"
                f"then delete the stale backups and re-run this script:\n"
                f"    find {path.parent.parent} -name '*.orig' -delete"
            )
        return
    if MARKER in path.read_text():
        sys.exit(
            f"ALREADY INSTRUMENTED with no backup: {path}\n"
            f"Reinstall the package to get a clean copy, then re-run:\n"
            f"    uv pip install --force-reinstall --no-deps sglang==0.5.9"
        )
    shutil.copy2(path, orig)


def ensure_import(text: str, line: str) -> str:
    if line in text:
        return text
    lines = text.split("\n")
    last = max(
        (i for i, l in enumerate(lines[:200])
         if l.startswith("from sglang.") or l.startswith("import ")),
        default=None,
    )
    if last is None:
        sys.exit(f"cannot find an import block to extend for: {line}")
    lines.insert(last + 1, line)
    return "\n".join(lines)


def insert_above(text: str, pattern: str, inserted: str) -> tuple[str, bool]:
    rx = re.compile(pattern, re.M)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if rx.match(line):
            if i and lines[i - 1].strip() == inserted.strip():
                return text, False  # already applied
            lines.insert(i, inserted)
            return "\n".join(lines), True
    return text, False


def insert_below(text: str, pattern: str, inserted: str) -> tuple[str, bool]:
    if inserted.split("\n")[0] in text:
        return text, False
    rx = re.compile(pattern, re.M)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if rx.match(line):
            lines.insert(i + 1, inserted)
            return "\n".join(lines), True
    return text, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--revert", action="store_true", help="restore .orig backups")
    ap.add_argument("--target", default=None, help="path to sglang/srt (default: active env)")
    args = ap.parse_args()

    dst = Path(args.target) if args.target else find_target()
    print(f"target: {dst}")
    if dst == SRC_ROOT:
        sys.exit("refusing to instrument this fork -- it already has the probes")

    if args.revert:
        n = 0
        for orig in dst.rglob("*.orig"):
            live = orig.with_suffix("")
            shutil.move(str(orig), str(live))
            print(f"  restored {live.relative_to(dst)}")
            n += 1
        for _, rel in NEW_MODULES:
            p = dst / rel
            if p.exists():
                p.unlink()
                print(f"  removed {rel}")
        print(f"reverted {n} file(s)")
        return 0

    if args.check:
        for _, rel in NEW_MODULES:
            print(f"  {'OK  ' if (dst / rel).exists() else 'MISS'} {rel}")
        for rel, pattern, dec, _req in DECORATORS:
            text = (dst / rel).read_text()
            print(f"  {'OK  ' if dec.strip() in text else 'MISS'} {dec.strip()}  ({rel})")
        mixin = (dst / "managers/scheduler_metrics_mixin.py").read_text()
        print(f"  {'OK  ' if '_record_forward_trace' in mixin else 'MISS'} forward CUDA-event hook")
        return 0

    for src_rel, dst_rel in NEW_MODULES:
        src = SRC_ROOT / src_rel
        tgt = dst / dst_rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
        print(f"  copied {dst_rel}")

    touched: dict[Path, str] = {}

    def load(rel: str) -> str:
        p = dst / rel
        if p not in touched:
            if not p.exists():
                sys.exit(f"missing {p} -- is this really an sglang install?")
            backup(p)
            touched[p] = p.read_text()
        return touched[p]

    for rel, pattern, dec, required in DECORATORS:
        text = load(rel)
        text, did = insert_above(text, pattern, dec)
        touched[dst / rel] = text
        if did:
            print(f"  hooked {dec.strip()}  ({rel})")
        elif dec.strip() in text:
            print(f"  already {dec.strip()}  ({rel})")
        elif required:
            sys.exit(f"FAILED: no line matching {pattern!r} in {rel}.\n"
                     f"Upstream signature differs; adjust DECORATORS in this script.")
        else:
            print(f"  absent (expected upstream): {dec.strip()}  ({rel})")

    for rel, anchor, call in INIT_CALLS:
        text = load(rel)
        text, did = insert_below(text, anchor, call)
        touched[dst / rel] = text
        print(f"  {'armed ' if did else 'already'} timer init in {rel}")

    # imports
    for rel in {r for r, _, _, _ in DECORATORS} | {r for r, _, _ in INIT_CALLS}:
        touched[dst / rel] = ensure_import(load(rel), IMPORT_LINE)
    mixin_rel = "managers/scheduler_metrics_mixin.py"
    touched[dst / mixin_rel] = ensure_import(load(mixin_rel), TRACER_IMPORT)

    # forward-pass CUDA events
    text = load(mixin_rel)
    if "_record_forward_trace" in text:
        print("  already forward CUDA-event hook")
    elif FORWARD_HOOK_OLD in text:
        touched[dst / mixin_rel] = text.replace(FORWARD_HOOK_OLD, FORWARD_HOOK_NEW, 1)
        print("  hooked forward CUDA-event timing")
    else:
        sys.exit(f"FAILED: record_forward_metrics in {mixin_rel} does not match the\n"
                 f"expected upstream text. Patch it by hand (see FORWARD_HOOK_NEW).")

    for path, text in touched.items():
        write_unlinked(path, text)
    print(f"\ninstrumented {len(touched)} file(s); backups at *.orig")
    print("verify with: python -c 'import sglang.srt.managers.schedule_batch'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
