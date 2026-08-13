#!/bin/bash
# A/B the sub-context mechanism on real SWE-bench traffic. Two comparisons,
# answering two different questions -- pick by what you need to claim.
#
#   toggle   Same env, same binary, same weights; the arms differ only by
#            SGLANG_DISABLE_SUBCONTEXT. Costs the MECHANISM. Everything else
#            about the fork sits on both sides and cancels. Use this one for
#            "how much does sub-context add".
#
#   replay   $BASE_ENV (stock sglang==0.5.9 from PyPI, instrumented by
#            instrument_sglang.py, no PYTHONPATH) vs $FORK_ENV + PYTHONPATH into
#            this fork. Costs the FORK AS A WHOLE against upstream, and pays for
#            it with two builds' worth of drift: on the last run decode -- which
#            sub-context cannot touch -- moved 2.4%, which is larger than the
#            prefill effect being measured. Do not attribute a per-stage delta
#            from this mode to the mechanism.
#
# Setup (once). Python MUST match the fork env (3.12) -- the host-side stages are
# pure-Python hot paths and 3.11+ sped those up a lot, so a version gap reads as
# sub-context overhead. Do NOT force-reinstall torch here: both envs already sit
# on torch 2.9.1, and desyncing them makes the GPU numbers incomparable too.
#   conda create -n sglang_orig python=3.12 -y && conda activate sglang_orig
#   export PYTHONNOUSERSITE=1          # ~/.local has a broken torch dist-info
#   pip install --upgrade pip && pip install uv
#   uv pip install --prerelease=allow sglang==0.5.9      # PIN THE VERSION
#   PYTHONNOUSERSITE=1 python /home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9/instrument_sglang.py
#
# Then:
#   ./run_ab.sh record    capture one real agent run (run swe_test.sh alongside)
#   ./run_ab.sh toggle    split off vs on, same binary   <-- start here
#   ./run_ab.sh replay    upstream vs fork, two envs
#
# The TRACE-* probes are off unless SUBCTX_TRACE=1: they print per insert and
# per namespace match, several of them from inside the regions host_timer is
# measuring, and they only fire when the split is on -- so left enabled they
# bill their own debug output to the mechanism under test.
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=${OUT:-$REPO/ab_out}
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}
PORT=${PORT:-30000}
GEN_TOKENS=${GEN_TOKENS:-256}
BASE_ENV=${BASE_ENV:-sglang_orig}
FORK_ENV=${FORK_ENV:-sglangv59}
mkdir -p "$OUT"

source /home/t2503-3090/miniconda3/etc/profile.d/conda.sh

# $1=env  $2=PYTHONPATH (empty for baseline)  $3=trace  $4=stage prefix  $5=capture
launch() {
  pkill -f "sglang\.launch_server" 2>/dev/null || true
  sleep 6
  [ -n "$3" ] && rm -f "$3"
  [ -n "$4" ] && rm -f "$4".* 2>/dev/null || true
  conda activate "$1"
  # ~/.local/lib/python3.*/site-packages sits AHEAD of the env on sys.path and
  # holds an orphaned torch-2.10.0.dist-info with no METADATA, which makes
  # importlib.metadata.version("torch") return None and transformers blow up in
  # version.parse(). Cut user-site off entirely rather than editing ~/.local.
  export PYTHONNOUSERSITE=1
  # An empty PYTHONPATH still puts CWD on sys.path; unset it for the baseline so
  # nothing from a fork checkout can shadow the stock install.
  if [ -n "$2" ]; then export PYTHONPATH="$2"; else unset PYTHONPATH; fi
  SGLANG_FORWARD_TRACE="$3" \
  SGLANG_STAGE_TRACE="$4" \
  SGLANG_CAPTURE_REQUESTS="$5" \
  SGLANG_DISABLE_SUBCONTEXT="${SUBCTX_OFF:-}" \
  SGLANG_SUBCTX_TRACE="${SUBCTX_TRACE:-}" \
  nohup python -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --quantization moe_wna16 \
    --tool-call-parser qwen3_coder \
    --enable-cache-report \
    --port "$PORT" --mem-fraction-static 0.85 \
    > "$OUT/server_${LOGTAG:-run}.log" 2>&1 &
  echo -n "  [$1] waiting"
  for _ in $(seq 1 300); do
    if grep -q "fired up and ready" "$OUT/server_${LOGTAG:-run}.log"; then echo " ready"; return 0; fi
    if ! pgrep -f "sglang\.launch_server" > /dev/null; then
      echo " DIED"; tail -30 "$OUT/server_${LOGTAG:-run}.log"; return 1
    fi
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -30 "$OUT/server_${LOGTAG:-run}.log"; return 1
}

# Replay the capture against whatever server is up. swe_test.sh sends
# --model meta-llama/Llama-3.1-8B-Instruct while the server actually holds
# Qwen3-Coder, so the captured bodies carry the wrong name; pin it to what is
# really loaded. The model field never reaches the prompt.
# $1=trace  $2=stage prefix  $3=client json
replay_arm() {
  python "$REPO/subcontext_bench.py" replay "$OUT/requests.jsonl" \
    --url "http://127.0.0.1:$PORT" \
    --trace "$1" --stage-trace "$2" --out "$3" \
    --model "$MODEL" --gen-tokens "$GEN_TOKENS"
}

# Confirm the baseline env is really stock and really instrumented.
check_baseline() {
  conda activate "$BASE_ENV"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  python - <<'PY'
import sys, pathlib, sysconfig
import sglang
root = pathlib.Path(sglang.__file__).resolve().parent
print(f"  baseline sglang: {sglang.__version__} at {root}")
if "sglang-v.0.5.9" in str(root) or "MiaoChen/sglang" in str(root):
    sys.exit(f"REFUSING: baseline env resolves to a fork checkout ({root}), not stock sglang")
# The timed stages are pure-Python hot paths (jinja render, tokenizer calls,
# radix tree walk). 3.11+ made those materially faster, so a version gap between
# the arms shows up as sub-context overhead that is really interpreter speed.
import os
want = os.environ.get("EXPECT_PY", "3.12")
have = f"{sys.version_info.major}.{sys.version_info.minor}"
if have != want:
    sys.exit(f"REFUSING: baseline runs Python {have} but the fork arm runs {want}.\n"
             f"Rebuild the baseline env at python={want}, or set EXPECT_PY to override\n"
             f"if you only care about the GPU numbers.")
src = (root / "srt" / "entrypoints" / "openai" / "serving_chat.py").read_text()
if "_compute_sub_context_ids" in src:
    sys.exit("REFUSING: baseline has the sub-context split -- not a clean baseline")
if "host_timer.timed" not in (root / "srt" / "mem_cache" / "radix_cache.py").read_text():
    sys.exit("baseline is NOT instrumented; run instrument_sglang.py in this env first")
# The probes are copied, not shared, so the baseline can be running an older
# host_timer than the fork. Without the marker it reports a window that starts
# at process launch while the fork's starts after warm-up, and the difference
# between two unequal windows gets read as mechanism cost.
if "_check_mark" not in (root / "srt" / "utils" / "host_timer.py").read_text():
    sys.exit("baseline has a stale host_timer (no warm-up marker).\n"
             "Re-run instrument_sglang.py in this env to refresh the probes.")
print("  baseline is stock + instrumented")
PY
}

finish() {
  # SIGTERM, not SIGKILL, so the timers get to flush.
  pkill -TERM -f "sglang\.launch_server" 2>/dev/null || true
  sleep 8
  echo; echo "======== GPU (CUDA events) ========"
  python "$REPO/subcontext_bench.py" report "$OUT/trace_base.jsonl" "$OUT/trace_sub.jsonl"
  echo; echo "======== HOST (CPU stages) ========"
  python "$REPO/subcontext_bench.py" stages "$OUT/stage_base" "$OUT/stage_sub"
}

case "${1:-}" in
  record)
    rm -f "$OUT/requests.jsonl"
    LOGTAG=record launch "$FORK_ENV" "$REPO/python" "" "" "$OUT/requests.jsonl"
    cat <<EOF

This server is now up and recording every chat request it receives.
It does NOT run SWE-bench itself -- you drive that as usual:

  STEP 2, in another shell:
      bash /home/t2503-3090/Desktop/MiaoChen/swe_bench/swe_test.sh

  STEP 3, once the agent has finished:
      $0 replay

Watch the capture grow with:  wc -l $OUT/requests.jsonl
EOF
    ;;
  replay)
    [ -s "$OUT/requests.jsonl" ] || { echo "no capture; run '$0 record' first"; exit 1; }
    echo "captured $(wc -l < "$OUT/requests.jsonl") requests"
    check_baseline

    echo; echo "======== BASELINE (stock sglang) ========"
    LOGTAG=base launch "$BASE_ENV" "" "$OUT/trace_base.jsonl" "$OUT/stage_base" ""
    conda activate "$FORK_ENV"
    replay_arm "$OUT/trace_base.jsonl" "$OUT/stage_base" "$OUT/client_base.json"

    echo; echo "======== SUB-CONTEXT (this fork) ========"
    LOGTAG=sub launch "$FORK_ENV" "$REPO/python" "$OUT/trace_sub.jsonl" "$OUT/stage_sub" ""
    replay_arm "$OUT/trace_sub.jsonl" "$OUT/stage_sub" "$OUT/client_sub.json"

    finish
    ;;
  toggle)
    # Same binary, same env, same weights -- the arms differ by one boolean.
    # This is the comparison that costs the *mechanism*: everything else about
    # the fork (extra_key plumbing, the stitch code path, this script's own
    # probes) is present on both sides and cancels. What it cannot answer is
    # "is my fork better than upstream" -- that needs `replay`.
    [ -s "$OUT/requests.jsonl" ] || { echo "no capture; run '$0 record' first"; exit 1; }
    echo "captured $(wc -l < "$OUT/requests.jsonl") requests"

    echo; echo "======== SPLIT OFF (same binary) ========"
    SUBCTX_OFF=1 LOGTAG=off \
      launch "$FORK_ENV" "$REPO/python" "$OUT/trace_base.jsonl" "$OUT/stage_base" ""
    grep -q "Sub-context split DISABLED" "$OUT/server_off.log" \
      || { echo "REFUSING: the split did not report itself disabled"; exit 1; }
    replay_arm "$OUT/trace_base.jsonl" "$OUT/stage_base" "$OUT/client_base.json"

    echo; echo "======== SPLIT ON ========"
    LOGTAG=on launch "$FORK_ENV" "$REPO/python" "$OUT/trace_sub.jsonl" "$OUT/stage_sub" ""
    replay_arm "$OUT/trace_sub.jsonl" "$OUT/stage_sub" "$OUT/client_sub.json"

    finish
    ;;
  *)
    echo "usage: $0 {record|replay|toggle}"; exit 1;;
esac
