#!/bin/bash
# A/B the sub-context mechanism against a pristine upstream sglang, on real
# SWE-bench traffic.
#
#   BASELINE   conda env $BASE_ENV, stock sglang==0.5.9 from PyPI, instrumented
#              by instrument_sglang.py. No PYTHONPATH -- imports its own copy.
#   TREATMENT  conda env $FORK_ENV + PYTHONPATH into this fork.
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
#   ./run_ab.sh replay    replay it against both, report GPU + host-stage diffs
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
print("  baseline is stock + instrumented")
PY
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
    # swe_test.sh sends --model meta-llama/Llama-3.1-8B-Instruct while the server
    # actually holds Qwen3-Coder, so the captured bodies carry the wrong name.
    # Pin it to what is really loaded; the model field never reaches the prompt.
    python "$REPO/subcontext_bench.py" replay "$OUT/requests.jsonl" \
      --url "http://127.0.0.1:$PORT" --trace "$OUT/trace_base.jsonl" \
      --model "$MODEL" \
      --gen-tokens "$GEN_TOKENS" --out "$OUT/client_base.json"

    echo; echo "======== SUB-CONTEXT (this fork) ========"
    LOGTAG=sub launch "$FORK_ENV" "$REPO/python" "$OUT/trace_sub.jsonl" "$OUT/stage_sub" ""
    python "$REPO/subcontext_bench.py" replay "$OUT/requests.jsonl" \
      --url "http://127.0.0.1:$PORT" --trace "$OUT/trace_sub.jsonl" \
      --model "$MODEL" \
      --gen-tokens "$GEN_TOKENS" --out "$OUT/client_sub.json"

    # SIGTERM, not SIGKILL, so the timers get to flush.
    pkill -TERM -f "sglang\.launch_server" 2>/dev/null || true
    sleep 8

    echo; echo "======== GPU (CUDA events) ========"
    python "$REPO/subcontext_bench.py" report "$OUT/trace_base.jsonl" "$OUT/trace_sub.jsonl"
    echo; echo "======== HOST (CPU stages) ========"
    python "$REPO/subcontext_bench.py" stages "$OUT/stage_base" "$OUT/stage_sub"
    ;;
  *)
    echo "usage: $0 {record|replay}"; exit 1;;
esac
