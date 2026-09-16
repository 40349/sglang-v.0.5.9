#!/bin/bash
# SWE-bench arm of the sub-context A/B.
#
#   ./run_swe.sh record   server records every chat request; you drive swe_test.sh
#   ./run_swe.sh toggle   replay that capture once per arm and print the tables
#
# Arms. Same words as sglang_server.sh and run_mas.sh; `rot` is rotation and Stage 2
# in all three:
#
#   off  no split at all -- the stock single-namespace radix cache
#   on   split into per-block namespaces, displaced hits dropped
#   rot  + rotate a displaced hit to where it is reused, + Stage 2
#   idx  + find blocks by content anywhere in the prompt, and prefill the gaps
#   cdc  + cut the blocks on content too, not on the roles the prompt was built from
#
# Default is "off cdc". on, rot and idx are the rungs between them and run when named.
#
#   ARMS="off rot idx cdc" ./run_swe.sh toggle          put the middle rungs back
#   AUDIT=1 FULL=1 ./run_swe.sh toggle                  correctness pass, timings unusable
#   REQUESTS=~/work/traces/requests_on_42.jsonl ...     a capture recorded elsewhere
#   MODEL=... CTXLEN=... ./run_swe.sh ...               a smaller model for a smoke run
#
# `toggle` restarts the server per arm and reads the traces it writes locally: run it
# on the box holding the GPU. Output lands in ab_out/swe. MASLab is run_mas.sh.
set -euo pipefail

# The script's own location, so this measures the checkout it sits in.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$REPO/ab_out/swe}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
PORT=${PORT:-30000}
CTXLEN=${CTXLEN:-32768}       # the longest SWE-bench prompt measured is 32,996 tokens
GEN_TOKENS=${GEN_TOKENS:-32}  # fixed generated length per replayed request
CONC=${CONC:-1}               # requests in flight
ENV=${ENV:-sglangv59}
# Empty means "drop the flag". No colon: ${VAR:-x} substitutes for empty too.
QUANT=${QUANT-}
TOOL_PARSER=${TOOL_PARSER-qwen}

ARMS=${ARMS:-"off cdc"}

# Pinned for every arm: triton is the only backend that takes a per-position mask.
BACKEND=${BACKEND:-triton}

# The finish-path audits walk the whole tree once per pass; the numbers below then
# include the audit.
AUDIT=${AUDIT:-}

# Let each request stop where it wants instead of pinning the length with ignore_eos.
# The arms then generate different amounts and the timings stop being comparable.
FULL=${FULL:-}

# conda.sh, resolved from CONDA_EXE rather than hardcoded.
if [ -z "${CONDA_SH:-}" ]; then
  _base=${CONDA_EXE:-}; _base=${_base%/bin/conda}
  [ -n "$_base" ] || _base=$(conda info --base 2>/dev/null || true)
  if [ -n "$_base" ] && [ -r "$_base/etc/profile.d/conda.sh" ]; then
    CONDA_SH=$_base/etc/profile.d/conda.sh
  else
    CONDA_SH=/home/t2503-3090/miniconda3/etc/profile.d/conda.sh
  fi
fi
# Override to replay a capture recorded elsewhere.
REQUESTS=${REQUESTS:-$OUT/requests.jsonl}
SWE_TEST=${SWE_TEST:-/home/t2503-3090/Desktop/MiaoChen/swe_bench/swe_test.sh}
CHECK_ARM=$REPO/scripts/subcontext_sim/check_remote_arm.py

mkdir -p "$OUT"
[ -r "$CONDA_SH" ] || {
  echo "no conda.sh at $CONDA_SH"
  echo "set CONDA_SH=<conda base>/etc/profile.d/conda.sh   (conda info --base)"
  exit 1
}
source "$CONDA_SH"
conda activate $ENV 2>/dev/null || {
  echo "no conda env '$ENV' under $CONDA_SH; set ENV=<name>. available:"
  conda env list
  exit 1
}
export PYTHONNOUSERSITE=1        # ignore ~/.local
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/python   # run THIS checkout, not the installed sglang

arm_switches() {
  # Cleared every time; an arm that does not set one gets the empty value.
  export SUBCTX_OFF= SUBCTX_ROTATE= SUBCTX_ACROSS= SUBCTX_INDEX= SUBCTX_SPLIT=blocks
  case "$1" in
    off) export SUBCTX_OFF=1 ;;
    on)  ;;
    rot) export SUBCTX_ROTATE=1 SUBCTX_ACROSS=1 ;;
    idx) export SUBCTX_ROTATE=1 SUBCTX_INDEX=1 ;;
    cdc) export SUBCTX_ROTATE=1 SUBCTX_INDEX=1 SUBCTX_SPLIT=cdc ;;
    *)   echo "unknown arm '$1'; want any of: off on rot idx cdc"; exit 1 ;;
  esac
}

# The bench names files by stem: client_base.json is the off arm, client_sub.json the
# on arm. The stems are what `summary` looks for.
arm_stem() {
  case "$1" in
    off) echo base ;;
    on)  echo sub ;;
    *)   echo "$1" ;;
  esac
}

# Ask the server what it is rather than trusting the switches.
verify_arm() {
  local split rotate index
  case "$1" in
    off) split=false; rotate=false; index=false ;;
    on)  split=true;  rotate=false; index=false ;;
    rot) split=true;  rotate=true;  index=false ;;
    idx) split=true;  rotate=true;  index=true  ;;
    cdc) split=true;  rotate=true;  index=true  ;;
  esac
  python "$CHECK_ARM" "http://127.0.0.1:$PORT" \
    --split $split --rotate $rotate --index $index \
    --split-mode "${SUBCTX_SPLIT:-blocks}" \
    --audit "$([ -n "$AUDIT" ] && echo true || echo false)"
}

# `sglang::scheduler` holds the weights and the KV pool, and setproctitle renames it out
# of reach of a "sglang.launch_server" match. `[s]` stops the pattern matching this shell.
SGLANG_PROCS='[s]glang::|[s]glang\.launch_server|[s]glang\.bench|[s]glang\.srt'

launch() {
  pkill -TERM -f "$SGLANG_PROCS" 2>/dev/null || true
  sleep 10
  if [ -n "${TRACE:-}" ]; then rm -f "$TRACE"; fi
  if [ -n "${STAGE:-}" ]; then rm -f "$STAGE".*; fi
  SGLANG_CAPTURE_REQUESTS=${CAPTURE:-} \
  SGLANG_FORWARD_TRACE=${TRACE:-} \
  SGLANG_STAGE_TRACE=${STAGE:-} \
  SGLANG_DISABLE_SUBCONTEXT=${SUBCTX_OFF:-} \
  SGLANG_SUBCONTEXT_ROTATE=${SUBCTX_ROTATE:-} \
  SGLANG_SUBCONTEXT_ROTATE_ACROSS=${SUBCTX_ACROSS:-} \
  SGLANG_SUBCTX_INDEX=${SUBCTX_INDEX:-} \
  SGLANG_SUBCTX_SPLIT=${SUBCTX_SPLIT:-blocks} \
  SGLANG_SUBCTX_AUDIT=${AUDIT:-} \
  SGLANG_SUBCTX_TRACE=${SUBCTX_TRACE:-} \
  nohup python -u -m sglang.launch_server \
    --model-path $MODEL \
    --context-length $CTXLEN \
    ${QUANT:+--quantization $QUANT} \
    ${TOOL_PARSER:+--tool-call-parser $TOOL_PARSER} \
    --attention-backend $BACKEND \
    --enable-cache-report \
    --port $PORT \
    --mem-fraction-static ${MEMFRAC:-0.90} \
    > "$LOG" 2>&1 &
  echo -n "  waiting"
  for _ in $(seq 1 300); do
    if grep -q "fired up and ready" "$LOG"; then echo " ready"; return 0; fi
    if ! pgrep -f "[s]glang\.launch_server" > /dev/null; then
      echo " DIED"; tail -30 "$LOG"; exit 1
    fi
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -30 "$LOG"; exit 1
}

stop() {
  pkill -TERM -f "$SGLANG_PROCS" 2>/dev/null || true   # TERM so timers flush
  sleep 10
}

# Callers set CLIENT (+ TRACE/STAGE). --model is pinned to what the server holds.
replay() {
  python $REPO/subcontext_bench.py replay "$REQUESTS" \
    --url http://127.0.0.1:$PORT \
    --trace "${TRACE:-}" --stage-trace "${STAGE:-}" --out "$CLIENT" \
    --model $MODEL --gen-tokens $GEN_TOKENS --concurrency $CONC --save-text \
    ${FULL:+--full}
}

# The tables at the end are written nowhere else. CONSOLE= disables the tee.
CONSOLE=${CONSOLE-$OUT/${1:-run}.txt}
if [ -n "$CONSOLE" ]; then
  echo "### $(date -Is)  $0 ${*:-}  ARMS='$ARMS' BACKEND=$BACKEND AUDIT=${AUDIT:-off} CONC=$CONC GEN_TOKENS=$GEN_TOKENS" >> "$CONSOLE"
  exec > >(tee -a "$CONSOLE") 2>&1
  TEE_PID=$!
  trap 'exec 1>&- 2>&-; wait $TEE_PID 2>/dev/null || true' EXIT
  echo "console -> $CONSOLE"
fi

case "${1:-}" in
  record)
    # Only ever the capture this script owns.
    [ "$REQUESTS" = "$OUT/requests.jsonl" ] || {
      echo "REFUSING: REQUESTS points at $REQUESTS, which this script did not make."
      echo "  Unset REQUESTS to record a new one into $OUT."
      exit 1
    }
    rm -f "$REQUESTS"
    arm_switches "${ARM:-on}"
    LOG=$OUT/server_record.log CAPTURE=$REQUESTS launch
    verify_arm "${ARM:-on}"
    cat <<TXT

Recording every chat request. This does NOT run SWE-bench -- drive it yourself:

  in another shell:   bash $SWE_TEST
  when it finishes:   $0 toggle

Watch the capture grow with:  wc -l $REQUESTS
TXT
    ;;

  toggle)
    [ -s "$REQUESTS" ] || { echo "no capture at $REQUESTS; run '$0 record' first"; exit 1; }
    echo "captured $(wc -l < "$REQUESTS") requests"
    echo "arms: $ARMS   backend: $BACKEND   audit: ${AUDIT:-off}"
    baseline=$(echo $ARMS | awk '{print $1}')

    for arm in $ARMS; do
      stem=$(arm_stem "$arm")
      echo; echo "======== ARM: $arm ========"
      arm_switches "$arm"
      LOG=$OUT/server_$arm.log TRACE=$OUT/trace_$stem.jsonl STAGE=$OUT/stage_$stem launch
      verify_arm "$arm"
      TRACE=$OUT/trace_$stem.jsonl STAGE=$OUT/stage_$stem \
        CLIENT=$OUT/client_$stem.json replay
      # The index's own counters; a request that did not fit one prefill pass reports
      # as having found nothing.
      case "$arm" in
        idx|cdc) grep "sub-context index:" $OUT/server_$arm.log | tail -1 || true ;;
      esac
    done

    stop

    {
      if [ -n "$AUDIT" ] || [ -n "$FULL" ]; then
        echo
        [ -n "$AUDIT" ] && echo "!! AUDIT=1: every pass walked the whole tree, so the" \
          "GPU and HOST numbers below price the audit, not the split."
        [ -n "$FULL" ] && echo "!! FULL=1: the arms generated different amounts, so the" \
          "timing columns are not comparable between them."
        echo "!! Correctness only."
      fi
      base_stem=$(arm_stem "$baseline")
      for arm in $ARMS; do
        [ "$arm" = "$baseline" ] && continue
        stem=$(arm_stem "$arm")
        echo; echo "######## $baseline vs $arm ########"
        echo; echo "======== GPU (CUDA events) ========"
        python $REPO/subcontext_bench.py report \
          $OUT/trace_$base_stem.jsonl $OUT/trace_$stem.jsonl
        echo; echo "======== HOST (CPU stages) ========"
        python $REPO/subcontext_bench.py stages $OUT/stage_$base_stem $OUT/stage_$stem
        echo; echo "======== PARITY (generated text) ========"
        python $REPO/subcontext_bench.py parity \
          $OUT/client_$base_stem.json $OUT/client_$stem.json || true
      done
      stems=""
      for arm in $ARMS; do stems="$stems,$(arm_stem "$arm")"; done
      echo; echo "======== SUMMARY (all arms) ========"
      python $REPO/subcontext_bench.py summary "$OUT" --arms "${stems#,}"
    } | tee $OUT/tables.txt
    ;;

  *)
    echo "usage: $0 {record|toggle}"
    echo "  record  server records SWE-bench traffic; you run $SWE_TEST"
    echo "  toggle  replay that capture once per arm  ->  $OUT"
    echo
    echo "  ARMS='off idx'        which arms, in order; the first is the baseline"
    echo "                        off|on|rot|idx; on and rot are the rungs between"
    echo "  AUDIT=1               turn on the leak/ownership audits (timings unusable)"
    echo "  FULL=1                let requests stop naturally, so one can stop during"
    echo "                        prefill -- the path a pinned replay never reaches"
    echo "  BACKEND=triton        attention backend, pinned across arms"
    echo "  MODEL= CTXLEN= PORT=  override for a smaller smoke run"
    echo "  REQUESTS=<path>       replay a capture recorded elsewhere (e.g. the H200)"
    echo "  QUANT= TOOL_PARSER=   empty to drop the flag (a non-MoE model has neither)"
    echo "  MEMFRAC=0.90          KV pool share"
    exit 1;;
esac
