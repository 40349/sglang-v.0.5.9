#!/bin/bash
#SBATCH --job-name=swe_toggle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=swe_toggle_%j.log
#
# SWE-bench arm of the sub-context A/B. Starts sglang_server.sh once per arm; arms and
# server knobs are that script's and pass straight through.
#
#   ./run_swe.sh record   start ARM's server and record every chat request; you drive
#                         the agent
#   ./run_swe.sh toggle   replay that capture once per arm in ARMS and print the tables
#   sbatch run_swe.sh toggle     the same as a batch job (submit from the checkout)
#
#   ARMS="off cdc cdc@0.15" CTXLEN=40960 ./run_swe.sh toggle
#   ARMS="off rot idx cdc" ./run_swe.sh toggle          the middle rungs; the first arm
#                                                       is the baseline
#   AUDIT=1 FULL=1 ./run_swe.sh toggle                  correctness pass, timings unusable
#   REQUESTS=~/work/traces/requests_on_42.jsonl ...     a capture recorded elsewhere
#
# `toggle` reads the traces the server writes locally: run it on the box holding the
# GPU. Output lands in ab_out/swe.
set -euo pipefail

# sbatch runs a spool copy of this file; the checkout is then where it was submitted.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[ -d "$REPO/python/sglang" ] || REPO=${SLURM_SUBMIT_DIR:-}
[ -d "$REPO/python/sglang" ] || { echo "REFUSING: no checkout found; submit from it"; exit 1; }
SERVER=$REPO/sglang_server.sh
OUT=${OUT:-$REPO/ab_out/swe}
GEN_TOKENS=${GEN_TOKENS:-32}  # fixed generated length per replayed request
CONC=${CONC:-1}               # requests in flight
# Use each request's own max_tokens instead of pinning the length with ignore_eos.
FULL=${FULL:-}
# Slot-ownership audits in the server (a full tree walk per pass; timings include it).
AUDIT=${AUDIT:-}
# Override to replay a capture recorded elsewhere.
REQUESTS=${REQUESTS:-$OUT/requests.jsonl}
SWE_TEST=${SWE_TEST:-/home/t2503-3090/Desktop/MiaoChen/swe_bench/swe_test.sh}
CHECK_ARM=$REPO/scripts/subcontext_sim/check_remote_arm.py

ARM=${ARM:-on}
ARMS=${ARMS:-"off cdc"}
export MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
export PORT=${PORT:-30000}
# Replay saves only `content`, and a reasoning parser would move the pinned generation,
# all of it inside <think>, out of it.
export REASONING_PARSER=${REASONING_PARSER-}

# Every arm is checked before the first one runs.
check_arm() {
  local out
  out=$(ARM=$1 bash "$SERVER" check) || { echo "$out"; exit 1; }
}
case "${1:-}" in
  record) check_arm "$ARM" ;;
  toggle) for a in $ARMS; do check_arm "$a"; done ;;
esac

mkdir -p "$OUT"
if command -v ml > /dev/null 2>&1; then ml load miniconda3; fi
eval "$(conda shell.bash hook)"
conda activate "${ENV:-sglangv59}"
export PYTHONNOUSERSITE=1        # ignore ~/.local
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/python   # run THIS checkout, not the installed sglang

# File stem per arm, as `summary` expects: off -> base, on -> sub, others by their tag.
arm_stem() {
  case "${1%%@*}" in
    off) echo base ;;
    on)  echo sub ;;
    *)   ARM=$1 bash "$SERVER" check ;;
  esac
}

# Ask the server what it is rather than trusting the switches.
verify_arm() {
  python "$CHECK_ARM" "http://127.0.0.1:$PORT" --arm "$1" \
    --audit "$([ -n "${AUDIT:-}" ] && echo true || echo false)"
}

# `sglang::scheduler` holds the weights and the KV pool, and setproctitle renames it out
# of reach of a "sglang.launch_server" match. `[s]` stops the pattern matching this shell.
SGLANG_PROCS='[s]glang::|[s]glang\.launch_server|[s]glang\.bench|[s]glang\.srt'

stop() {
  pkill -TERM -f "$SGLANG_PROCS" 2>/dev/null || true   # TERM so timers flush
  sleep 10
}

# Starts ARM $1 through sglang_server.sh and waits for it. Callers set LOG, and
# optionally CAPTURE, FWD_TRACE, STAGE.
launch() {
  stop
  if [ -n "${FWD_TRACE:-}" ]; then rm -f "$FWD_TRACE"; fi
  if [ -n "${STAGE:-}" ]; then rm -f "$STAGE".*; fi
  rm -f "$LOG"
  ARM=$1 SERVER_LOG=$LOG CAPTURE=${CAPTURE:-} FWD_TRACE=${FWD_TRACE:-} STAGE=${STAGE:-} \
    nohup bash "$SERVER" > "$LOG.start" 2>&1 &
  local pid=$!
  echo -n "  waiting"
  for _ in $(seq 1 300); do
    if [ -f "$LOG" ] && grep -q "fired up and ready" "$LOG"; then echo " ready"; return 0; fi
    if ! kill -0 $pid 2>/dev/null; then
      echo " DIED"; cat "$LOG.start"; [ ! -f "$LOG" ] || tail -30 "$LOG"; exit 1
    fi
    echo -n .; sleep 2
  done
  echo " TIMEOUT"; tail -30 "$LOG"; exit 1
}

# Callers set CLIENT (+ FWD_TRACE/STAGE). --model is pinned to what the server holds.
replay() {
  python $REPO/subcontext_bench.py replay "$REQUESTS" \
    --url http://127.0.0.1:$PORT \
    --trace "${FWD_TRACE:-}" --stage-trace "${STAGE:-}" --out "$CLIENT" \
    --model $MODEL --gen-tokens $GEN_TOKENS --concurrency $CONC --save-text \
    ${FULL:+--full}
}

# The tables at the end are written nowhere else. CONSOLE= disables the tee.
CONSOLE=${CONSOLE-$OUT/${1:-run}.txt}
if [ -n "$CONSOLE" ]; then
  echo "### $(date -Is)  $0 ${*:-}  ARMS='$ARMS' AUDIT=${AUDIT:-off} CONC=$CONC GEN_TOKENS=$GEN_TOKENS" >> "$CONSOLE"
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
    LOG=$OUT/server_record.log CAPTURE=$REQUESTS launch "$ARM"
    verify_arm "$ARM"
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
    echo "arms: $ARMS   audit: ${AUDIT:-off}"
    baseline=$(echo $ARMS | awk '{print $1}')

    for arm in $ARMS; do
      stem=$(arm_stem "$arm")
      echo; echo "======== ARM: $arm ========"
      log=$OUT/server_$(ARM=$arm bash "$SERVER" check).log
      LOG=$log FWD_TRACE=$OUT/trace_$stem.jsonl STAGE=$OUT/stage_$stem launch "$arm"
      verify_arm "$arm"
      FWD_TRACE=$OUT/trace_$stem.jsonl STAGE=$OUT/stage_$stem \
        CLIENT=$OUT/client_$stem.json replay
      # The index's own counters.
      case "${arm%%@*}" in
        idx|cdc) grep "sub-context index:" "$log" | tail -1 || true ;;
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
    echo "  ARM=on                the arm 'record' runs"
    echo "  ARMS='off cdc@0.15'   which arms 'toggle' replays, in order; the first is the"
    echo "                        baseline. off|on|rot|idx|cdc, idx/cdc@ratio (sglang_server.sh)"
    echo "  AUDIT=1               turn on the leak/ownership audits (timings unusable)"
    echo "  FULL=1                let requests stop naturally, so one can stop during"
    echo "                        prefill -- the path a pinned replay never reaches"
    echo "  REQUESTS=<path>       replay a capture recorded elsewhere (e.g. the H200)"
    echo "  MODEL CTXLEN PORT MEMFRAC BACKEND CHUNKED_PREFILL TOOL_PARSER QUANT ..."
    echo "                        server knobs, passed through to sglang_server.sh"
    exit 1;;
esac
