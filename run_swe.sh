#!/bin/bash
# SWE-bench arm of the sub-context A/B.
#
#   ./run_swe.sh record   bring up a server that records every chat request;
#                         you drive swe_test.sh yourself in another shell
#   ./run_swe.sh toggle   replay that capture twice on the same binary,
#                         split OFF then ON, and print the tables
#
# Everything lands in ab_out/swe. MASLab is run_mas.sh.
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=$REPO/ab_out/swe
MODEL=QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
PORT=30000
CTXLEN=32768          # 16384 讓 astropy 那題在 16393 tokens 撞牆；KV pool 放得下 32k
GEN_TOKENS=32          # replay generates a fixed length so both arms do equal work
# Requests in flight during a replay. 1 keeps the arms reproducible; raise it when
# the question is serving capacity rather than what the split does to one request.
CONC=${CONC:-1}
ENV=sglangv59
REQUESTS=$OUT/requests.jsonl
SWE_TEST=/home/t2503-3090/Desktop/MiaoChen/swe_bench/swe_test.sh

mkdir -p "$OUT"
source /home/t2503-3090/miniconda3/etc/profile.d/conda.sh
conda activate $ENV
export PYTHONNOUSERSITE=1        # ~/.local has a broken torch dist-info ahead of the env
export PYTHONPATH=$REPO/python   # run THIS checkout, not the installed sglang

# The server command lives here, once. Callers set LOG, and optionally
# CAPTURE / TRACE / STAGE / SUBCTX_OFF / SUBCTX_TRACE before calling.
# `sglang::scheduler` is where the weights and the KV pool live, and setproctitle
# renames it out of reach of a "sglang.launch_server" match -- see run_mas.sh.
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
  SGLANG_SUBCTX_TRACE=${SUBCTX_TRACE:-} \
  nohup python -u -m sglang.launch_server \
    --model-path $MODEL \
    --context-length $CTXLEN \
    --quantization moe_wna16 \
    --tool-call-parser qwen3_coder \
    --enable-cache-report \
    --port $PORT \
    --mem-fraction-static 0.90 \
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

# Replay the capture against whatever server is up. Callers set CLIENT (+ TRACE/STAGE).
# swe_test.sh sends a different --model than the server really holds, so pin it here.
replay() {
  python $REPO/subcontext_bench.py replay "$REQUESTS" \
    --url http://127.0.0.1:$PORT \
    --trace "${TRACE:-}" --stage-trace "${STAGE:-}" --out "$CLIENT" \
    --model $MODEL --gen-tokens $GEN_TOKENS --concurrency $CONC --save-text
}

case "${1:-}" in
  record)
    rm -f "$REQUESTS"
    LOG=$OUT/server_record.log CAPTURE=$REQUESTS launch
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

    echo; echo "======== SPLIT OFF ========"
    SUBCTX_OFF=1 LOG=$OUT/server_off.log \
      TRACE=$OUT/trace_base.jsonl STAGE=$OUT/stage_base launch
    # An arm meant to be the baseline that silently ran with the split on is worse
    # than no arm at all: it looks like a valid comparison.
    grep -q "Sub-context split DISABLED" $OUT/server_off.log \
      || { echo "REFUSING: the split did not report itself disabled"; exit 1; }
    TRACE=$OUT/trace_base.jsonl STAGE=$OUT/stage_base CLIENT=$OUT/client_base.json replay

    echo; echo "======== SPLIT ON ========"
    LOG=$OUT/server_on.log TRACE=$OUT/trace_sub.jsonl STAGE=$OUT/stage_sub launch
    TRACE=$OUT/trace_sub.jsonl STAGE=$OUT/stage_sub CLIENT=$OUT/client_sub.json replay

    stop

    # Keep a copy of the tables: they are otherwise only in the terminal. The inputs
    # stay on disk, so this is a convenience, not the record of the run.
    {
      echo; echo "======== GPU (CUDA events) ========"
      python $REPO/subcontext_bench.py report $OUT/trace_base.jsonl $OUT/trace_sub.jsonl
      echo; echo "======== HOST (CPU stages) ========"
      python $REPO/subcontext_bench.py stages $OUT/stage_base $OUT/stage_sub
      echo; echo "======== PARITY (generated text) ========"
      python $REPO/subcontext_bench.py parity $OUT/client_base.json $OUT/client_sub.json || true
      echo; echo "======== SUMMARY (all arms) ========"
      python $REPO/subcontext_bench.py summary "$OUT"
    } | tee $OUT/tables.txt
    ;;

  *)
    echo "usage: $0 {record|toggle}"
    echo "  record  server records SWE-bench traffic; you run $SWE_TEST"
    echo "  toggle  replay that capture, split off vs on  ->  $OUT"
    exit 1;;
esac
