#!/usr/bin/env bash
# One-shot check that sub-context KV rotation is alive and doing what it claims.
#
# Launches a server with both rotation stages on, sends three requests built so that
# one hits each rotation path (see scripts/subcontext_sim/probe_rotate_client.py), and
# prints the per-pass forward trace.
#
# Diagnostic only -- do NOT read GPU or host timings out of this run: SGLANG_SUBCTX_TRACE
# is on, and its probes sit inside the regions host_timer measures.
#
#   ./probe_rotate.sh            rotation on  (both stages)
#   ROTATE=0 ./probe_rotate.sh   control: same requests, same binary, rotation off
#
# Run both. The control is what makes the numbers mean anything: it shows the same
# three requests dropping the hits that the rotation arm wins back.
#
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=$REPO/ab_out/rotate
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}
PORT=${PORT:-30000}
CTXLEN=${CTXLEN:-16384}
ENV=${ENV:-sglangv59}

ROTATE=${ROTATE:-1}
if [ "$ROTATE" = "0" ]; then SUF=_off; else SUF=_on; fi

mkdir -p "$OUT"
LOG=$OUT/server$SUF.log
TRACE=$OUT/trace$SUF.jsonl

source /home/t2503-3090/miniconda3/etc/profile.d/conda.sh
conda activate $ENV
export PYTHONPATH=$REPO/python   # run THIS checkout, not the installed sglang

stop() { pkill -f "[s]glang\.launch_server" 2>/dev/null || true; sleep 5; }
trap stop EXIT

stop
rm -f "$TRACE"
SGLANG_FORWARD_TRACE=$TRACE \
SGLANG_SUBCTX_TRACE=1 \
SGLANG_SUBCONTEXT_ROTATE=$( [ "$ROTATE" = "0" ] && echo "" || echo 1 ) \
SGLANG_SUBCONTEXT_ROTATE_ACROSS=$( [ "$ROTATE" = "0" ] && echo "" || echo 1 ) \
nohup python -u -m sglang.launch_server \
  --model-path $MODEL \
  --context-length $CTXLEN \
  --quantization moe_wna16 \
  --tool-call-parser qwen3_coder \
  --enable-cache-report \
  --port $PORT \
  --mem-fraction-static 0.85 \
  > "$LOG" 2>&1 &
disown   # otherwise the trap's kill prints a job-control "Killed" line

echo -n "waiting"
for _ in $(seq 1 300); do
  if grep -q "fired up and ready" "$LOG"; then echo " ready"; break; fi
  if ! pgrep -f "[s]glang\.launch_server" > /dev/null; then
    echo " DIED"; tail -30 "$LOG"; exit 1
  fi
  echo -n .; sleep 2
done

# An arm that silently ran without the rotation it is named for is worse than no arm
# at all -- and so is a control that silently ran with it.
if [ "$ROTATE" = "0" ]; then
  grep -q "Sub-context KV rotation ENABLED" "$LOG" \
    && { echo "REFUSING: the control arm has rotation enabled"; exit 1; }
  echo "control arm: rotation off"
else
  grep -q "Sub-context KV rotation ENABLED" "$LOG" \
    || { echo "REFUSING: rotation did not report itself enabled"; tail -30 "$LOG"; exit 1; }
  grep "Sub-context KV rotation ENABLED" "$LOG"
fi

python $REPO/scripts/subcontext_sim/probe_rotate_client.py "$TRACE" \
  --url "http://127.0.0.1:$PORT" --model "$MODEL" \
  $( [ "$ROTATE" = "0" ] && echo --control )

echo
echo "server log -> $LOG   (its subctx TRACE lines are buffered; trust $TRACE)"
