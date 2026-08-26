#!/bin/bash
# Diagnostic replay: why does the contiguity rule discard so much?
#
# On av_he the split matched 265,398 tokens in the tree and could only spend
# 227,812 of them -- 37,586 thrown away, more than twice the 18,271 it actually
# gained. With a [system][messages] split the rule discards a segment's hit
# only when an EARLIER segment failed to hit in full, so every one of those
# tokens is a messages-block hit gated by the much smaller system block. There
# are only four distinct system prompts in this workload, so after the first
# occurrence each should stay in the tree -- unless it is being evicted, or
# never inserted properly. Those two have very different fixes.
#
# SUBCTX_TRACE is deliberately ON here even though it bills its own debug output
# to the mechanism: this run is for the per-namespace hit lengths, not timings.
# Do not read GPU or host numbers out of it.
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=$REPO/ab_out
TAG=${TAG:-av_he}
REQUESTS=${REQUESTS:-$OUT/requests_${TAG}_on.jsonl}
LOG=$OUT/probe_${TAG}.log
PORT=${PORT:-30000}
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}

source /home/t2503-3090/miniconda3/etc/profile.d/conda.sh
pkill -f "sglang\.launch_server" 2>/dev/null || true
sleep 6
conda activate sglangv59
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/python"

SGLANG_SUBCTX_TRACE=1 \
nohup python -u -m sglang.launch_server \
  --model-path "$MODEL" --context-length 16384 --quantization moe_wna16 \
  --tool-call-parser qwen3_coder --enable-cache-report \
  --port "$PORT" --mem-fraction-static 0.85 > "$LOG" 2>&1 &

echo -n "waiting for server"
for _ in $(seq 1 300); do
  grep -q "fired up and ready" "$LOG" && { echo " ready"; break; }
  pgrep -f "sglang\.launch_server" > /dev/null || { echo " DIED"; tail -20 "$LOG"; exit 1; }
  echo -n .; sleep 2
done
grep -o "max_total_num_tokens=[0-9]*" "$LOG" | head -1

python "$REPO/subcontext_bench.py" replay "$REQUESTS" \
  --url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --gen-tokens 32 --out "$OUT/client_probe_${TAG}.json" > /dev/null

pkill -TERM -f "sglang\.launch_server" 2>/dev/null || true
sleep 8
echo "trace written to $LOG"
grep -c "sub-context match" "$LOG" || true
