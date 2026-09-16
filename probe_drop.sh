#!/bin/bash
# Diagnostic replay: per-namespace hit lengths, to account for what the contiguity
# rule discards. On av_he the split matched 265,398 tokens and spent 227,812.
#
# SUBCTX_TRACE is ON. Do not read GPU or host numbers out of this run.
set -euo pipefail

REPO=/home/t2503-3090/Desktop/MiaoChen/sglang-v.0.5.9
OUT=$REPO/ab_out
TAG=${TAG:-av_he}
REQUESTS=${REQUESTS:-$OUT/requests_${TAG}_on.jsonl}
LOG=$OUT/probe_${TAG}.log
PORT=${PORT:-30000}
MODEL=${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}

source /home/t2503-3090/miniconda3/etc/profile.d/conda.sh
# `[s]` so this script's own command line cannot match; `sglang::` is the scheduler,
# which renames itself and holds the pool.
pkill -TERM -f '[s]glang::|[s]glang\.launch_server' 2>/dev/null || true
sleep 10
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

pkill -TERM -f '[s]glang::|[s]glang\.launch_server' 2>/dev/null || true
sleep 10
echo "trace written to $LOG"
grep -c "sub-context match" "$LOG" || true
