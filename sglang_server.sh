#!/bin/bash
#SBATCH --job-name=sglang_server
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=sglang_%j.log
#SBATCH --error=sglang_%j.err
#
# sglang server for the sub-context experiments, one arm per start. This is the only
# place an arm becomes server settings; run_swe.sh starts it once per arm as well.
#
#   ARM=cdc@0.15 CTXLEN=40960 sbatch sglang_server.sh   a job
#   ARM=cdc@0.15 CTXLEN=40960 bash sglang_server.sh     this box
#   ARM=cdc@0.15 bash sglang_server.sh check      validate, print the arm's file tag
#
# Arms:
#
#   off   no split -- the stock single-namespace radix cache
#   on    split into per-block namespaces, displaced hits dropped
#   rot   + rotate a displaced hit to where it is reused, + Stage 2
#   idx   + find blocks by content anywhere in the prompt, and prefill the gaps
#   cdc   + cut the blocks on content too, not on the roles the prompt was built from
#
# idx and cdc take a recompute ratio after `@`: cdc@0.15 recomputes 15% of the tokens
# it reuses. The arm is the only place a ratio is set.
#
# Knobs:
#
#   MODEL=Qwen/Qwen3-30B-A3B CTXLEN=32768 PORT=30000 MEMFRAC=0.90 BACKEND=triton
#   CHUNKED_PREFILL        unset keeps the server default; the same for every arm compared
#   TOOL_PARSER=qwen REASONING_PARSER=qwen3 QUANT=    empty drops the flag
#   TOPK_LAYER CDC_TARGET CDC_MIN CDC_MAX             unset keeps the server default
#   AUDIT TRACE ROTATE_GPU DUMP_TREE                  diagnostics; timings then unusable
#   SERVER_LOG CAPTURE FWD_TRACE STAGE                output paths; empty turns one off
#
# By default every request is recorded to $WORK_DIR/traces/requests_<tag>_<job>.jsonl.

set -euo pipefail

# sbatch runs a spool copy of this file; the checkout is then where it was submitted.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[ -d "$REPO/python/sglang" ] || REPO=${SLURM_SUBMIT_DIR:-}
[ -d "$REPO/python/sglang" ] || { echo "REFUSING: no checkout found; submit from it"; exit 1; }
WORK_DIR=${WORK_DIR:-$(dirname "$REPO")}

ARM=${ARM:-on}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
CTXLEN=${CTXLEN:-32768}
PORT=${PORT:-30000}
MEMFRAC=${MEMFRAC:-0.90}
# Same backend for every arm; idx/cdc need triton (per-position mask).
BACKEND=${BACKEND:-triton}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-}
# No colon: an explicitly empty value drops the flag.
TOOL_PARSER=${TOOL_PARSER-qwen}
REASONING_PARSER=${REASONING_PARSER-qwen3}
QUANT=${QUANT:-}
AUDIT=${AUDIT:-}
TRACE=${TRACE:-}

for v in "TOPK_RATIO|write the ratio into the arm, e.g. ARM=cdc@0.15" \
         "MODEL_PATH|use MODEL" "SUBCTX_AUDIT|use AUDIT" "SUBCTX_TRACE|use TRACE"; do
  if [ -n "$(printenv "${v%%|*}")" ]; then
    echo "REFUSING: ${v%%|*} is no longer read; ${v#*|}."; exit 1
  fi
done
for v in SGLANG_DISABLE_SUBCONTEXT SGLANG_SUBCONTEXT_ROTATE SGLANG_SUBCONTEXT_ROTATE_ACROSS \
         SGLANG_SUBCTX_INDEX SGLANG_SUBCTX_SPLIT SGLANG_SUBCTX_TOPK_RATIO \
         SGLANG_SUBCTX_TOPK_LAYER SGLANG_SUBCTX_CDC_TARGET SGLANG_SUBCTX_MIN_CHUNK \
         SGLANG_SUBCTX_CDC_MAX SGLANG_SUBCTX_AUDIT SGLANG_SUBCTX_TRACE \
         SGLANG_SUBCTX_ROTATE_GPU SGLANG_DUMP_TREE; do
  if [ -n "$(printenv "$v")" ]; then
    echo "REFUSING: $v is set in the environment; the arm and the knobs above set it."
    exit 1
  fi
done

NAME=${ARM%%@*}
RATIO=0
case "$ARM" in *@*) RATIO=${ARM#*@} ;; esac
case "$NAME" in
  off|on|rot|idx|cdc) ;;
  *) echo "REFUSING: unknown ARM='$ARM'; want off|on|rot|idx|cdc, idx/cdc@ratio"; exit 1 ;;
esac
if [ "$ARM" != "$NAME" ]; then
  case "$NAME" in
    idx|cdc) ;;
    *) echo "REFUSING: ARM=$ARM: only idx and cdc take a ratio"; exit 1 ;;
  esac
  awk -v r="$RATIO" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+$/ && r >= 0 && r <= 1) }' || {
    echo "REFUSING: ARM=$ARM: the ratio must be a number in [0, 1]"; exit 1; }
fi
TAG=$NAME
if awk -v r="$RATIO" 'BEGIN { exit !(r > 0) }'; then
  TAG=${NAME}_r$(awk -v r="$RATIO" 'BEGIN { printf "%02d", int(r * 100 + 0.5) }')
fi

if [ "${1:-}" = check ]; then echo "$TAG"; exit 0; fi

if command -v ml > /dev/null 2>&1; then ml load miniconda3; fi
eval "$(conda shell.bash hook)"
conda activate "${ENV:-sglangv59}"

export PYTHONPATH=$REPO/python
export PYTHONNOUSERSITE=1

# Refuse unless sglang resolves to this checkout.
RESOLVED=$(python -c "import sglang, inspect; print(inspect.getfile(sglang))")
case "$RESOLVED" in
  "$REPO"/*) echo "sglang resolves to the fork: $RESOLVED" ;;
  *) echo "REFUSING: sglang resolves to $RESOLVED, not $REPO"; exit 1 ;;
esac

# Token goes in ~/.cache/huggingface/token, not here.
export SGLANG_DISABLE_CUDNN_CHECK=1

SUF="${TAG}_${SLURM_JOB_ID:-manual}"
SERVER_LOG=${SERVER_LOG:-$WORK_DIR/logs/sglang_${SUF}.log}
CAPTURE=${CAPTURE-$WORK_DIR/traces/requests_${SUF}.jsonl}
FWD_TRACE=${FWD_TRACE-$WORK_DIR/traces/trace_${SUF}.jsonl}
STAGE=${STAGE-$WORK_DIR/traces/stage_${SUF}}
for p in "$SERVER_LOG" "$CAPTURE" "$FWD_TRACE" "$STAGE"; do
  [ -z "$p" ] || mkdir -p "$(dirname "$p")"
done

# Refuse if something already serves this port.
if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
  echo "REFUSING: something is already serving 127.0.0.1:${PORT} on $(hostname)."
  echo "  A previous sglang job is still up. Its server would take this run's traffic"
  echo "  and this job would hold a GPU for nothing. Check with: squeue -u \$USER"
  echo "  then scancel the old job before resubmitting."
  exit 1
fi

export SGLANG_DISABLE_SUBCONTEXT= SGLANG_SUBCONTEXT_ROTATE= SGLANG_SUBCONTEXT_ROTATE_ACROSS= \
  SGLANG_SUBCTX_INDEX= SGLANG_SUBCTX_SPLIT=blocks SGLANG_SUBCTX_TOPK_RATIO=0
case "$NAME" in
  off) SGLANG_DISABLE_SUBCONTEXT=1 ;;
  on)  ;;
  rot) SGLANG_SUBCONTEXT_ROTATE=1 SGLANG_SUBCONTEXT_ROTATE_ACROSS=1 ;;
  idx) SGLANG_SUBCONTEXT_ROTATE=1 SGLANG_SUBCTX_INDEX=1 SGLANG_SUBCTX_TOPK_RATIO=$RATIO ;;
  cdc) SGLANG_SUBCONTEXT_ROTATE=1 SGLANG_SUBCTX_INDEX=1 SGLANG_SUBCTX_SPLIT=cdc \
         SGLANG_SUBCTX_TOPK_RATIO=$RATIO ;;
esac
export SGLANG_SUBCTX_AUDIT=$AUDIT SGLANG_SUBCTX_TRACE=$TRACE \
  SGLANG_SUBCTX_ROTATE_GPU=${ROTATE_GPU:-} SGLANG_DUMP_TREE=${DUMP_TREE:-}
[ -z "${TOPK_LAYER:-}" ] || export SGLANG_SUBCTX_TOPK_LAYER=$TOPK_LAYER
[ -z "${CDC_TARGET:-}" ] || export SGLANG_SUBCTX_CDC_TARGET=$CDC_TARGET
[ -z "${CDC_MIN:-}" ] || export SGLANG_SUBCTX_MIN_CHUNK=$CDC_MIN
[ -z "${CDC_MAX:-}" ] || export SGLANG_SUBCTX_CDC_MAX=$CDC_MAX

# A job serves other machines; on this box only this box needs it. Not read from the
# environment: conda activation exports HOST as the compiler triplet.
HOST=$([ -n "${SLURM_JOB_ID:-}" ] && echo 0.0.0.0 || echo 127.0.0.1)
NODE_IP=$(hostname -I | awk '{print $1}')

cat <<EOF
==========================================
sglang server -- arm: $ARM
model:  $MODEL
backend:$BACKEND   audit: ${AUDIT:-off}   ctxlen: $CTXLEN   chunked prefill: ${CHUNKED_PREFILL:-default}
node:   $(hostname)  ip: $NODE_IP
URL:    http://$([ "$HOST" = 0.0.0.0 ] && echo "$NODE_IP" || echo "$HOST"):${PORT}/v1
log:    $SERVER_LOG
capture:${CAPTURE:- off}
==========================================
EOF

# --enable-cache-report puts cached_tokens in the usage payload.
PYTHONUNBUFFERED=${TRACE:+1} \
SGLANG_CAPTURE_REQUESTS=$CAPTURE \
SGLANG_FORWARD_TRACE=$FWD_TRACE \
SGLANG_STAGE_TRACE=$STAGE \
python -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 1 \
    --context-length "$CTXLEN" \
    ${CHUNKED_PREFILL:+--chunked-prefill-size "$CHUNKED_PREFILL"} \
    --mem-fraction-static "$MEMFRAC" \
    --enable-cache-report \
    --attention-backend "$BACKEND" \
    ${TOOL_PARSER:+--tool-call-parser "$TOOL_PARSER"} \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${QUANT:+--quantization "$QUANT"} \
    > "$SERVER_LOG" 2>&1
