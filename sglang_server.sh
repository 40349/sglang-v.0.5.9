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
# sglang server for the sub-context A/B, run on the H200 while MASLab stays local.
#
#   ARM=off   no split -- the stock radix cache, the baseline
#   ARM=on    split into per-block namespaces (the default)
#   ARM=rot   + rotate a displaced block into place, + Stage 2
#   ARM=idx   + find blocks by content anywhere in the prompt, prefill the gaps
#   ARM=cdc   + cut the blocks on content too, not on the roles the prompt was built from
#
# The arm is read once at start: one job per arm. The compute node's address is in
# the job log.
#
# Diagnostics, off by default, each one makes the timings unusable for the A/B:
# SUBCTX_TRACE=1 (per-match tracing), SUBCTX_AUDIT=1 (leak / ownership check on the
# finish path), DUMP_TREE=1, ROTATE_GPU=1 (CUDA events around the rotation).

set -euo pipefail

REPO=/home/m11402151/work/sglang-v.0.5.9
WORK_DIR=/home/m11402151/work
PORT=${PORT:-30000}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
CTXLEN=${CTXLEN:-16384}
ARM=${ARM:-on}

# Pinned for every arm: triton is the only backend that takes a per-position mask.
BACKEND=${BACKEND:-triton}

# run_swe.sh spells these AUDIT and TRACE; accept both.
SUBCTX_AUDIT=${SUBCTX_AUDIT:-${AUDIT:-}}
SUBCTX_TRACE=${SUBCTX_TRACE:-${TRACE:-}}
SPLIT=blocks

# Fraction of the reused tokens that get recomputed anyway. A dial, not part of the
# arm name: 0 is the arm as it stands, and the interesting runs sweep it.
TOPK_RATIO=${TOPK_RATIO:-0}
TOPK_LAYER=${TOPK_LAYER:-1}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-}

ml load miniconda3
eval "$(conda shell.bash hook)"
conda activate sglangv59

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

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/traces"
SUF="${ARM}_${SLURM_JOB_ID:-manual}"
SERVER_LOG="$WORK_DIR/logs/sglang_${SUF}.log"

# Per-arm switches; everything else about the binary is identical across arms.
case "$ARM" in
  on)  SUBCTX_OFF=""; ROT=""; ROT_ACROSS=""; INDEX="" ;;
  off) SUBCTX_OFF="1"; ROT=""; ROT_ACROSS=""; INDEX="" ;;
  rot) SUBCTX_OFF=""; ROT="1"; ROT_ACROSS="1"; INDEX="" ;;
  idx) SUBCTX_OFF=""; ROT="1"; ROT_ACROSS=""; INDEX="1" ;;
  cdc) SUBCTX_OFF=""; ROT="1"; ROT_ACROSS=""; INDEX="1"; SPLIT="cdc" ;;
  *)   echo "REFUSING: unknown ARM='$ARM' (want on|off|rot|idx|cdc)"; exit 1 ;;
esac

# Scoring a reused token needs a key computed for it, so the first layers run the whole
# prompt -- and a prompt longer than the chunked-prefill budget takes the stitch path
# instead, which is exactly the long, heavily-reused prompt this is for. It would still
# serve, and the arm would quietly be measuring something else, so refuse instead of
# picking a budget: changing it changes every arm's chunking and the caller has to
# decide that for the whole comparison, not for one run.
if [ "$TOPK_RATIO" != "0" ] && [ -z "$CHUNKED_PREFILL" ]; then
  echo "REFUSING: TOPK_RATIO=$TOPK_RATIO needs CHUNKED_PREFILL set to at least the"
  echo "  longest prompt you will send (the probe pass runs full length, so anything"
  echo "  above the budget silently falls back to the stitch and reuses less)."
  echo "  e.g. CHUNKED_PREFILL=$CTXLEN -- and set it for EVERY arm in the comparison,"
  echo "  or the arms differ in their chunking as well as in the thing under test."
  exit 1
fi

NODE_IP=$(hostname -I | awk '{print $1}')

# Refuse if something already serves this port.
if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
  echo "REFUSING: something is already serving 127.0.0.1:${PORT} on $(hostname)."
  echo "  A previous sglang job is still up. Its server would take this run's traffic"
  echo "  and this job would hold a GPU for nothing. Check with: squeue -u \$USER"
  echo "  then scancel the old job before resubmitting."
  exit 1
fi

cat <<EOF
==========================================
sglang server -- arm: $ARM
model:  $MODEL_PATH
backend:$BACKEND
split:  $SPLIT   audit: ${SUBCTX_AUDIT:-off}   ctxlen: $CTXLEN
topk:   ratio $TOPK_RATIO at layer $TOPK_LAYER   chunked prefill: ${CHUNKED_PREFILL:-default}
node:   $(hostname)  ip: $NODE_IP
URL:    http://${NODE_IP}:${PORT}/v1
log:    $SERVER_LOG
trace:  $WORK_DIR/traces/*_${SUF}.*
==========================================
EOF

# --reasoning-parser qwen3 keeps the <think> block out of the scored content.
# --enable-cache-report puts cached_tokens in the usage payload.
SGLANG_DISABLE_SUBCONTEXT=$SUBCTX_OFF \
SGLANG_SUBCONTEXT_ROTATE=$ROT \
SGLANG_SUBCONTEXT_ROTATE_ACROSS=$ROT_ACROSS \
SGLANG_SUBCTX_INDEX=$INDEX \
SGLANG_SUBCTX_SPLIT=$SPLIT \
SGLANG_SUBCTX_TOPK_RATIO=${TOPK_RATIO:-0} \
SGLANG_SUBCTX_TOPK_LAYER=${TOPK_LAYER:-1} \
SGLANG_SUBCTX_TRACE=$SUBCTX_TRACE \
SGLANG_SUBCTX_ROTATE_GPU=${ROTATE_GPU:-} \
SGLANG_SUBCTX_AUDIT=$SUBCTX_AUDIT \
PYTHONUNBUFFERED=${SUBCTX_TRACE:+1} \
SGLANG_DUMP_TREE=${DUMP_TREE:-} \
SGLANG_CAPTURE_REQUESTS=$WORK_DIR/traces/requests_${SUF}.jsonl \
SGLANG_FORWARD_TRACE=$WORK_DIR/traces/trace_${SUF}.jsonl \
SGLANG_STAGE_TRACE=$WORK_DIR/traces/stage_${SUF} \
python -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size 1 \
    --context-length "$CTXLEN" \
    ${CHUNKED_PREFILL:+--chunked-prefill-size "$CHUNKED_PREFILL"} \
    --mem-fraction-static 0.9 \
    --enable-cache-report \
    --attention-backend "$BACKEND" \
    --tool-call-parser qwen \
    --reasoning-parser qwen3 \
    > "$SERVER_LOG" 2>&1
