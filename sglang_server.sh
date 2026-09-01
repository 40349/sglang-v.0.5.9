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
#   sbatch sglang_server.sh              split ON, no rotation  (the "on" arm)
#   ARM=off sbatch sglang_server.sh      split OFF              (the baseline arm)
#   ARM=rot sbatch sglang_server.sh      split ON + rotation    (the new arm)
#
# Diagnostic switches, both off by default and both of which make the run's timings
# unusable for the A/B -- they bill host work to one arm:
#   SUBCTX_TRACE=1  per-match/insert/finish tracing. Also sets PYTHONUNBUFFERED: the
#                   scheduler is a spawned child and does not inherit `python -u`, so
#                   its print() output is block-buffered and the last -- most
#                   interesting -- lines are lost when it dies.
#
# The pool-accounting audit (SUBCTX-IMBALANCE) needs neither: it is always on, costs
# three O(1) reads per finished request, and goes through the logger.
#   DUMP_TREE=1     print the whole radix tree after every extend pass
#
# The arm is an env var read once at server start, so it cannot be changed without a
# restart -- that is why each arm is a separate job.
#
# Read the compute node's address out of the job log and point the local
# run_mas.sh at it: a Slurm job lands on whichever node the scheduler picked, so the
# address is NOT stable across submissions.

set -euo pipefail

REPO=/home/m11402151/work/sglang-v.0.5.9
WORK_DIR=/home/m11402151/work
PORT=${PORT:-30000}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
CTXLEN=${CTXLEN:-16384}
ARM=${ARM:-on}

ml load miniconda3
eval "$(conda shell.bash hook)"
conda activate sglangv59

export PYTHONPATH=$REPO/python
export PYTHONNOUSERSITE=1

# Refuse rather than measure the wrong tree.
RESOLVED=$(python -c "import sglang, inspect; print(inspect.getfile(sglang))")
case "$RESOLVED" in
  "$REPO"/*) echo "sglang resolves to the fork: $RESOLVED" ;;
  *) echo "REFUSING: sglang resolves to $RESOLVED, not $REPO"; exit 1 ;;
esac

# Put the token in ~/.cache/huggingface/token (`huggingface-cli login`) instead of
# here -- a token in a script gets copied into logs, job output and version control.
export SGLANG_DISABLE_CUDNN_CHECK=1

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/traces"
SUF="${ARM}_${SLURM_JOB_ID:-manual}"
SERVER_LOG="$WORK_DIR/logs/sglang_${SUF}.log"

# Per-arm switches. Everything else about the binary is identical across arms.
case "$ARM" in
  on)  SUBCTX_OFF=""; ROT=""; ROT_ACROSS="" ;;
  off) SUBCTX_OFF="1"; ROT=""; ROT_ACROSS="" ;;
  rot) SUBCTX_OFF=""; ROT="1"; ROT_ACROSS="1" ;;
  *)   echo "REFUSING: unknown ARM='$ARM' (want on|off|rot)"; exit 1 ;;
esac

NODE_IP=$(hostname -I | awk '{print $1}')
cat <<EOF
==========================================
sglang server -- arm: $ARM
model:  $MODEL_PATH
node:   $(hostname)  ip: $NODE_IP
URL:    http://${NODE_IP}:${PORT}/v1
log:    $SERVER_LOG
trace:  $WORK_DIR/traces/*_${SUF}.*
==========================================
EOF

# --reasoning-parser is deliberately absent: Qwen3-Coder-30B-A3B-Instruct is a
# non-thinking model and emits no <think> block for it to strip.
# --enable-cache-report is what puts cached_tokens in the usage payload.
# No --quantization: on an H200 there is no VRAM reason to, and AWQ's dequant kernels
# sit inside the prefill time this experiment measures.
SGLANG_DISABLE_SUBCONTEXT=$SUBCTX_OFF \
SGLANG_SUBCONTEXT_ROTATE=$ROT \
SGLANG_SUBCONTEXT_ROTATE_ACROSS=$ROT_ACROSS \
SGLANG_SUBCTX_TRACE=${SUBCTX_TRACE:-} \
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
    --mem-fraction-static 0.9 \
    --enable-cache-report \
    --tool-call-parser qwen3_coder \
    > "$SERVER_LOG" 2>&1
