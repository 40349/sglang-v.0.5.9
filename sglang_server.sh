#!/bin/bash
#SBATCH --job-name=sglang_server
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sglang_%j.log
#SBATCH --error=sglang_%j.err
#
# sglang server for the sub-context experiments, one arm per start. This is the only
# place an arm becomes server settings; run_toggle.sh starts it once per arm as well.
#
#   ARM=cdc@0.15 sbatch sglang_server.sh          a job
#   ARM=cdc@0.15 bash sglang_server.sh            this box
#   ARM=cdc@0.15 bash sglang_server.sh check      validate, print the arm's file tag
#
# Arms (ARM is required):
#
#   off   no split -- the stock radix cache
#   cdc   cut each prompt by content, match its prefix, find its cached chunks past the
#         prefix in the index, rotate the displaced ones into place, prefill the gaps
#   relay cdc, with the reused tokens rectified the RelayCaching way (arXiv 2603.13289)
#
# cdc takes a recompute ratio after `@`: cdc@0.15 recomputes 15% of the tokens it
# reuses, chosen at layer TOPK_LAYER. Several ratios score consecutive layers from
# TOPK_LAYER on, each picking from what the previous one kept: cdc@0.3,0.2,0.15. The arm
# is the only place a ratio is set.
#
# relay takes its layers after `@`: relay@1,3,18 is L_start 1, L_det 3, L_end 18.
# relay@profile recomputes every reused token on every layer and writes per-layer
# deviations to RELAY_PROF; scripts/subcontext_sim/relay_layers.py picks the layers
# from that file.
#
# Knobs:
#
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct CTXLEN=81920 PORT=8000 MEMFRAC=0.90 BACKEND=triton
#   CHUNKED_PREFILL        unset keeps the server default; the same for every arm compared
#   KV_SPLITS=64           triton decode: most pieces one sequence's KV is split into;
#                          empty keeps the server default (8); the same for every arm compared
#   TOOL_PARSER=qwen3_coder REASONING_PARSER= QUANT=  empty drops the flag
#   Qwen3-30B-A3B:  MODEL=Qwen/Qwen3-30B-A3B CTXLEN=40960 TOOL_PARSER=qwen REASONING_PARSER=qwen3
#   TOPK_LAYER CDC_TARGET CDC_MIN CDC_MAX             unset keeps the server default
#   RELAY_TAU_DEV=1.5 RELAY_TAU_INF=1.45 RELAY_SUFFIX=10   relay's selection thresholds
#   AUDIT TRACE ROTATE_GPU DUMP_TREE                  diagnostics; timings then unusable
#   SERVER_LOG CAPTURE FWD_TRACE STAGE                output paths; empty turns one off
#   RELAY_PROF                                        relay@profile's output path
#
# By default every request is recorded to $WORK_DIR/traces/requests_<tag>_<job>.jsonl.

set -euo pipefail

# sbatch runs a spool copy of this file; the checkout is then where it was submitted.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[ -d "$REPO/python/sglang" ] || REPO=${SLURM_SUBMIT_DIR:-}
[ -d "$REPO/python/sglang" ] || { echo "REFUSING: no checkout found; submit from it"; exit 1; }
WORK_DIR=${WORK_DIR:-$(dirname "$REPO")}

ARM=${ARM:?set ARM=off, cdc, cdc@ratio, relay@L_start,L_det,L_end or relay@profile}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
CTXLEN=${CTXLEN:-81920}
PORT=${PORT:-8000}
MEMFRAC=${MEMFRAC:-0.90}
# Same backend for every arm; cdc needs triton (per-position mask).
BACKEND=${BACKEND:-triton}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-}
# No colon: an explicitly empty value drops the flag.
KV_SPLITS=${KV_SPLITS-64}
TOOL_PARSER=${TOOL_PARSER-qwen3_coder}
# Only for a model that closes </think>: on one that never does, the parser takes the
# whole reply, tool calls included, when the request carries no chat_template_kwargs.
REASONING_PARSER=${REASONING_PARSER-}
QUANT=${QUANT:-}
AUDIT=${AUDIT:-}
TRACE=${TRACE:-}

for v in "TOPK_RATIO|write the ratio into the arm, e.g. ARM=cdc@0.15" \
         "MODEL_PATH|use MODEL" "SUBCTX_AUDIT|use AUDIT" "SUBCTX_TRACE|use TRACE"; do
  if [ -n "$(printenv "${v%%|*}")" ]; then
    echo "REFUSING: ${v%%|*} is no longer read; ${v#*|}."; exit 1
  fi
done
for v in SGLANG_DISABLE_SUBCONTEXT SGLANG_SUBCTX_TOPK_RATIO SGLANG_SUBCTX_TOPK_LAYER \
         SGLANG_SUBCTX_CDC_TARGET SGLANG_SUBCTX_MIN_CHUNK SGLANG_SUBCTX_CDC_MAX \
         SGLANG_SUBCTX_RELAY_LAYERS SGLANG_SUBCTX_RELAY_PROFILE SGLANG_SUBCTX_RELAY_TAU_DEV \
         SGLANG_SUBCTX_RELAY_TAU_INF SGLANG_SUBCTX_RELAY_SUFFIX \
         SGLANG_SUBCTX_AUDIT SGLANG_SUBCTX_TRACE SGLANG_SUBCTX_ROTATE_GPU SGLANG_DUMP_TREE; do
  if [ -n "$(printenv "$v")" ]; then
    echo "REFUSING: $v is set in the environment; the arm and the knobs above set it."
    exit 1
  fi
done

NAME=${ARM%%@*}
RATIO=0
case "$ARM" in *@*) RATIO=${ARM#*@} ;; esac
case "$NAME" in
  off|cdc|relay) ;;
  *) echo "REFUSING: unknown ARM='$ARM'; want off, cdc, cdc@ratio or relay@..."; exit 1 ;;
esac
RELAY_LAYERS=
if [ "$NAME" = relay ]; then
  case "$RATIO" in
    profile) ;;
    *) awk -v ls="$RATIO" 'BEGIN { n = split(ls, l, ","); if (n != 3) exit 1
         for (i = 1; i <= 3; i++) if (l[i] !~ /^[0-9]+$/) exit 1
         exit !(l[1] < l[2] && l[2] <= l[3]) }' || {
         echo "REFUSING: ARM=$ARM: want relay@L_start,L_det,L_end with" \
              "L_start < L_det <= L_end, or relay@profile"; exit 1; }
       RELAY_LAYERS=$RATIO ;;
  esac
  RATIO=0
elif [ "$ARM" != "$NAME" ]; then
  [ "$NAME" = cdc ] || { echo "REFUSING: ARM=$ARM: only cdc takes a ratio"; exit 1; }
  awk -v rs="$RATIO" 'BEGIN { n = split(rs, r, ","); for (i = 1; i <= n; i++) {
      if (!(r[i] ~ /^[0-9]*\.?[0-9]+$/ && r[i] >= 0 && r[i] <= 1)) exit 1
      if (i > 1 && r[i] > r[i - 1]) exit 1 } }' || {
    echo "REFUSING: ARM=$ARM: want ratios in [0, 1], comma-separated, not increasing"
    exit 1; }
fi
TAG=$NAME
if awk -v rs="$RATIO" 'BEGIN { split(rs, r, ","); exit !(r[1] > 0) }'; then
  TAG=${NAME}_r$(awk -v rs="$RATIO" 'BEGIN { n = split(rs, r, ",")
    for (i = 1; i <= n; i++) printf "%s%02d", (i > 1 ? "-" : ""), int(r[i] * 100 + 0.5) }')
fi
if [ "$NAME" = relay ]; then
  TAG=relay_$([ -n "$RELAY_LAYERS" ] && echo "L${RELAY_LAYERS//,/-}" || echo profile)
  # Thresholds set in the environment show in the tag.
  TAG=$TAG${RELAY_TAU_DEV:+_dev$RELAY_TAU_DEV}${RELAY_TAU_INF:+_inf$RELAY_TAU_INF}
  TAG=$TAG${RELAY_SUFFIX:+_suf$RELAY_SUFFIX}
fi

if [ "${1:-}" = check ]; then echo "$TAG"; exit 0; fi

if command -v ml > /dev/null 2>&1; then ml load miniconda3; fi
# conda's shell hook and activate/deactivate scripts run with nounset off
set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV:-sglangv59}"
set -u

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
RELAY_PROF=${RELAY_PROF:-$WORK_DIR/traces/relay_profile_${SUF}.jsonl}
{ [ "$NAME" = relay ] && [ -z "$RELAY_LAYERS" ]; } || RELAY_PROF=
for p in "$SERVER_LOG" "$CAPTURE" "$FWD_TRACE" "$STAGE" "$RELAY_PROF"; do
  [ -z "$p" ] || mkdir -p "$(dirname "$p")"
done

# Refuse if anything holds this port, a draining server's 503 included. curl exit 7:
# nothing listening.
rc=0
curl -s -o /dev/null --max-time 5 "http://127.0.0.1:${PORT}/health" || rc=$?
if [ "$rc" != 7 ]; then
  echo "REFUSING: something already holds 127.0.0.1:${PORT} on $(hostname)."
  echo "  Another sglang is still up here: another job's, or one draining after a"
  echo "  SIGTERM (it answers 503 and never finishes). Its server would take this run's"
  echo "  traffic. Check with: squeue -u \$USER, then scancel the old job, or pick"
  echo "  another PORT= to run beside it."
  exit 1
fi

export SGLANG_DISABLE_SUBCONTEXT= SGLANG_SUBCTX_TOPK_RATIO=0 \
  SGLANG_SUBCTX_RELAY_LAYERS= SGLANG_SUBCTX_RELAY_PROFILE=
case "$NAME" in
  off) SGLANG_DISABLE_SUBCONTEXT=1 ;;
  cdc) SGLANG_SUBCTX_TOPK_RATIO=$RATIO ;;
  relay) SGLANG_SUBCTX_RELAY_LAYERS=$RELAY_LAYERS SGLANG_SUBCTX_RELAY_PROFILE=$RELAY_PROF ;;
esac
[ -z "${RELAY_TAU_DEV:-}" ] || export SGLANG_SUBCTX_RELAY_TAU_DEV=$RELAY_TAU_DEV
[ -z "${RELAY_TAU_INF:-}" ] || export SGLANG_SUBCTX_RELAY_TAU_INF=$RELAY_TAU_INF
[ -z "${RELAY_SUFFIX:-}" ] || export SGLANG_SUBCTX_RELAY_SUFFIX=$RELAY_SUFFIX
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
backend:$BACKEND   audit: ${AUDIT:-off}   ctxlen: $CTXLEN   chunked prefill: ${CHUNKED_PREFILL:-default}   kv splits: ${KV_SPLITS:-default}
node:   $(hostname)  ip: $NODE_IP
URL:    http://$([ "$HOST" = 0.0.0.0 ] && echo "$NODE_IP" || echo "$HOST"):${PORT}/v1
log:    $SERVER_LOG
capture:${CAPTURE:- off}$([ "$NAME" = relay ] && echo "
relay:  tau_dev ${RELAY_TAU_DEV:-1.5}   tau_inf ${RELAY_TAU_INF:-1.45}   suffix ${RELAY_SUFFIX:-10}")${RELAY_PROF:+
relay profile: $RELAY_PROF}
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
    ${KV_SPLITS:+--triton-attention-num-kv-splits "$KV_SPLITS"} \
    ${TOOL_PARSER:+--tool-call-parser "$TOOL_PARSER"} \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    ${QUANT:+--quantization "$QUANT"} \
    > "$SERVER_LOG" 2>&1
