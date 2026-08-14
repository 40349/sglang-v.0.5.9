#!/bin/bash
#SBATCH --job-name=train_cope
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cope_%j.log
#SBATCH --error=cope_%j.err

set -euo pipefail

REPO_DIR="/home/m11402151/work/sglang-v.0.5.9"
# Dense Qwen3, NOT the MoE or Next variants:
#   Qwen3-8B          -> Qwen3ForCausalLM      standard radix cache, CoPE train+serve OK
#   Qwen3-Coder-30B   -> Qwen3MoeForCausalLM   standard radix, CoPE SERVING not ported yet
#   Qwen3-Coder-Next  -> Qwen3NextForCausalLM  hybrid SSM cache -- avoid
# Must be unquantized: LoRA needs real nn.Linear to attach to and merge back into.
MODEL="Qwen/Qwen3-8B"
HF_DATASET="Crystalcareai/Code-feedback-sharegpt-renamed"

WORK_DIR=${REPO_DIR}/scripts/cope
JOB=${SLURM_JOB_ID:-manual}
OUT_DIR=${WORK_DIR}/cope_adapter/qwen3_8b_${JOB}
LOG_DIR=${WORK_DIR}/logs
TRAIN_LOG="${LOG_DIR}/train_cope_${JOB}.log"

# Fresh run: the Llama adapters are a different architecture (32 vs 36 layers, QK-Norm,
# different shapes) and cannot be resumed from. Set this only to continue a Qwen3 run,
# and point it at that run's cope_adapter_last.pt -- best.pt carries no optimizer state.
RESUME_FROM=""

# Back to 1e-3, with gate_bias FROZEN -- a combination that has never actually run.
# History: 6e-6 gave pos/attn 0.005 (inert, ppl fine); 1e-3 with a TRAINABLE gate_bias
# gave pos/attn 0.148 (CoPE load-bearing) but oscillating ppl; 2e-4 with gate_bias frozen
# gave clean convergence (Qwen3-8B, ppl 3.18 = 0.87x the RoPE baseline) but pos/attn stuck
# at 0.017 -- inert again, the LoRA doing all the work. The oscillation at 1e-3 was
# attributed to gate_bias chasing pos_emb, but that fix and the LR cut landed together, so
# this run isolates the one variable that was changed for the wrong reason.
#
# Watch pos/attn at the step-149 eval: still under ~0.03 there and the problem is not the
# learning rate, it is that next-token prediction on this corpus does not need position at
# all -- a position-free model already beats the RoPE baseline on it. The fallback is then
# to freeze the LoRA for the first N steps so pos_emb is the only thing that can reduce
# the loss.
POS_EMB_LR=1e-3

# Rows [0, TRAIN_ROWS) train; the tail stays unseen for verify_cope --skip_samples.
TRAIN_ROWS=60000

# ============================================================
# 環境
# ============================================================
ml load miniconda3
eval "$(conda shell.bash hook)"
conda activate sglangv59

# torch 2.9 only reads PYTORCH_CUDA_ALLOC_CONF at runtime; PYTORCH_ALLOC_CONF is the
# documented future name but is not wired up yet, so setting only that is a no-op.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "ERROR: RESUME_FROM does not exist: ${RESUME_FROM}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${RESUME_FROM}")
  echo "resuming from ${RESUME_FROM} with pos_emb_lr=${POS_EMB_LR}"
fi

# ============================================================
# 訓練
# ============================================================
# --gate_bias_span 256 is calibrated against the MEDIAN rendered conversation, which the
# script measures and prints ("sample length: median ...") -- Code-Feedback sits around
# 1150 tokens, so an unbiased sigmoid would start the span near 575 and overrun
# npos_max=1024 on the longer rows. --max_seq_len 4096 is only a truncation cap here
# (1 row in 200 hits it) and costs nothing on short samples at batch_size 1.
#
# --gate_reg is OFF: get the position signal converging first, then re-enable the span
# cap on top of a model that is actually using CoPE.
srun python -u scripts/cope/train_cope.py \
  --model "${MODEL}" \
  --tokenizer "${MODEL}" \
  --hf_dataset "${HF_DATASET}" \
  --max_samples ${TRAIN_ROWS} \
  --max_seq_len 4096 --npos_max 1024 --tile_q 1024 \
  --lr 1e-4 --pos_emb_lr "${POS_EMB_LR}" \
  --gate_bias_span 256 \
  --gate_reg 0 --gate_span_target 512 --gate_bimod 0 \
  --bf16 --grad_ckpt --batch_size 1 --grad_accum 16 \
  --seed 0 --steps 2000 --eval_every 50 --save_every 200 \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --output_dir "${OUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"

echo "done. adapter -> ${OUT_DIR}/cope_adapter_best.pt"
echo "if the job was killed, set RESUME_FROM to:"
echo "  ${OUT_DIR}/cope_adapter_last.pt"
echo
echo "next, on the same node:"
echo "  python scripts/cope/verify_cope.py \\"
echo "    --model ${MODEL} --tokenizer ${MODEL} \\"
echo "    --adapter ${OUT_DIR}/cope_adapter_best.pt \\"
echo "    --hf_dataset ${HF_DATASET} --skip_samples ${TRAIN_ROWS} \\"
echo "    --max_seq_len 1024 --eval_samples 100 --bf16"
