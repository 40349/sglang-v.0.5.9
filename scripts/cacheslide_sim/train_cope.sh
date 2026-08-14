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
MODEL="meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET="Crystalcareai/Code-feedback-sharegpt-renamed"

WORK_DIR=${REPO_DIR}/scripts/cope
JOB=${SLURM_JOB_ID:-manual}
# Per-job output dir. The old script wrote every run into the same cope_adapter/,
# so each launch silently overwrote the previous cope_adapter_best.pt.
OUT_DIR=${WORK_DIR}/cope_adapter/run_${JOB}
LOG_DIR=${WORK_DIR}/logs
TRAIN_LOG="${LOG_DIR}/train_cope_${JOB}.log"

# Rows [0, TRAIN_ROWS) are used for training (train_cope holds 200 of them out for its
# own eval). Everything past TRAIN_ROWS stays untouched so verify_cope.py --skip_samples
# has genuinely unseen data. The full split is 66383 rows and 2000 steps x 16 only
# consumes 32000 samples, so reserving the tail costs nothing.
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

# ============================================================
# 訓練
# ============================================================
# --pos_emb_lr is now ABOVE --lr, not 1/5 of it. Measured on the previous run's adapter:
# the trained pos_emb columns have norm ~0.0067, while a content key contributes
# ||k||*scaling with ||k|| = O(1-10) -- the position term came out 13-132x too weak to
# move any attention weight, so that run's ppl 470->2.78 was the LoRA alone and CoPE was
# inert. Watch the `pos/attn` number printed at every eval; if it stays under 0.05,
# raise --pos_emb_lr again rather than letting the job run 24h.
#
# --gate_bias_span 256 keeps the initial contextual span inside npos_max=1024. Unbiased
# gates start at seq_len/2 = 2048 positions, which clamps, and clamp has zero gradient.
#
# --gate_reg is OFF for this run: fix the position signal first and confirm pos/attn
# climbs, then re-enable the span cap on top of a model that is actually using CoPE.
srun python -u scripts/cope/train_cope.py \
  --model "${MODEL}" \
  --tokenizer "${MODEL}" \
  --hf_dataset "${HF_DATASET}" \
  --max_samples ${TRAIN_ROWS} \
  --max_seq_len 4096 --npos_max 1024 --tile_q 1024 \
  --lr 1e-4 --pos_emb_lr 1e-3 \
  --gate_bias_span 256 \
  --gate_reg 0 --gate_span_target 512 --gate_bimod 0 \
  --bf16 --grad_ckpt --batch_size 1 --grad_accum 16 \
  --seed 0 --steps 2000 --eval_every 50 --save_every 200 \
  --output_dir "${OUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"

echo "done. adapter -> ${OUT_DIR}/cope_adapter_best.pt"
echo "if the job was killed, restart it with:"
echo "  ... --resume ${OUT_DIR}/cope_adapter_last.pt"
echo
echo "next, on the same node:"
echo "  python scripts/cope/verify_cope.py \\"
echo "    --model ${MODEL} --tokenizer ${MODEL} \\"
echo "    --adapter ${OUT_DIR}/cope_adapter_best.pt \\"
echo "    --hf_dataset ${HF_DATASET} --skip_samples ${TRAIN_ROWS} \\"
echo "    --max_seq_len 1024 --eval_samples 100 --bf16"
