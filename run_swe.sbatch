#!/bin/bash
#SBATCH --job-name=swe_toggle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=swe_toggle_%j.log
#
# run_swe.sh as a batch job. Submit from the checkout; the caller's environment is
# passed through:
#
#   REQUESTS=... ARMS="off cdc cdc@0.15" CTXLEN=40960 CHUNKED_PREFILL=40960 \
#     sbatch run_swe.sbatch toggle
#
# sbatch runs a spool copy of this file, so run_swe.sh is called by its path in the
# submit directory.
cd "$SLURM_SUBMIT_DIR"
[ -f run_swe.sh ] || { echo "REFUSING: no run_swe.sh in $SLURM_SUBMIT_DIR; submit from the checkout"; exit 1; }
exec bash ./run_swe.sh "${@:-toggle}"
