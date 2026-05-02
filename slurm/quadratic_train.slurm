#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=4
#SBATCH --time=40:00:00
#SBATCH --job-name=quadratic
#SBATCH --output=log/quadratic/%j.out

# Load modules and activate environment
conda deactivate || true
module purge
module reset
module load pytorch-conda
conda activate shortformer
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1
module unload cudatoolkit
module load cuda/11.8

BASE_SAVE_DIR="checkpoints/quadratic"
SAVE_DIR="${BASE_SAVE_DIR}_${SLURM_JOB_ID:-local}"

# If the directory exists, append a timestamp to make it unique
if [ -d "$SAVE_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SAVE_DIR="${BASE_SAVE_DIR}_${TIMESTAMP}"
fi
echo "Results will be saved to: $SAVE_DIR"

python train.py data/wikitext-103 \
    --task language_modeling \
    --save-dir $SAVE_DIR \
    --arch transformer_lm_wiki103 \
    --max-update 26433 \
    --max-lr 1.0 \
    --t-mult 2 \
    --lr-period-updates 50943 \
    --lr-scheduler cosine \
    --lr-shrink 0.75 \
    --warmup-updates 3018 \
    --warmup-init-lr 1e-07 \
    --min-lr 1e-09 \
    --optimizer nag \
    --lr 0.0001 \
    --clip-norm 0.1 \
    --criterion adaptive_loss \
    --max-tokens 12288 \
    --update-freq 1 \
    --seed 1 \
    --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --tokens-per-sample 128 \
    --max-tokens-valid 128 \
    --tokens-from-prev 128 \
    --curriculum 1000 \
    --required-batch-size-multiple 1 \
    --temp-degree 2 \
    --temp-pieces 1 \
    --save-interval-updates 1000 \
    --log-interval 50 \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \


