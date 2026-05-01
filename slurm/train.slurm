#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --job-name=baseline_train
#SBATCH --output=log/baseline_train_%j.out

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

python train.py data/wikitext-103 \
    --task language_modeling \
    --save-dir checkpoints/baseline/ \
    --arch transformer_lm_wiki103 \
    --max-update 140100 \
    --max-lr 1.0 \
    --t-mult 2 \
    --lr-period-updates 270000 \
    --lr-scheduler cosine \
    --lr-shrink 0.75 \
    --warmup-updates 16000 \
    --warmup-init-lr 1e-07 \
    --min-lr 1e-09 \
    --optimizer nag \
    --lr 0.0001 \
    --clip-norm 0.1 \
    --criterion adaptive_loss \
    --max-tokens 9216 \
    --update-freq 1 \
    --fp16 \
    --seed 1 \
    --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --tokens-per-sample 128 \
    --max-tokens-valid 128 \
    --tokens-from-prev 128 \
    --curriculum 1000 \
    --required-batch-size-multiple 1 \
    --save-interval 100 \
