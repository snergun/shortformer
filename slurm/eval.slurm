#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=0:20:00
#SBATCH --job-name=eval_sf
#SBATCH --output=log/eval_%j.out

# Load modules and activate environment
conda deactivate || true
module purge
module reset
module load pytorch-conda
module purge cudatoolkit
module load cuda/11.8
conda activate shortformer
# wikitext-103
bash pod_scripts/eval_lm.sh