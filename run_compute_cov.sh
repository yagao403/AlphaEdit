#!/bin/bash
#SBATCH --job-name=alphaedit_cov
#SBATCH --account=project_462000919
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --gpus=8
#SBATCH --output=logs/compute_cov_%j.out
#SBATCH --error=logs/compute_cov_%j.err

mkdir -p logs

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

cd /scratch/project_462000919/yagao/baselines/AlphaEdit
source .venv/bin/activate

export HF_HOME=/scratch/project_462000812/hf-cache

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"

python compute_cov_and_proj.py \
    --model_name Qwen/Qwen3-32B \
    --hparams_fname Qwen3-32B.json \
    --output_dir ./precomputed \
    --torch_dtype float16

echo "Job completed at $(date)"
