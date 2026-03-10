#!/bin/bash
#SBATCH --job-name=alphaedit_causal_trace
#SBATCH --account=project_462000919
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --gpus=8
#SBATCH --output=logs/causal_trace_%j.out
#SBATCH --error=logs/causal_trace_%j.err

mkdir -p logs

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

cd /scratch/project_462000919/yagao/code_repo/AlphaEdit/AlphaEdit
source .venv/bin/activate

export HF_HOME=/scratch/project_462000812/hf-cache

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"

python -m experiments.causal_trace \
    --model_name Qwen/Qwen3-32B \
    --device_map auto \
    --noise_level s3 \
    --torch_dtype bfloat16

echo "Job completed at $(date)"
