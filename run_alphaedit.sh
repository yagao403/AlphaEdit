#!/bin/bash
#SBATCH --job-name=alphaedit_test
#SBATCH --account=project_462000919
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=dev-g
#SBATCH --gpus=8
#SBATCH --output=logs/alphaedit_%j.out
#SBATCH --error=logs/alphaedit_%j.err

# Create logs directory
mkdir -p logs

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# Set paths
cd /scratch/project_462000919/yagao/code_repo/AlphaEdit/AlphaEdit
source .venv/bin/activate


export HF_HOME=/scratch/project_462000919/yagao/hf-cache
echo "HF_HOME: $HF_HOME"

SAVE_NAME=${1:-"AlphaEdit"}

# Run evaluation
python experiments/evaluate.py \
    --alg_name AlphaEdit \
    --model_name Qwen/Qwen3-32B \
    --hparams_fname Qwen3-32B.json \
    --ds_name mquake_t \
    --dataset_size_limit 38 \
    --skip_generation_tests \
    --num_edits 1 \
    --torch_dtype bfloat16 \
    --save_model /scratch/project_462000919/yagao/checkpoints/AlphaEdit/${SAVE_NAME}


echo "Job completed at $(date)"
