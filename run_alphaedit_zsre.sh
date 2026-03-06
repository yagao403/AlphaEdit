#!/bin/bash
#SBATCH --job-name=alphaedit_zsre_2000_100
#SBATCH --account=project_462000919
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=dev-g
#SBATCH --gpus=1
#SBATCH --output=logs/alphaedit_zsre_%j.out
#SBATCH --error=logs/alphaedit_zsre_%j.err

# Create logs directory
mkdir -p logs

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# Set paths
cd /scratch/project_462000812/yagao/baselines/AlphaEdit
source .venv/bin/activate

# Set HuggingFace cache (adjust to your model location)
export HF_HOME=/scratch/project_462000812/hf-cache
echo "HF_HOME: $HF_HOME"

# Run evaluation
python3 -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=zsre \
    --dataset_size_limit=2000 \
    --num_edits=100 \
    --downstream_eval_steps=5

echo "Job completed at $(date)"
