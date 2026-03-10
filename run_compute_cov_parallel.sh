#!/bin/bash
#
# Submit 5 parallel covariance jobs (one per layer) + 1 aggregation job.
# Usage: bash run_compute_cov_parallel.sh
#

mkdir -p logs

LAYERS=(6 7 8 9 10)
JOB_IDS=""

for LAYER in "${LAYERS[@]}"; do
    # Write per-layer script to a temp file
    SCRIPT="logs/_compute_layer${LAYER}.sh"
    cat > "${SCRIPT}" <<INNER
#!/bin/bash
#SBATCH --job-name=cov_L${LAYER}
#SBATCH --account=project_462000919
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --gpus=8
#SBATCH --output=logs/compute_cov_L${LAYER}_%j.out
#SBATCH --error=logs/compute_cov_L${LAYER}_%j.err

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

cd /scratch/project_462000919/yagao/code_repo/AlphaEdit/AlphaEdit
source .venv/bin/activate

export HF_HOME=/scratch/project_462000919/yagao/hf-cache

echo "Job started at \$(date) — Layer ${LAYER}"
echo "Running on node: \$(hostname)"

python compute_cov_parallel.py \\
    --mode compute \\
    --layer ${LAYER} \\
    --model_name Qwen/Qwen3-32B \\
    --hparams_fname Qwen3-32B.json \\
    --output_dir ./precomputed \\
    --torch_dtype bfloat16

echo "Job completed at \$(date) — Layer ${LAYER}"
INNER

    JOB_ID=$(sbatch --parsable "${SCRIPT}")
    echo "Submitted layer ${LAYER} as job ${JOB_ID}"
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="${JOB_ID}"
    else
        JOB_IDS="${JOB_IDS}:${JOB_ID}"
    fi
done

# Write aggregation script
AGG_SCRIPT="logs/_compute_aggregate.sh"
cat > "${AGG_SCRIPT}" <<'INNER'
#!/bin/bash
#SBATCH --job-name=cov_agg
#SBATCH --account=project_462000919
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --gpus=0
#SBATCH --output=logs/compute_cov_agg_%j.out
#SBATCH --error=logs/compute_cov_agg_%j.err

module use /appl/local/csc/modulefiles/
module load pytorch/2.7

cd /scratch/project_462000919/yagao/code_repo/AlphaEdit/AlphaEdit
source .venv/bin/activate

echo "Aggregation started at $(date)"

python compute_cov_parallel.py \
    --mode aggregate \
    --hparams_fname Qwen3-32B.json \
    --output_dir ./precomputed

echo "Aggregation completed at $(date)"
INNER

AGG_ID=$(sbatch --parsable --dependency=afterok:${JOB_IDS} "${AGG_SCRIPT}")
echo "Submitted aggregation as job ${AGG_ID} (depends on ${JOB_IDS})"
echo "Done. Monitor with: squeue -u \$USER"
