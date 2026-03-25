#!/bin/bash
#SBATCH --job-name=gen_plots           # Job name
#SBATCH --output=logs/Rb_%a.out      # Output (saved to logs folder, %a is array ID)
#SBATCH --error=logs/Rb_%a.err       # Error
#SBATCH --time=00:10:00              # Time limit per task
#SBATCH --mem=1G                     # Memory per task
#SBATCH --array=0-1000             # Array range (x jobs total)
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --comment="preemption=yes;requeue=true"


# --- CONFIGURATION ---

# 1. Calculate Lambda for this specific task
# Formula: Lambda = 0 + (Task_ID * 0.01)
# We use 'awk' for floating point math since bash cannot do it natively.
LAMBDA_VAL="$(awk -v id=$SLURM_ARRAY_TASK_ID 'BEGIN { print 0 + (id * 0.01) }')"

echo "Running MLS_full.bash with Lambda: $LAMBDA_VAL"

# --- EXECUTION ---

# We pass the SINGLE calculated lambda to the script.
module purge

bash /share/apps/images/run-intel-oneapi-2025.2.1.bash << EOF
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/avd383/qed-ci/qed_foley/
echo $LAMBDA_VAL
bash MLS_full.bash $LAMBDA_VAL

EOF

