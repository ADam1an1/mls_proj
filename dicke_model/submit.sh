#!/bin/bash
#SBATCH --job-name=2_2_10           # Job name
#SBATCH --output=/scratch/avd383/dicke_model/logs/H2_%a.out      # Output (saved to logs folder, %a is array ID)
#SBATCH --error=/scratch/avd383/dicke_model/logs/H2_%a.err       # Error
#SBATCH --time=00:15:00              # Time limit per task
#SBATCH --mem=2G                     # Memory per task
#SBATCH --array=0-100             # Array range (x jobs total)
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --comment="preemption=yes;requeue=true"


# --- CONFIGURATION ---

# 1. Calculate Lambda for this specific task
# Formula: Lambda = 0 + (Task_ID * 0.01)
# We use 'awk' for floating point math since bash cannot do it natively.
echo "Running MLS_full.bash with Lambda: $LAMBDA_VAL"

# --- EXECUTION ---

# We pass the SINGLE calculated lambda to the script.
module purge

p_freqs=0.63
pmn=10
chain=1
eigen=20
bases="6-311g_cas_2_2"
filepath="/scratch/avd383/qed-ci/H2_cas"
dir_dest="/scratch/avd383/dicke_model/H2${chain}_pn${pmn}_res_eigen${eigen}_${bases}"
mkdir -p ${dir_dest}

cp MLS_full.bash ${dir_dest}
cp MOL_trial.py ${dir_dest}
cd ${dir_dest}
sed -i "s|PFREQ|$p_freqs|g" MLS_full.bash	
sed -i "s|PMN|$pmn|g" MLS_full.bash	
sed -i "s|CHAIN|$chain|g" MLS_full.bash	
sed -i "s|EIGEN|$eigen|g" MLS_full.bash	
sed -i "s|BASES|$bases|g" MLS_full.bash	
sed -i "s|FILEPATH|$filepath|g" MLS_full.bash	
sed -i "s|DIR_DEST|$dir_dest|g" MLS_full.bash	


for (( i=0; i<40; i++ ));
do 
	LAMBDA_VAL="$(awk -v id=$SLURM_ARRAY_TASK_ID -v i=$i 'BEGIN { print (i * 0.1) + (id * 0.01) }')"

	echo "Running MLS_full.bash with Lambda: $LAMBDA_VAL"
	bash /share/apps/images/run-intel-oneapi-2025.2.1.bash << EOF
	source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
	conda activate /scratch/avd383/qed-ci/qed_foley/
	echo $LAMBDA_VAL
	bash MLS_full.bash $LAMBDA_VAL 

EOF
done

