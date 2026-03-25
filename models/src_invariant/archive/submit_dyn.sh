#!/bin/bash
#SBATCH --job-name=dynam_sim            # Job name
#SBATCH --time=2:00:00                  # Maximum run time (10 hours)
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1               # CPU cores per task
#SBATCH --mem=16GB                      # Memory per job (adjust as needed)

file_path=$1
params=$2
trial_name=$3
operators=$4

module load 
run_trial(file_path, trial_name, params, operators)


