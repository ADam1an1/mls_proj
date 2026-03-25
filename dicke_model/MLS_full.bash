#!/bin/bash

# setting up the help command
usage() {
        echo ""
        echo "Bash:"
        echo " Submit MLS_trial.py files with varying lambda"
        echo ""
}

# getting the variables
lambdas=($1)
lambdas=$(printf "%.3f " "${lambdas}")
lambdas=${lambdas% }


p_freqs=0.0583614
pmn=100
chain=1
eigen=10
dir_dest="/home/avd383/dicke_model/Rb${chain}_pn${pmn}_res_eigen${eigen}"
mkdir -p ${dir_dest}

# 0.0583614 = 384/6579.68974479
PYTHONPATH=/scratch/avd383/qed-ci/src/ python \
	MLS_trial.py \
  	--photon_freqs ${p_freqs} \
  	--photon_max_nums ${pmn} \
  	--lambdas 0 0 ${lambdas} \
  	--chain_length ${chain}  \
   	--system_e_levels rb  \
	--mus 0  \
	--eigenstates ${eigen} \
 	--save_dir ${dir_dest} \
	--save_name lambda_${lambdas}


echo "Finished generating all pieces of the Hamiltonian"

