#!/bin/bash

# setting up the help command
usage() {
        echo ""
        echo "Bash:"
        echo " Submit MLS_sstrial.py files with varying lambda"
        echo ""
}

# getting the variables
lambdas=($1)
lambdas=$(printf "%.3f " "${lambdas}")
lambdas=${lambdas% }


p_freqs=0.0583614
pmn=10
chain=1
eigen=30
kappa=0.2
gamma=0.2
dir_dest="/home/avd383/dicke_model/Rb${chain}_pn${pmn}_res_eigen${eigen}_ss_k${kappa}_g${gamma}"
mkdir -p ${dir_dest}

# 0.0583614 = 384/6579.68974479
PYTHONPATH=/scratch/avd383/qed-ci/src/ python \
	MLS_sstrial.py \
  	--photon_freqs ${p_freqs} \
  	--photon_max_nums ${pmn} \
  	--lambdas 0 0 ${lambdas} \
  	--chain_length ${chain}  \
   	--system_e_levels rb  \
	--mus 0  \
	--eigenstates ${eigen} \
 	--save_dir ${dir_dest} \
	--save_name lambda_${lambdas} \
	--kappa ${kappa} \
	--gamma ${gamma} 


echo "Finished generating all pieces of the Hamiltonian"

