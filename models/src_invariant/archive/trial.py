import os, sys 
sys.path.append("/home/avd383/tavis_cummings_model")
sys.path.append("home/avd383/tavis_cummings_model/src")
from src.run_sim import *
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str, help='location to run and save trial')
parser.add_argument('--chain_len', type=int, default=-1, help='frame to use from trajectory')
parser.add_argument('--lambda_scaling', type=str, default='uni')



parser.add_argument('--time', type=int, default=100)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--base_coupling', type=float, default=0)
parser.add_argument('--system_e_levels', nargs='+', default=[0, 1])
parser.add_argument('--system_starts', nargs='+', default=None)
photon_freqs()
parser.add_argument('--dicke', default=False)



args = parser.parse_args()



file_path = args.file_path
chain_len = args.chain_len

params = {
    'time': 1000,
    'steps': 50000,
    'base_coupling': 3,
    'system_e_levels': None,
    'system_starts': None,
    'photon_starts': [0],
    'photon_freqs': [1],
    'photon_max_nums': None,
    'spatial': False,
    'couplings': None,
    'dicke': True,
    'descr': ""
}

system_e_level = [0, 1]
operators = ["photons"]

trial_name = 'run_{}_{}_{}_{}_{}'.format(i, "uni", params['photon_freqs'], params["base_coupling"], params["time"], params["steps"])
params['system_e_levels'] = np.array([system_e_level for _ in range(i)])
params['system_starts'] = np.array([1 for _ in range(i)])
params['photon_max_nums'] =[i]
    
coupling_strengths = []
coupling_strengths.append([])
for j in range(i):
    coupling_strengths[0].append([[params['base_coupling']]])
params['couplings'] = coupling_strengths
    
run_trial(file_path, trial_name, params, operators)

