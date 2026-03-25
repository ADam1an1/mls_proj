import os, sys
from qutip import *
import numpy as np
from tqdm import tqdm
from total import *
import copy

def gen_hamiltonian(params):
    return TotalSystem(systems = params['system_e_levels'],
                       photon_freqs = params['photon_freqs'],
                       max_photon_nums = params['photon_max_nums'],
                       lambdas = params['lambdas'],
                       mus = params['mus'],
                       positions = params['positions'],
                       model = params['model'], 
                       filepath=params['filepath'])

def gen_two_tls_sep(params, mini, maxi, step):
    systems = []
    params['positions'][0] = 1 / 4 / params['photon_freqs'][0]
    for position in tqdm(np.arange(mini, maxi, step)):
        params['positions'][1] = params['positions'][0] * 2 + position
        total_system = gen_hamiltonian(params)
        systems.append(total_system)
    return systems

def gen_mult_tls_trans(params, mini, maxi, step):
    systems = []
    for position in tqdm(np.arange(mini, maxi, step)):
        params['positions'] = [position] * len(params['system_e_levels'])
        total_system = gen_hamiltonian(params)
        systems.append(total_system)
    return systems

def get_hamiltonians(params, chain_length, mini , maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lam_factor = coupling * np.sqrt(2 * params['photon_freqs'][0])
        lambdas = [[0, 0, lam_factor]]
        params['lambdas'] = lambdas   
        total_system = gen_hamiltonian(params)
        systems.append(total_system)
    return systems

def get_2ls_hamiltonians(params, chain_length, mini , maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    if not params['mus']:
        parmas['mus'] = [[[[0, -1], [-1, 0]]] * 3] * chain_length
    
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lam_factor = coupling * np.sqrt(2 * params['photon_freqs'][0])
        lambdas = [[0, 0, lam_factor]]
        # generation of mus needs to be adaptes
        params['lambdas'] = lambdas
        total_system = gen_hamiltonian(params)
        systems.append(total_system)

    return systems
    
def get_3ls_hamiltonians(params, chain_length, mini, maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lam_factor = coupling * np.sqrt(2 * params['photon_freqs'][0])
        lambdas = [[0, 0, lam_factor]]
        params['lambdas'] = lambdas
        total_system = gen_hamiltonian(params)
        systems.append(total_system)
    return systems

def get_4ls_hamiltonians(params, chain_length, mini, maxi, step):
    # basic chain with no spatial components
    systems = []        
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lam_factor = coupling * np.sqrt(2 * params['photon_freqs'][0])
        lambdas = [[0, 0, lam_factor]]
        params['lambdas'] = lambdas        
        total_system = gen_hamiltonian(params)
        systems.append(total_system)
    return systems

def transition_generator(levels_list, trans_dict):
    mus = []
    for sys_id in range(len(levels_list)):
        trans = np.zeros((levels_list[sys_id], levels_list[sys_id]))
        for start in trans_dict[sys_id].keys():
            for end in trans_dict[sys_id][start].keys():
                trans[start, end] += trans_dict[sys_id][start][end]
                trans[end, start] += trans_dict[sys_id][start][end]
        mus.append([trans, trans, trans])
    return mus

