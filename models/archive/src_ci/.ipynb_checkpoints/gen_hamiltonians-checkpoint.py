import os, sys
from qutip import *
import numpy as np
from tqdm import tqdm
from total import *

def get_hamiltonians(params, chain_length, mini , maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        params['lambdas'] = [coupling * np.sqrt(2 * params['photon_freqs'][0])] * len(params['system_e_levels'])
        total_system = TotalSystem(params['system_e_levels'],
                                   params['photon_freqs'],
                                   params['photon_max_nums'],
                                   params['lambdas'],
                                   params['mus'],
                                   model=params['model'])
        systems.append(total_system)
    return systems

def get_2ls_hamiltonians(params, chain_length, mini , maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lambdas = [[0, 0, coupling * np.sqrt(2 * params['photon_freqs'][0])]]
        # generation of mus needs to be adapted
        mus = [[[[0, -1], [-1, 0]]] * 3] * chain_length
        params['mus'] = mus
        params['lambdas'] = lambdas
        total_system = TotalSystem(params['system_e_levels'],
                                   params['photon_freqs'],
                                   params['photon_max_nums'],
                                   params['lambdas'],
                                   params['mus'],
                                   model=params['model'])
        systems.append(total_system)

    return systems

def get_3ls_hamiltonians(params, chain_length, mini, maxi, step):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        lambdas = [[[0, 0, coupling * np.sqrt(2 * params['photon_freqs'][0])]] * chain_length]
        # generation of mus needs to be adapted
        mus = [[[[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]] * 3] * chain_length
        params['mus'] = mus
        params['lambdas'] = lambdas
        total_system = TotalSystem(params['system_e_levels'],
                                   params['photon_freqs'],
                                   params['photon_max_nums'],
                                   params['lambdas'],
                                   params['mus'],
                                   model=params['model'])
        systems.append(total_system)

    return systems
