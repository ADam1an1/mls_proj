import numpy as np
from tqdm import tqdm
from total import *
from qutip import *
import os, sys

def get_hamiltonians(params, chain_length, min , max, step, 
                     scale=False, spatial=None):
    # basic chain with no spatial components, 1/sqrt(N) factor
    systems = []
    for coupling in np.arange(min, max, step):       
        coupling_strengths = []
        coupling_strengths.append([])
        for j in range(chain_length):
            if scale:
                factor = 1
                if j != 0:
                    factor *= np.exp(complex(0, np.pi * coupling))
                coupling_strengths[0].append([[0, params['base_coupling'] * factor], []])
            else: 
                coupling_strengths[0].append([[coupling, 0], []])

        params['couplings'] = coupling_strengths
        
        total_system = TotalSystem(params['system_e_levels'], 
                                   params['photon_freqs'], 
                                   params['photon_max_nums'],
                                   params['couplings'], 
                                   params['system_starts'], 
                                   params['photon_starts'], 
                                   model=params['model'])
        systems.append(total_system)
        
    return systems

def get_hamiltonians_spatial(params, min , max, step, scale=False, base_pos=0):
    def calc_spatial_coupling(couplings_dict, system_positions, system_e_levels, photon_freqs):
        """
        Modify the coupling dict taking into account the system positions within
        the photon mode (breaking long wave approximation)
        """
        for n in range(len(photon_freqs)):
            for system in range(len(system_positions)):
                for level_start in range(len(system_e_levels[system])):
                    for level_end in range(level_start):
                        cur_coupling = couplings_dict[n][system][level_end][level_start - level_end - 1] 
                        # going from high -> low always 
                        factor = np.abs(np.sin(system_positions[system] * photon_freqs[n]))
                        couplings_dict[n][system][level_end][level_start - level_end - 1] = cur_coupling * factor
        return couplings_dict
        
    systems = []
    for spacing in np.arange(min, max, step):       
        positions = [base_pos, base_pos + spacing]
        params['couplings'] = calc_spatial_coupling(params['couplings'], positions,
                                                    params['system_e_levels'], params['photon_freqs'])
        total_system = TotalSystem(params['system_e_levels'], 
                                   params['photon_freqs'], 
                                   params['photon_max_nums'],
                                   params['couplings'], 
                                   params['system_starts'], 
                                   params['photon_starts'], 
                                   model=params['model'])
        systems.append(total_system)
    return hamiltonians