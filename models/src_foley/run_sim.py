# run_sim.py

import numpy as np
from qutip import mesolve, mcsolve, tensor, destroy
import tqdm
from total import TotalSystem


def get_nyquist(time, steps):
    dt = time / steps
    print("resolution given by nyquist:", dt/2)

def run_simulation(total_system, system_starts, photon_starts, time, steps, losses={}, track=[], model='me', ntraj=1):
    """
    Function to actually run the simulation
    """
    # check that parameters are valid, and track photon nums, e level state population
    get_nyquist(time, steps)
    # set up the system as specified
    print("Generating Hamiltonian")
    
    psi0 = total_system.gen_total_state(system_starts, photon_starts)
    print("Starting state has dims: {}".format(psi0.dims))
    H = total_system.total_hamiltonian
    
    # set up system_operators
    print("Making operators")
    operators = []
    
    if "energy" in track:
        print(total_system)
        operators.append(total_system.total_hamiltonian)
    if "photons" in track:
        operators += total_system.gen_cavity_operators()
    if "states" in track:
        operators += total_system.gen_sys_operators()
    eigs = []
    for tracker in track:
        if isinstance(tracker, int):
            eigs.append(tracker)    
    operators += total_system.gen_pol_operators(eigs)
    
    # run dynamics
    # start in state specified
    print("calculating loss operators")
    jumps = []
    if 'gamma' in losses.keys():
        jumps.extend(total_system.gen_gamma_losses(losses['gamma']))
        
    if 'kappa' in losses.keys():
        jumps.extend(total_system.gen_kappa_losses(losses['kappa']))
            
    print("starting dynamics")
    tlist = np.linspace(0, time, steps)
    if model == 'mc':
        result = mcsolve(H, psi0, tlist, c_ops=jumps, e_ops=operators, ntraj=ntraj, progress_bar='tqdm')
    if model == 'me':
        result = mesolve(H, psi0, tlist, c_ops=jumps, e_ops=operators, progress_bar='tqdm')
    
    return result
