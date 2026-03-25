# run_sim.py

import numpy as np
from qutip import mesolve
import tqdm
from total import TotalSystem


def get_nyquist(time, steps):
    dt = time / steps
    print("resolution given by nyquist:", dt/2)

def run_simulation(system_starts, photon_starts, total_system, losses=[], track=[], time, steps):
    """
    Function to actually run the simulation
    """
    # check that parameters are valid, and track photon nums, e level state population
    get_nyquist(time, steps)

    # check_simulation(system_e_levels, photon_freqs, max_photon_nums,
    #                  couplings_dict, system_starts, photon_starts)

    # set up the system as specified
    
    psi0 = total_system.gen_total_state(system_starts, photon_starts)
    print("Starting state has dims: {}".format(psi0.dims))
    H = total_system.total_hamiltonian

    # set up system_operators
    print("Making operators")
    operators = []
    if "energy" in track:
        operators.append(total_system.total_hamiltonian)
    if "photons" in track:
        operators += total_system.gen_cavity_operators()
    if "inversion" in track:
        operators += total_system.gen_inv_operator()
    if "emi_rate" in track:
        operators += total_system.gen_emi_operator()
    if "j2" in track:
        operators += total_system.gen_j2_operator()
    if "states" in track:
        operators += total_system.gen_sys_operators()

    # run dynamics
    # start in state specified
    print("Systems starting in energies: " + str([levels[start] for start, levels in zip(system_starts, system_e_levels)]))
    print("Photons starting with: " + str(photon_starts))

    tlist = np.linspace(0, time, steps)
    result = sesolve(H, psi0, tlist, e_ops=operators, progress_bar='tqdm')

    return result
