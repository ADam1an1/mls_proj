# run_sim.py

import numpy as np
from qutip import sesolve
import tqdm
from total import TotalSystem


def get_nyquist(time, steps):
    dt = time / steps
    print("resolution given by nyquist:", dt/2)

def check_simulation(system_e_levels, photon_freqs, max_photon_nums,
                     couplings_dict, system_starts, photon_starts):
    """
    Function to check a multipartite system in cavity is properly specified
    """

    # check modes all have a frequency and counts
    if len(photon_freqs) != len(max_photon_nums):
        print("Photon number and listed frequencies different lengthes")
        raise

    # check that each mode has a valid starting state
    if len(photon_starts) != len(photon_freqs):
        print("Photon(s) initial state unclear")
        raise
    for mode in range(len(photon_starts)):
        if photon_starts[mode] < 0 or photon_starts[mode] > max_photon_nums[mode]:
            print("Photon number unphysical for mode {}".format(mode))


    # check that each system has a valid starting state 
    for i in range(len(system_e_levels)):
        if len(system_e_levels[i]) < system_starts[i]:
            print("System {} starts in an unphysical system energy level".format(i))
            raise

    # check that each mode has a coupling dict
    if len(couplings_dict.keys()) != len(photon_freqs):
        print("Coupling strengths specified for {} photons while {} photon freqs".format(len(coupling_strs), len(photon_freqs)))
        raise

    # check that each system e transition has a coupling for each mode
    for i in couplings_dict.keys():
        if len(couplings_dict[i].keys()) != len(system_starts):
            print("Coupling strengths not specified for all systems at {} photon freq".format(i))
            raise
        for sys in range(len(system_starts)):
            for level_start in range(len(system_e_levels[sys])):
                for level_end in range(level_start):
                    try:
                        coupling = couplings_dict[i][sys][level_start][level_end]
                    except:
                        print("Coupling strength not defined for freq {} and system {} between {} and {}".format(i, sys, level_start, level_end))
                        raise


def check_system_positions(length, system_positions):
    """
    Check that systems are positioned within the cavity
    """
    for i in range(len(system_positions)):
        system_pos = system_positions[i]
        if system_pos < 0 or system_pos > length:
            print("System position for system {} outside of cavity!".format(i))
            raise

def calc_spatial_coupling(couplings_dict, system_positions, system_e_levels, photon_freqs):
    """
    Modify the coupling dict taking into account the system positions within
    the photon mode (breaking long wave approximation)
    """
    print("Updating couplings")
    for freq in photon_freqs:
        for system in range(len(system_positions)):
            for level_start in range(len(system_e_levels[system])):
                for level_end in range(level_start):
                    cur_coupling = couplings_dict[freq][system][level_start][level_end]
                    factor = np.abs(np.sin(system_positions[system] * freq))
                    couplings_dict[freq][system][level_start][level_end] = cur_coupling * factor

    print(couplings_dict)
    return couplings_dict

def run_simulation(system_e_levels, photon_freqs,
                   max_photon_nums, couplings_dict,
                   system_starts, photon_starts, time, steps,
                   spatial, system_positions=[], track=[]):

    """
    Function to actually run the simulation
    """

    # check that parameters are valid, and track photon nums, e level state population
    get_nyquist(time, steps)

    check_simulation(system_e_levels, photon_freqs, max_photon_nums,
                     couplings_dict, system_starts, photon_starts)

    # set up the system as specified
    print("Generating Hamiltonian")
    print("Systems positioned at: {}".format(str(system_positions)))
    if spatial:
        couplings_dict = calc_spatial_coupling(couplings_dict, system_positions,
                                              system_e_levels, photon_freqs)

    total_system = TotalSystem(system_e_levels, photon_freqs, max_photon_nums,
                 couplings_dict, system_starts, photon_starts)

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
    if "states" in track:
        operators += total_system.gen_sys_operators()

    # run dynamics
    # start in state specified
    print("Systems starting in energies: " + str([levels[start] for start, levels in zip(system_starts, system_e_levels)]))
    print("Photons starting with: " + str(photon_starts))

    tlist = np.linspace(0, time, steps)
    result = sesolve(H, psi0, tlist, e_ops=operators, progress_bar='tqdm')

    return result
