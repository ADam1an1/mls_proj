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
    TODO: NEED TO REIMPLEMENT FOR OUR CURRENT METHODS
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
    if len(couplings_dict) != len(photon_freqs):
        print("Coupling strengths specified for {} photons while {} photon freqs".format(len(coupling_strs), len(photon_freqs)))
        raise

    # check that each system e transition has a coupling for each mode
    for i in range(len(couplings_dict)):
        if len(couplings_dict[i]) != len(system_starts):
            print("Coupling strengths not specified for all systems at {} photon freq".format(photon_freqs[i]))
            raise
        for sys in range(len(system_starts)):
            for level_start in range(len(system_e_levels[sys])):
                for level_end in range(level_start):
                    try:
                        coupling = couplings_dict[i][sys][level_end][level_start - level_end - 1]
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
    for n in range(len(photon_freqs)):
        for system in range(len(system_positions)):
            for level_start in range(len(system_e_levels[system])):
                for level_end in range(level_start):
                    cur_coupling = couplings_dict[n][system][level_end][level_start - level_end - 1] # going from high -> low always 
                    factor = np.abs(np.sin(system_positions[system] * photon_freqs[n]))
                    couplings_dict[n][system][level_end][level_start - level_end - 1] = cur_coupling * factor
    print(couplings_dict)
    return couplings_dict

def run_simulation(system_e_levels, photon_freqs,
                   max_photon_nums, couplings_dict,
                   system_starts, photon_starts, time, steps,
                   spatial, system_positions=[], track=[], model=""):

    """
    Function to actually run the simulation
    """

    # check that parameters are valid, and track photon nums, e level state population
    get_nyquist(time, steps)

    # check_simulation(system_e_levels, photon_freqs, max_photon_nums,
    #                  couplings_dict, system_starts, photon_starts)

    # set up the system as specified
    print("Generating Hamiltonian")
    print("Systems positioned at: {}".format(str(system_positions)))
    if spatial:
        couplings_dict = calc_spatial_coupling(couplings_dict, system_positions,
                                              systems, photon_freqs)

    total_system = TotalSystem(systems, photon_freqs, max_photon_nums,
                 lambdas, mus, system_starts, photon_starts, model=model)

    psi0 = total_system.gen_total_state(systems_dirac, photons_dirac)
    print("Starting state has dims: {}".format(psi0.dims))

    # set up system_operators
    print("Making operators")
    operators = []
    operator_labels = []
    if "energy" in track:
        operators.append(total_system.total_hamiltonian)
        operator_labels.append("Energy")
    if "photons" in track:
        operators += total_system.gen_cavity_operators()
        for freq in photon_freqs:
            operator_labels.append("Photons of freqs {}".format(freq))
    if "states" in track:
        operators += total_system.gen_sys_operators()
        operator_labels.append()
        

    # run dynamics
    # start in state specified
    print("Systems starting in energies: " + str([levels[start] for start, levels in zip(system_starts, system_e_levels)]))
    print("Photons starting with: " + str(photon_starts))

    tlist = np.linspace(0, time, steps)
    result = sesolve(H, psi0, tlist, e_ops=operators, progress_bar='tqdm')

    return result
