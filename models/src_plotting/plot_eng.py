import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm
from qutip import expect


def plot_energies(systems, states, mini, maxi, step, omega = 1, norm_state=None, xlabel="", model="", lims=[]):
    i = 0
    xs = []
    ys = [[] for _ in range(len(states))]
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)        
        for j in range(len(states)):
            state = states[j]
            if norm_state != None:
                ys[state].append((eigenE[state] - eigenE[norm_state]) / omega)
            else:
                ys[state].append(eigenE[state])
        i += 1

    for j in range(len(states)):
        plt.plot(xs, ys[j], label=states[j])
    # plt.legend(ncols = len(states) // 5)
    plt.xlabel(xlabel)

    if norm_state != None:
        plt.title("Eigenstate Energies Normalized to State {}, model = {}".format(norm_state, model))
        plt.ylabel("(E_j - E_0)/{} (a.u.)".format(omega))
    else:
        plt.title("Eigenstate Energies, model = {}".format(model))
        plt.ylabel("E_j (a.u.)".format(omega))

    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.show()

def plot_eng_comp(systems, state, mini, maxi, step, omega = 1, norm_state=None, xlabel="", model="", lims=[]):
    i = 0
    eng_types = ["elec", "dse", "photon", "blc"]
    xs = []
    ys = {}
    for eng_type in eng_types:
        ys[eng_type] = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)

        if "pzw" in model:
            U = systems[i].gen_pzw()
        else:
            U = systems[i].identity
            
        operator_elec = U * systems[i].total_elec * U.dag()
        operator_dse = U * systems[i].total_dse * U.dag()
        operator_photon = U * systems[i].total_photon * U.dag()
        operator_blc = U * systems[i].total_blc * U.dag()
        if norm_state != None:
            ys["elec"].append((expect(operator_elec, eigenV[state]) - expect(operator_elec, eigenV[state])) / omega)
            ys["dse"].append((expect(operator_dse, eigenV[state]) - expect(operator_dse, eigenV[state])) / omega)
            ys["photon"].append((expect(operator_photon, eigenV[state]) - expect(operator_photon, eigenV[state])) / omega)
            ys["blc"].append((expect(operator_blc, eigenV[state]) - expect(operator_blc, eigenV[state])) / omega)
        else:
            ys["elec"].append(expect(operator_elec, eigenV[state]))
            ys["dse"].append(expect(operator_dse, eigenV[state]))
            ys["photon"].append(expect(operator_photon, eigenV[state]))
            ys["blc"].append(expect(operator_blc, eigenV[state]))
        
        i += 1

    for eng_type in eng_types:
        plt.plot(xs, ys[eng_type], label=eng_type)
            
    plt.legend()
    plt.xlabel(xlabel)

    if norm_state != None:
        plt.title("Eigenstate Energies Normalized to State {}, model = {}".format(norm_state, model))
        plt.ylabel("(E_j - E_0)/{} (a.u.)".format(omega))
    else:
        plt.title("Energy Decomposition for State {}, model = {}".format(state, model))
        plt.ylabel("E_j (a.u.)".format(omega))

    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.show()
