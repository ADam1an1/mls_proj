import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm
from qutip import expect
import h5py


def plot_energies(test_dir, file_prefix, states, mini, maxi, step, omega = 1, norm_state=None, xlabel="", model="", lims=[]):
    i = 0
    xs = []
    ys = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)   
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:
            eigenE = f['energies']['eigenE']
            if norm_state != None:
                ys.append((eigenE[states] - [eigenE[norm_state] for _ in states]) / omega)
            else:
                ys.append(eigenE[states])
    print(np.array(ys).shape)
    for j in range(len(states)):
        plt.plot(xs, np.array(ys)[:, j], label=states[j])
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

def plot_eng_comp(test_dir, file_prefix, state, mini, maxi, step, omega = 1, norm_state=None, xlabel="", model="", lims=[]):
    i = 0
    eng_types = ["elec", "dse", "photon", "blc"]
    xs = []
    ys = {}
    for eng_type in eng_types:
        ys[eng_type] = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:
            feng = f['energies']
            eigenE = feng['eigenE']
            if norm_state != None:
                ys["elec"].append((feng['elec_eng'][state] - feng['elec_eng'][norm_state]) / omega)
                ys["dse"].append((feng['dse_eng'][state] - feng['dse_eng'][norm_state]) / omega)
                ys["photon"].append((feng['photon_eng'][state] - feng['photon_eng'][norm_state]) / omega)
                ys["blc"].append((feng['blc_eng'][state] - feng['blc_eng'][norm_state]) / omega)
            else:
                ys["elec"].append(feng['elec_eng'][state])
                ys["dse"].append(feng['dse_eng'][state])
                ys["photon"].append(feng['photon_eng'][state])
                ys["blc"].append(feng['blc_eng'][state])
        
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
