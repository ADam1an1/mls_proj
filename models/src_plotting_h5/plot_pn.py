import matplotlib.pyplot as plt
import numpy as np
from qutip import expect
from tqdm import tqdm
import h5py

def plot_aa(test_dir, file_prefix, states, mini, maxi, step, xlabel = "", title = None, freq_ind=0):
    i = 0
    xs = []
    ys = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:   
            freq = str(f.attrs['photon_freqs'][freq_ind])
            ys.append(f["photon_numbers"][freq][states])

    for j in range(len(states)):
        plt.plot(xs, np.array(ys)[:, j], label=states[j])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Photon Number <a dagger a>")
    if title:
        title = title
    else:
        title = "Photon Number for Eigenstates"
    plt.title(title)
    plt.show()
