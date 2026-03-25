import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from qutip import expect
from tqdm import tqdm
import h5py

def plot_polaritons(test_dir, file_prefix, states, mini, maxi, step, omega = 1, norm_state=None,
                    xlabel="", model="",
                    ax=None, cmap='viridis', linewidth=2, freq_ind=0):
    if ax is None:
        ax = plt.gca()

    i = 0
    xs = []
    ys = []
    colors = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)   
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:
            eigenE = f['energies']['eigenE']
            if norm_state != None:
                ys.append((eigenE[states] - [eigenE[norm_state] for _ in states]) / omega)
            else:
                ys.append(eigenE[states])
            freq = str(f.attrs['photon_freqs'][freq_ind])
            colors.append(f["photon_numbers"][freq][states])
        i += 1

    all_colors = np.concatenate(colors).flatten()
    print(all_colors.shape)
    cmin, cmax = all_colors.min(), all_colors.max()
    norm = plt.Normalize(cmin, cmax)
    for j in range(len(states)):
        points = np.array([xs, np.array(ys)[:, j]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.array(colors)[:, j].flatten())
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    if norm_state != None:
        plt.title("Eigenstate Energies Normalized to State {}, model = {}".format(norm_state, model))
        plt.ylabel("(E_j - E_0)/{} (a.u.)".format(omega))
    else:
        plt.title("Eigenstate Energies, model = {}".format(model))
        plt.ylabel("(E_j (a.u.)".format(omega))

    ax.autoscale()
    plt.colorbar(lc, label=r"Average Photon Number $\langle a^\dagger a\rangle $")
    plt.xlabel(xlabel)
    plt.show()
