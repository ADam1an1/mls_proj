import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from qutip import expect
from tqdm import tqdm

def plot_polaritons(systems, states, mini, maxi, step, omega = 1, norm_state=None,
                    xlabel="", model="",
                    ax=None, cmap='viridis', linewidth=2):
    if ax is None:
        ax = plt.gca()

    i = 0
    xs = []
    ys = [[] for _ in range(len(states))]
    colors = [[] for _ in range(len(states))]
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        photon_op = systems[i].gen_cavity_operators()[0]
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)
        for j in range(len(states)):
            state = states[j]
            if norm_state != None:
                ys[state].append((eigenE[state] - eigenE[norm_state]) / omega)
            else:
                ys[state].append(eigenE[state])
            colors[state].append(expect(photon_op, eigenV[state]))
        i += 1

    all_colors = np.concatenate(colors)
    cmin, cmax = all_colors.min(), all_colors.max()
    norm = plt.Normalize(cmin, cmax)
    for j in range(len(states)):
        points = np.array([xs, ys[j]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(colors[j])
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
