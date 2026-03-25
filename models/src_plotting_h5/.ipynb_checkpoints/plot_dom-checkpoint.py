import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from qutip import expect
from tqdm import tqdm
import h5py

def plot_domstate(test_dir, file_prefix, states, mini, maxi, step, omega = 1, norm_state=None, 
                    xlabel="", model="", op_list=[], 
                    ax=None, cmap='viridis', linewidth=2, lims=[]):        
    # generate the operators and their labels to use
    # plot the colors and eigenstates requested
    xs = []
    ys = []
    colors = [[] for _ in states]
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)   
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:
            eigenE = f['energies']['eigenE']
            if norm_state != None:
                ys.append((eigenE[states] - [eigenE[norm_state] for _ in states]) / omega)
            else:
                ys.append(eigenE[states])

            for state in states:
                excitations = f['projections'][str(state)][op_list]
                colors[state].append(np.argmax(excitations))
            op_labels = f['projections'].attrs['op_desc']

    # add in the legend for the plot
    ax = plt.gca()
    all_colors = np.concatenate(colors)
    cmin, cmax = all_colors.min(), all_colors.max()
    norm = plt.Normalize(cmin, cmax)
    for j in range(len(states)):
        points = np.array([xs, np.array(ys)[:, j]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(colors[j]) 
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    unique_ops = np.unique(all_colors).astype(int)
    for idx in range(len(op_list)):
        label = op_labels[op_list][idx]
        color = mapper.to_rgba(idx)
        plt.scatter([], [], c=[color], label=label, s=50)
    plt.legend(title="Main Projection")

    # add labels
    if norm_state != None:
        plt.title("Eigenstate Energies Normalized to State {}, model = {}".format(norm_state, model))
        plt.ylabel("(E_j - E_0)/{} (a.u.)".format(omega))
    else:
        plt.title("Eigenstate Energies, model = {}".format(model))
        plt.ylabel("(E_j (a.u.)".format(omega))
    
    ax.autoscale()
    plt.xlabel(xlabel)
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
        
    plt.show()
    plt.show()
