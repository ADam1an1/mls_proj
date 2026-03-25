import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from qutip import expect
from tqdm import tqdm


def plot_domstate(systems, states, mini, maxi, step, omega = 1, norm_state=None, 
                    xlabel="", model="", op_list=[], 
                    ax=None, cmap='viridis', linewidth=2, lims=[]):        
    # generate the operators and their labels to use
    if not op_list:
        for sys_id in range(systems[0].mp_systems.nsystems):
            for state in range(systems[0].mp_systems.dims[sys_id]):
                sys_dirac = ["I" if i != sys_id
                             else state 
                             for i in range(systems[0].mp_systems.nsystems)]
                op_list.append([sys_dirac, ["I"]])
    
    # plot the colors and eigenstates requested
    i = 0
    xs = []
    ys = [[] for _ in range(len(states))]
    colors = [[] for _ in range(len(states))]
    nsystems = systems[0].mp_systems.nsystems
    for coupling in tqdm(np.arange(mini, maxi, step)):
        ex_ops = []
        for sys_dirac, cav_dirac in op_list:
            ex_ops.append(systems[i].gen_joint_operator(sys_dirac, cav_dirac))
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)
        for j in range(len(states)):
            state = states[j]
            if norm_state != None:
                ys[state].append((eigenE[state] - eigenE[norm_state]) / omega) 
            else:
                ys[state].append(eigenE[state])
            excitations = [expect(ex_op, eigenV[state]) for ex_op in ex_ops]
            colors[state].append(np.argmax(excitations))
        i += 1

    # add in the legend for the plot
    ax = plt.gca()
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
        
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    unique_ops = np.unique(all_colors).astype(int)
    op_labels = []
    for sys_dirac, cav_dirac in op_list:
        op_labels.append(systems[0].gen_joint_label(sys_dirac, cav_dirac))
    for idx in unique_ops:
        label = op_labels[idx] if (op_labels and idx < len(op_labels)) else f"Op {idx}"
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
