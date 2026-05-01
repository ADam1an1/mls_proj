import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm
from qutip import expect

def plot_compare(systems_s, states, mini, maxi, step, 
                 omega = 1, norm_state=None, xlabel="", ylabel="", title="", lims=[], operator="", models_s=None):
    i = 0
    xs = []
    ys_s = [[] for k in range(len(states))]
    nsystems = len(systems_s)
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        hamiltonians = [systems_s[j][i].total_hamiltonian for j in range(nsystems)]
        eigenEs, eigenVs = zip(*(hamiltonian.eigenstates() for hamiltonian in hamiltonians))
        eigenEs, eigenVs = list(eigenEs), list(eigenVs)

        Us = [systems_s[j][i].gen_transform(models_s[j]) 
              for j in range(nsystems)]
        
        if operator == "pn":
            operator_evals = [Us[j] * systems_s[j][i].gen_cavity_operators()[0] * Us[j].dag()
                              for j in range(nsystems)]
        elif operator == "elec":
            operator_evals = [Us[j] * systems_s[j][i].total_elec * Us[j].dag()
                              for j in range(nsystems)]
        elif operator == "photon":
            operator_evals = [Us[j] * systems_s[j][i].total_photon * Us[j].dag()
                              for j in range(nsystems)]
        elif operator == "blc":
            operator_evals = [Us[j] * systems_s[j][i].total_blc * Us[j].dag()
                              for j in range(nsystems)]
        elif operator == "dse":
            operator_evals = [Us[j] * systems_s[j][i].total_dse * Us[j].dag()
                              for j in range(nsystems)]
            
        for k in range(len(states)):
            state = states[k]
            if operator == "" or operator == "energy":
                if norm_state != None:
                    ys_s[k].append([(eigenEs[j][state] - eigenEs[j][norm_state]) / omega 
                                  for j in range(nsystems)])
                else:
                    ys_s[k].append([eigenEs[j][state] for j in range(nsystems)])
            else:
                ys_s[k].append([expect(operator_evals[j], eigenVs[j][state]) for j in range(nsystems)])
        i += 1

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    styles = ['-', ';', '--', '-.']
    ys_s = np.array(ys_s)
    if operator == "" or operator == "energy":
        ys_s = ys_s.squeeze()
    for k in range(len(states)):
        for j in range(nsystems):
            plt.plot(xs, ys_s[k, :, j], c=colors[j % len(colors)], linestyle=styles[j % len(styles)], lw=2)
            
    legend_elements = [
    mpl.lines.Line2D([0], [0], color=colors[j % len(colors)], linestyle=styles[j % len(styles)], label=models_s[j])
    for j in range(nsystems)
    ]
    plt.legend(handles=legend_elements, fontsize=10)
    plt.xlabel(xlabel, fontsize=14)
    
    if not ylabel:
        if operator == "" or operator=="energy":
            if norm_state != None:
                ylabel = "(E_j - E_0)/{} (a.u.)".format(omega)
            else:
                ylabel = "E_j (a.u.)".format(omega)
        if operator == "pn":
            ylabel = r"Photon Number $\langle a^\dagger a \rangle$"
        elif operator == "elec":
            ylabel = "Energy (a.u.)"
        elif operator == "photon":
            ylabel = "Energy (a.u.)"
        elif operator == "blc":
            ylabel = "Energy (a.u.)"
        elif operator == "dse":
            ylabel = "Energy (a.u.)"

    if not title:
        if operator == "" or operator=="energy":
            if norm_state != None:
                title = "Eigenstate Energies Normalized to State {}".format(norm_state)
            else:
                title = "Eigenstate Energies"
        if operator == "pn":
            title = "Photon Number for Eigenstate {}".format(states[0])
        elif operator == "elec":
            title = "Electronic Contribution to Eigenstate {} Energy".format(states[0])
        elif operator == "photon":
            title = "Photon Contribution to Eigenstate {} Energy".format(states[0])
        elif operator == "blc":
            title = "Bilinear Coupling Contribution to Eigenstate {} Energy".format(states[0])
        elif operator == "dse":
            title = "Dipole Self Energy Contribution to Eigenstate {} Energy".format(states[0])        
    
    plt.title(title)
    plt.ylabel(ylabel, fontsize=14)
    
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.show()



def plot_compare_lambdas(systems_s, states, mini, maxi, step, lambdas, 
                 omega = 1, norm_state=None, xlabel="", lims=[], operator="", models_s=None):
    i = 0
    ys_s = [[] for k in range(len(states))]
    nsystems = len(systems_s)
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonians = [systems_s[j][i].total_hamiltonian for j in range(nsystems)]
        eigenEs, eigenVs = zip(*(hamiltonian.eigenstates() for hamiltonian in hamiltonians))
        eigenEs, eigenVs = list(eigenEs), list(eigenVs)

        Us = [systems_s[j][i].gen_transform(models_s[j]) 
              for j in range(nsystems)]
        
        operator_evals = [Us[j] * systems_s[j][i].gen_cavity_operators()[0] * Us[j].dag()
                          for j in range(nsystems)]
        
        for k in range(len(states)):
            state = states[k]
            ys_s[k].append([expect(operator_evals[j], eigenVs[j][state]) for j in range(nsystems)])
        
        i += 1

    ys_s = np.array(ys_s)
    for k in range(len(states)):
        for j in range(nsystems):
            plt.scatter(lambdas[j], np.argmax(ys_s[k, :, j]) * step)
            
    plt.xlabel(xlabel)
    plt.title("Photon Number for Eigenstate {} as Function of $\lambda$".format(states[0]))
    plt.ylabel("Absolute Field Magnitude, $\lambda$, at Max PN")
    
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    else:
        plt.ylim((mini, maxi))
    plt.show()