import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm
from qutip import expect

def plot_compare(systems1, systems2, states, mini, maxi, step, 
                 omega = 1, norm_state=None, xlabel="", lims=[], operator="", model1="", model2=""):
    i = 0
    xs = []
    y1 = [[] for _ in range(len(states))]
    y2 = [[] for _ in range(len(states))]
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        hamiltonian1 = systems1[i].total_hamiltonian
        eigenE1, eigenV1 = hamiltonian1.eigenstates()
        hamiltonian2 = systems2[i].total_hamiltonian
        eigenE2, eigenV2 = hamiltonian2.eigenstates()

        if "pzw" in model1:
            U1 = systems1[i].gen_pzw()
        else:
            U1 = systems1[i].identity

        if "pzw" in model2:
            U2 = systems2[i].gen_pzw()
        else:
            U2 = systems2[i].identity
        
        if operator == "pn":
            operator_eval1 = U1 * systems1[i].gen_cavity_operators()[0] * U1.dag()
            operator_eval2 = U2 * systems2[i].gen_cavity_operators()[0] * U2.dag()
        elif operator == "elec":
            operator_eval1 = U1 * systems1[i].total_elec * U1.dag()
            operator_eval2 = U2 * systems2[i].total_elec * U2.dag()
        elif operator == "photon":
            operator_eval1 = U1 * systems1[i].total_photon * U1.dag()
            operator_eval2 = U2 * systems2[i].total_photon * U2.dag()
        elif operator == "blc":
            operator_eval1 = U1 * systems1[i].total_blc * U1.dag()
            operator_eval2 = U2 * systems2[i].total_blc * U2.dag()
        elif operator == "dse":
            operator_eval1 = U1 * systems1[i].total_dse * U1.dag()
            operator_eval2 = U2 * systems2[i].total_dse * U2.dag()
            
        for j in range(len(states)):
            state = states[j]
            if operator == "" or operator == "energy":
                if norm_state != None:
                    y1[state].append((eigenE1[state] - eigenE1[norm_state]) / omega) 
                    y2[state].append((eigenE2[state] - eigenE2[norm_state]) / omega) 
                else:
                    y1[state].append(eigenE1[state])
                    y2[state].append(eigenE2[state])
            else:
                y1[state].append(expect(operator_eval1, eigenV1[state]))
                y2[state].append(expect(operator_eval2, eigenV2[state]))
        i += 1

    for j in range(len(states)):
        plt.plot(xs, y2[j], c='r', linestyle=':', lw=3)
        plt.plot(xs, y1[j], c='b', lw=1)
    legend_elements = [
    mpl.lines.Line2D([0], [0], color='b', linestyle='-', label=model1),
    mpl.lines.Line2D([0], [0], color='r', linestyle=':', label=model2)
    ]
    plt.legend(handles=legend_elements, fontsize=10)
    plt.xlabel(xlabel)

    if operator == "" or operator=="energy":
        if norm_state != None:
            plt.title("Eigenstate Energies Normalized to State {}".format(norm_state))
            plt.ylabel("(E_j - E_0)/{} (a.u.)".format(omega))
        else:
            plt.title("Eigenstate Energies")
            plt.ylabel("E_j (a.u.)".format(omega))
    if operator == "pn":
        plt.title("Photon Number for Eigenstate {}".format(states[0]))
        plt.ylabel("Photon Number <a dagger a>")
    elif operator == "elec":
        plt.title("Electronic Contribution to Eigenstate {} Energy".format(states[0]))
        plt.ylabel("Energy (a.u.)")
    elif operator == "photon":
        plt.title("Photon Contribution to Eigenstate {} Energy".format(states[0]))
        plt.ylabel("Energy (a.u.)")
    elif operator == "blc":
        plt.title("Bilinear Coupling Contribution to Eigenstate {} Energy".format(states[0]))
        plt.ylabel("Energy (a.u.)")
    elif operator == "dse":
        plt.title("Dipole Self Energy Contribution to Eigenstate {} Energy".format(states[0]))
        plt.ylabel("Energy (a.u.)")
    
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.show()