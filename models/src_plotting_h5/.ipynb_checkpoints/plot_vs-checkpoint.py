import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from tqdm import tqdm
import h5py

def plot_compare(test_dir1, file_prefix1, test_dir2, file_prefix2,
                 states, mini, maxi, step, 
                 omega = 1, norm_state=None, xlabel="", lims=[], operator="", model1="", model2="", freq_ind=0):
    i = 0
    xs = []
    y1 = []
    y2 = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        with h5py.File("{}/{}_{:.3f}".format(test_dir1, file_prefix1, coupling), "r") as f1:
            with h5py.File("{}/{}_{:.3f}".format(test_dir2, file_prefix2, coupling), "r") as f2:
        
                if operator == "pn":
                    group = 'photon_freqs'
                    subgroup = str(f.attrs['photon_freqs'][freq_ind])
                else:
                    group = 'energies'
                    
                    if operator == "elec":
                        subgroup = "elec_eng"
                    elif operator == "photon":
                        subgroup ='photon_eng'
                    elif operator == "blc":
                        subgroup = 'blc_eng'
                    elif operator == "dse":
                        subgroup ='dse_eng'
                    elif operator == "" or operator == "energy":
                        subgroup = "eigenE"
                        
                if norm_state != None:
                    y1.append((f1[group][subgroup][states] - f1[group][subgroup][norm_state]) / omega) 
                    y2.append((f2[group][subgroup][states] - f2[group][subgroup][norm_state]) / omega) 
                else:
                    y1.append(f1[group][subgroup][states])
                    y2.append(f2[group][subgroup][states])

    for j in range(len(states)):
        plt.plot(xs, np.array(y2)[:, j], c='r', linestyle=':', lw=3)
        plt.plot(xs, np.array(y1)[:, j], c='b', lw=1)
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