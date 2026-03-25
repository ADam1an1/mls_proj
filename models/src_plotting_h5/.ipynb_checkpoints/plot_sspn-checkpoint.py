import matplotlib.pyplot as plt
import numpy as np
from qutip import expect, steadystate
from tqdm import tqdm
import h5py

def plot_ss_op(test_dir, file_prefix, mini, maxi, step, xlabel="", ylabel="", lims=[], operator="", model="", freq_ind=0):
    
    # identify which operator to use            
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
    
    
    xs = []
    ys = []
    for test_factor in tqdm(np.arange(mini, maxi, step)):
        xs.append(test_factor)   
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, test_factor), "r") as f:
            if operator == "pn":
                group = 'photon_numbers'
                subgroup = str(f.attrs['photon_freqs'][freq_ind])
                ys.append(f[group][subgroup][0])
                print(ys)
            else:
                group = 'energies'
                        
                ys.append(f[group][subgroup])
    print(xs, ys)
    plt.plot(xs, np.array(ys))
    plt.title("model")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.show()