import matplotlib.pyplot as plt
import numpy as np
from qutip import expect
from tqdm import tqdm

def plot_aa(systems, state, mini, maxi, step, xlabel = "", title = None):
    i = 0
    xs = []
    ys = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        operator = systems[i].gen_cavity_operators()
        xs.append(coupling)
        ys.append(expect(operator[0], eigenV[state]))
        i += 1
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel("Photon Number <a dagger a>")
    if title:
        title = title
    else:
        title = "Photon Number for {} State".format(state)
    plt.title(title)
    plt.show()
