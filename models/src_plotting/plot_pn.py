import matplotlib.pyplot as plt
import numpy as np
from qutip import expect
from tqdm import tqdm

def plot_aa(systems, state, mini, maxi, step, xlabel = "", title = None, model=""):
    i = 0
    xs = []
    ys = []
    for factor in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        U = systems[i].gen_transform(model)
        operator = U * systems[i].gen_cavity_operators()[0] * U.dag()
        xs.append(factor)
        ys.append(expect(operator, eigenV[state]))
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

def plot_aa_2gx(systems, state, mini, maxi, step, xlabel = "", title = None, model=""):
    xs = []
    y1s = []
    y2s = []
    
    for i, factor in enumerate(tqdm(np.arange(mini, maxi, step))):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        U = systems[i].gen_transform(model)
        operator = U * systems[i].gen_cavity_operators()[0] * U.dag()
        xs.append(factor / systems[i].cavity.freqs[0])
        y1s.append(expect(operator, eigenV[state]))
        y2s.append(systems[i].lambdas_spatial[1][0][-1] / systems[i].lambdas_spatial[0][0][-1])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot First Axis (Photon Number)
    color1 = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"Photon Number $\langle a^\dagger a \rangle$", color=color1)
    ax1.plot(xs, y1s, color=color1, label="Photon Number")
    ax1.tick_params(axis='y', labelcolor=color1)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    
    # Plot Second Axis (Coupling Ratio)
    color2 = 'tab:red'
    ax2.set_ylabel(r"$\lambda(x) / \lambda_0$", color=color2)
    ax2.plot(xs, y2s, color=color2, label="Coupling Ratio", linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title handling
    if not title:
        title = "Photon Number and Coupling for State {}".format(state)
    plt.title(title)
    plt.show()
