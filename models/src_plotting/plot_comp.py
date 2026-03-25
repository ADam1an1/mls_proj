import numpy as np
import matplotlib.pyplot as plt
from qutip import expect
from tqdm import tqdm

def plot_state_comp(systems, state, op_list, mini, maxi, step, xlabel="", model=""):
    i = 0
    xs = []
    ys = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        proj_operators = []
        for sys_dirac, cav_dirac in op_list:
            proj_operators.append(systems[i].gen_joint_operator(sys_dirac, cav_dirac))
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)
        ys.append([expect(proj_op, eigenV[state]) for proj_op in proj_operators])
        i += 1

    op_labels = [systems[0].gen_joint_label(sys_dirac, cav_dirac)
                 for sys_dirac, cav_dirac in op_list]

    for j in range(len(op_list)):
        plt.plot(xs, np.array(ys).T[j], label=op_labels[j])
    # plt.legend(ncols = len(states) // 5)
    plt.xlabel(xlabel)
    plt.title("Compositions of Eigenstate {}, model = {}".format(state, model))
    plt.ylabel("Projection Magnitude")
    plt.legend()
    plt.show()

def plot_state_comp_rev(systems, state_dirac, eigs, mini, maxi, step, xlabel="", model=""):
    i = 0
    xs = []
    ys = []
    state = systems[0].gen_total_state(state_dirac[0], state_dirac[1])
    for coupling in tqdm(np.arange(mini, maxi, step)):
        hamiltonian = systems[i].total_hamiltonian
        eigenE, eigenV = hamiltonian.eigenstates()
        xs.append(coupling)
        proj_operators = [eigenV[eig] * eigenV[eig].dag() for eig in eigs]
        ys.append([expect(proj_op, state) for proj_op in proj_operators])
        i += 1

    for j in range(len(eigs)):
        plt.plot(xs, np.array(ys).T[j], label=eigs[j])
    # plt.legend(ncols = len(states) // 5)
    plt.xlabel(xlabel)
    plt.title("Rediagonalization of State {}, model = {}".format(state_dirac, model))
    plt.ylabel("Projection Magnitude")
    plt.legend()
    plt.show()

