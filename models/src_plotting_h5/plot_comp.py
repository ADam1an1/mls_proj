import numpy as np
import matplotlib.pyplot as plt
from qutip import expect
from tqdm import tqdm
import h5py

def plot_state_comp(test_dir, file_prefix, state, op_list, mini, maxi, step, xlabel="", model=""):
    xs = []
    ys = []
    for coupling in tqdm(np.arange(mini, maxi, step)):
        xs.append(coupling)
        with h5py.File("{}/{}_{:.3f}".format(test_dir, file_prefix, coupling), "r") as f:
            ys.append(f['projections'][str(state)][op_list])
            op_desc = f['projections'].attrs['op_desc']
                
    for j in range(len(op_list)):
        plt.plot(xs, np.array(ys)[:, j], label=op_desc[j])

    plt.xlabel(xlabel)
    plt.title("Compositions of Eigenstate {}, model = {}".format(state, model))
    plt.ylabel("Projection Magnitude")
    plt.legend()
    plt.show()

# hasnt been translated for new file system
# def plot_state_comp_rev(systems, state_dirac, eigs, mini, maxi, step, xlabel="", model=""):
#     i = 0
#     xs = []
#     ys = []
#     state = systems[0].gen_total_state(state_dirac[0], state_dirac[1])
#     for coupling in tqdm(np.arange(mini, maxi, step)):
#         hamiltonian = systems[i].total_hamiltonian
#         eigenE, eigenV = hamiltonian.eigenstates()
#         xs.append(coupling)
#         proj_operators = [eigenV[eig] * eigenV[eig].dag() for eig in eigs]
#         ys.append([expect(proj_op, state) for proj_op in proj_operators])
#         i += 1

#     for j in range(len(eigs)):
#         plt.plot(xs, np.array(ys).T[j], label=eigs[j])
#     # plt.legend(ncols = len(states) // 5)
#     plt.xlabel(xlabel)
#     plt.title("Rediagonalization of State {}, model = {}".format(state_dirac, model))
#     plt.ylabel("Projection Magnitude")
#     plt.legend()
#     plt.show()

