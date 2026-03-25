import matplotlib.pyplot as plt
import numpy as np
from qutip import expect, steadystate
from tqdm import tqdm

def plot_aa_sep(systems, mini, maxi, step, gamma, kappa):
    xs = np.arange(mini, maxi, step) / systems[0].cavity.freqs[0]
    ys = []
    for ind in tqdm(range(len(xs))):
        total_system = systems[ind]
        jump_ops = []
        jump_ops.extend(total_system.gen_gamma_losses(gamma))
        jump_ops.extend(total_system.gen_kappa_losses(kappa))
        rho_ss = steadystate(total_system.total_hamiltonian, jump_ops)
        num_op = total_system.gen_cavity_operators()
        ys.extend(expect(num_op, rho_ss))
    plt.plot(xs, np.array(ys)/ys[0], label="n2 / n1")
    plt.xlabel("Distance d / $\lambda$")
    plt.ylabel("Cavity photon number n2/n1")    
    plt.show()