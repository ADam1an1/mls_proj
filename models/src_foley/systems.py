# systems.py

import numpy as np
from qutip import Qobj, tensor, qeye, basis, qdiags, qzero, basis, destroy
from scipy.linalg import sqrtm
import os

class Molecule:
    """
    Class to define a molecule from psi4 calc
    NOTE: THIS MUST BE PRECOMPUTED
    """
    def __init__(self, molecule, lambdas, filepath):
        self.file = filepath
        self.molecule = molecule
        if self.molecule == "H2":
            self.dims = 4
        self.lambdas = lambdas
        self.lambda_eff = np.round(np.sqrt(np.sum([lamb **2 for lamb in lambdas])), decimals=3)
        
        hpf_elec = np.load(self.gen_file_path(0, 0, "HPF"))
        self.elec = Qobj(hpf_elec[:self.dims, :self.dims])
        
        hpf_elecdse = np.load(self.gen_file_path(0, self.lambda_eff, "HPF"))
        self.elecdse = Qobj(hpf_elecdse[:self.dims, :self.dims])
        self.dse = Qobj(hpf_elecdse[:self.dims, :self.dims] - hpf_elec[:self.dims, :self.dims])

    def gen_file_path(self, freq, lamb, d_info):
        dir_part = self.file + "/{}_{}/{}_{}_{:.3f}_sto3g_hpf/".format(self.molecule, freq, self.molecule, freq, lamb)
        file_path = dir_part + "{}_{}_{:.3f}_sto3g_hpf_{}.npy".format(self.molecule, freq, lamb, d_info)
        if os.path.exists(file_path):
            return file_path
        else:
            dir_part = self.file + "/{}_{}/{}_{}_{:.2f}0_sto3g_hpf/".format(self.molecule, freq, self.molecule, freq, lamb)
            file_path = dir_part + "{}_{}_{:.2f}0_sto3g_hpf_{}.npy".format(self.molecule, freq, lamb, d_info)
            return file_path
        
    def gen_lambdadotmu_terms(self, freq=0, lamb=0, sqrd=False):
        if sqrd:
            # returns lambda dot mu part of 1/2 (lambda dot mu)^2
            return Qobj(2 * self.dse).sqrtm()

        # returns -sqrt(omega / 2) lambda dot mu
        return Qobj(np.load(self.gen_file_path(freq, lamb, "G")))

    def gen_sys_state(self, mol_dirac):
        if self.molecule == "H2":
            li = [0, 2, 5, 6]
        CI_eg_vecs = np.load(self.gen_file_path(0, 0, "Vecs"))
        CI_eg_vec = CI_eg_vecs[:self.dims, li[mol_dirac]]
        return Qobj(CI_eg_vec) 

class MultilevelSystem:
    """
    Class to define one multilevel system
    """
    def __init__(self, sys_e_levels, lambdas, mus):
        """
        Make class with specified energy levels
        """
        # NOTE: ENERGY LEVELS ARE LOWEST TO HIGHEST
        sys_e_levels.sort()
        self.e_levels = sys_e_levels
        self.dims = len(sys_e_levels)
        self.lambdadotmus = [Qobj(np.einsum('i,ijk->jk', lambda_mode, mus)) for lambda_mode in lambdas]
        self.elec = qdiags(self.e_levels)

    def gen_sys_state(self, sys_dirac):
        """
        Gives ket for system 
        """
        return basis(self.dims, sys_dirac)


class MultipartiteSystem:
    """
    Class to define a set of multilevel systems
    """
    def __init__(self, systems, lambdas, mus, positions, filepath="/scratch/avd383/qed-ci/H2_chain"):
        """
        Makes as many systems as needed 
        with specified energy levels
        """
        self.nsystems = len(systems)
        if not positions:
            positions = [None] * self.nsystems
        self.systems = []
        for sys_id in range(self.nsystems):
            if isinstance(systems[sys_id], str):
                self.systems.append(Molecule(systems[sys_id], lambdas[sys_id], filepath))
            else:
                self.systems.append(MultilevelSystem(systems[sys_id], 
                                                 lambdas[sys_id], mus[sys_id]))
        self.dims = [self.systems[sys_id].dims for sys_id in range(self.nsystems)]
        self.identity = tensor(*[qeye(dim) for dim in self.dims])


    def gen_system_lower(self, sys_id):
        system_lower = tensor(*[destroy(self.dims[sys_id]) if sys_id==cur_id
                                else qeye(self.dims[cur_id])
                                for cur_id in range(self.nsystems)])
        return system_lower
        

    def gen_sys_hamiltonian(self):
        """
        The energy operator for the system
        """
        hamiltonian = qzero(self.dims)
        for i in range(self.nsystems):
            hamiltonian += tensor(*[self.systems[i].elec if i==j
                                    else qeye(self.dims[j]) 
                                    for j in range(self.nsystems)])
        return hamiltonian 
        
    def gen_systems_state(self, systems_dirac):
        """
        Gives the ket for all system components from dirac notation
        if taking identity use *
        """
        systems_state = tensor(*[self.systems[sys_di].gen_sys_state(systems_dirac[sys_di])
                                 if isinstance(systems_dirac[sys_di], int) 
                                 else qeye(self.dims[sys_di])
                                 for sys_di in range(self.nsystems)])
        return systems_state

    def gen_sys_state_op(self, systems_dirac):
        """
        Gives operators to track a 
        particular system state 
        """
        return self.gen_systems_state(systems_dirac) * self.gen_systems_state(systems_dirac).dag()


