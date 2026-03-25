# systems.py

import numpy as np
from qutip import Qobj, tensor, qeye, basis, qdiags, qzero, basis
from scipy.linalg import sqrtm

class Molecule:
    """
    Class to define a molecule from psi4 calc
    NOTE: THIS MUST BE PRECOMPUTED
    """
    def __init__(self, molecule, lambdas, file_path="/scratch/avd383/qed-ci/H2_chain"):
        self.file = file_path
        self.molecule = molecule
        if self.molecule == "H2":
            self.dims = 4
        self.lambdas = lambdas
        self.lambda_eff = np.round(np.sqrt(np.sum([lamb **2 for lamb in lambdas])), decimals=2)
        
        hpf_elec = np.load(self.gen_file_path(0, 0, "HPF"))
        self.elec = hpf_elec[:self.dims, :self.dims]
        
        hpf_elecdse = np.load(self.gen_file_path(0, self.lambda_eff, "HPF"))
        self.elecdse = Qobj(hpf_elecdse[:self.dims, :self.dims])
        self.dse = np.subtract(hpf_elecdse[:self.dims, :self.dims], hpf_elec[:self.dims, :self.dims])

    def gen_file_path(self, freq, _lambda, d_info):
        dir_part = self.file + "/{}_{}/{}_{}_{:.3f}_sto3g_hpf/".format(self.molecule, freq, self.molecule, freq, _lambda)
        return dir_part + "{}_{}_{:.3f}_sto3g_hpf_{}.npy".format(self.molecule, freq, _lambda, d_info)

    def gen_lambdadotmu_terms(self, freq=0, lamb=0, sqrd=False):
        if sqrd:
            return Qobj(2 * self.dse)          
        G = np.load(self.gen_file_path(freq, lamb, "G"))
        return Qobj(-G * np.sqrt(2/freq))

    def gen_sys_state(self, mol_dirac):
        if self.molecule == "H2":
            if mol_dirac == 1:
                mol_dirac = 2
            elif mol_dirac == 2:
                mol_dirac = 5
            elif mol_dirac == 3:
                mol_dirac = 6
        CI_eg_vecs = np.load(self.gen_file_path(0, 0, "Vecs"))
        return Qobj(CI_eg_vecs[:self.dims, mol_dirac]) 

class MultilevelSystem:
    """
    Class to define one multilevel system
    """
    def __init__(self, sys_e_levels, lambdas, mus, diag=True):
        """
        Make class with specified energy levels
        """
        sys_e_levels.sort() # lowest to highest energy levels
        self.diag = diag
        self.dims = len(sys_e_levels)
        self.lambdadotmus = [np.einsum('i,ijk->jk', l_freq, mus) for l_freq in lambdas]
        if diag:
            self.elecdse = qdiags(sys_e_levels)
        # add else for diagonalized
            

    def gen_sys_state(self, sys_dirac):
        """
        Gives ket for system 
        """
        return basis(self.dims, sys_dirac)

    def gen_lambdadotmu_terms(self, mode, sqrd=False):
        ldm = Qobj(self.lambdadotmus[mode])
        if sqrd and not self.diag:
            return ldm ** 2
        elif sqrd:
            return qzero(self.dims)
        return ldm


class MultipartiteSystem:
    """
    Class to define a set of multilevel systems
    """
    def __init__(self, systems, lambdas, mus):
        """
        Makes as many systems as needed 
        with specified energy levels
        """
        self.nsystems = len(systems)
        self.systems = []
        for i in range(self.nsystems):
            if isinstance(systems[i], str):
                self.systems.append(Molecule(systems[i], lambdas[i]))
            else:
                self.systems.append(MultilevelSystem(systems[i], lambdas[i], mus[i]))

        self.dims = [self.systems[i].dims for i in range(self.nsystems)]
        self.identity = tensor(*[qeye(dim) for dim in self.dims])

    def gen_sys_hamiltonian(self):
        """
        The energy operator for the system
        """
        hamiltonian = qzero(self.dims)
        for i in range(self.nsystems):
            hamiltonian += tensor(*[self.systems[i].elecdse if i==j
                                    else qeye(self.dims[j]) 
                                    for j in range(self.nsystems)])
        return hamiltonian 

    def gen_trans_state(self, start_dirac, end_dirac):
        """
        Gives the transition operator from start -> end states
        """
        trans_sys = self.gen_systems_state(end_dirac) * self.gen_systems_state(start_dirac).dag() 
        return trans_sys

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

    # TODO:
    # Currently implemented dumbly for TLS (could just use qutip built in functions)
    # Eventually want to expand this to be used for multilevel systems which 
    # would require these functions to construct but with different 
    # internal structure

    # def get_sys_energy(self, sys_id, level_id):
    #     """
    #     Gives the energy of a system level
    #     """
    #     return self.systems[sys_id].e_levels[level_id]
    
    # def gen_jz(self):
    #     """
    #     Gives Jz based on the dimension of each system
    #     and their corresponding energy levels
    #     (Nori definition)
    #     Note that this is NOT the energy operator for the system
    #     """
    #     jz = qzero(self.dims)
    #     for i in range(self.nsystems):
    #         jz += self.gen_sigmaz(i)
    #     return jz / 2

    # def gen_sigmaz(self, sys_ind):
    #     """
    #     Gives sigma_z for a TLS
    #     """
    #     sigma_z = tensor(*[qdiags(self.systems[sys_ind].e_levels) if sys_ind==j
    #                   else qeye(self.systems[j].dims)
    #                   for j in range(self.nsystems)])
    #     return sigma_z
    
    # def gen_jy(self):
    #     """
    #     Gives Jy
    #     (Nori definition)
    #     """
    #     jy = qzero(self.dims)
    #     for i in range(self.nsystems):
    #         jy += self.gen_sigmay(sys_ind)
    #     return jy / 2 
    
    # def gen_sigmay(self, sys_ind):
    #     """
    #     Gives sigma_y for a TLS
    #     """
    #     lower_sys_full = self.gen_trans_system(sys_ind, 1, 0)
    #     raise_sys_full = self.gen_trans_system(sys_ind, 0, 1)
    #     return raise_sys_full * complex(0, -1) + lower_sys_full * complex(0, 1)

    # def gen_jx(self):
    #     """
    #     Gives Jx 
    #     (Nori definition)
    #     """
    #     jx = qzero(self.dims)
    #     for i in range(self.nsystems):
    #         jx += self.gen_sigmax(sys_ind)
    #     return jx / 2       
    
    # def gen_sigmax(self, sys_ind):
    #     """
    #     Gives sigma_x for a TLS
    #     """
    #     lower_sys_full = self.gen_trans_system(sys_ind, 1, 0)
    #     raise_sys_full = self.gen_trans_system(sys_ind, 0, 1)
    #     return lower_sys_full + raise_sys_full

    # def gen_trans_system(self, sys_ind, start, end):
    #     """
    #     Gives the transition operator from start -> end states
    #     """
    #     sys = self.systems[sys_ind]
    #     trans_sys = tensor(*[basis(sys.dims, end) * basis(sys.dims, start).dag() 
    #                          if sys_ind==j
    #                          else qeye(self.dims[j])
    #                          for j in range(self.nsystems)])
    #     return trans_sys

    # def gen_jraise(self):
    #     jraise = qzero(self.dims)
    #     for i in range(self.nsystems):
    #         # going from lower energy to higher
    #         for level_start in range(self.systems[i].e_levels):
    #             for level_end in range(level_start + 1, self.systems[i].e_levels):
    #                 cur_trans = self.gen_trans_system(sys_ind, level_start, level_end)
    #                 jraise += tensor(*[cur_trans if i==j
    #                               else qeye(self.systems[j].dims)
    #                               for j in range(self.nsystems)])
    #     return jraise

    # def gen_jlower(self):
    #     jlower = qzero(self.dims)
    #     for i in range(self.nsystems):
    #         # going from higher energy to lower
    #         for level_start in range(self.systems[i].e_levels):
    #             for level_end in range(level_start):
    #                 cur_trans = self.gen_trans_system(sys_ind, level_start, level_end)
    #                 jlower += tensor(*[cur_trans if i==j
    #                               else qeye(self.systems[j].dims)
    #                               for j in range(self.nsystems)])
    #     return jlower

    


