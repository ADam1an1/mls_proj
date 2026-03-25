# systems.py

import numpy as np
from qutip import Qobj, tensor, qeye, basis, qdiags, qzero, basis

class MultilevelSystem:
    """
    Class to define one multilevel system
    """
    def __init__(self, sys_e_levels):
        """
        Make class with specified energy levels
        """
        self.e_levels = sys_e_levels
        self.dims = len(sys_e_levels)

    def gen_sys_state(self, sys_dirac):
        """
        Gives ket for system 
        """
        return basis(self.dims, sys_dirac)


class MultipartiteSystem:
    """
    Class to define a set of multilevel systems
    """
    def __init__(self, systems_e_levels):
        """
        Makes as many systems as needed 
        with specified energy levels
        """
        self.nsystems = len(systems_e_levels)
        self.systems = []
        for i in range(self.nsystems):
            self.systems.append(MultilevelSystem(systems_e_levels[i]))

        self.dims = [self.systems[i].dims for i in range(self.nsystems)]
        self.identity = tensor(*[qeye(dim) for dim in self.dims])

    def get_sys_energy(self, sys_id, level_id):
        """
        Gives the energy of a system level
        """
        return self.systems[sys_id].e_levels[level_id]

    def gen_systems_state(self, systems_dirac):
        """
        Gives the ket for all system components from dirac notation
        """
        systems_state = tensor(*[self.systems[i].gen_sys_state(systems_dirac[i])
                                for i in range(self.nsystems)])
        return systems_state

    def gen_systems_hamiltonian(self):
        """
        Gives hamiltonian based on the dimension of each system
        and their corresponding energy levels
        """
        total_hamiltonian = qzero(self.dims)
        for i in range(self.nsystems):
            cur_sys = qdiags(self.systems[i].e_levels)
            total_hamiltonian += tensor(*[cur_sys if i==j
                                          else qeye(self.systems[j].dims)
                                          for j in range(self.nsystems)])
        return total_hamiltonian

    def gen_systems_operators(self, photon_dims):
        """
        Gives operators to track population of every systems' energy levels
        """
        operators = []
        labels = []
        for i in range(self.nsystems):
            for level in range(self.systems[i].dims):
                cur_sys = basis(self.systems[i].dims, level) * basis(self.systems[i].dims, level).dag()
                operators.append(tensor(*[cur_sys if j==i
                                          else qeye(self.systems[j].dims)
                                          for j in range(self.nsystems)], qeye(photon_dims)))

        return operators


