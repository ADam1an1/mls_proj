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
        sys_e_levels.sort()
        self.e_levels = sys_e_levels[::-1]
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


    def gen_sys_hamiltonian(self):
        """
        The energy operator for the system
        """
        hamiltonian = qzero(self.dims)
        for i in range(self.nsystems):
            hamiltonian += self.gen_sigmaz(i)
        return hamiltonian 
        
    # TODO:
    # Currently implemented dumbly for TLS (could just use qutip built in functions)
    # Eventually want to expand this to be used for multilevel systems which 
    # would require these functions to construct but with different 
    # internal structure
    
    def gen_jz(self):
        """
        Gives Jz based on the dimension of each system
        and their corresponding energy levels
        (Nori definition)
        Note that this is NOT the energy operator for the system
        """
        jz = qzero(self.dims)
        for i in range(self.nsystems):
            jz += self.gen_sigmaz(i)
        return jz / 2

    def gen_sigmaz(self, sys_ind):
        """
        Gives sigma_z for a TLS
        """
        sigma_z = tensor(*[qdiags(self.systems[sys_ind].e_levels) if sys_ind==j
                      else qeye(self.systems[j].dims)
                      for j in range(self.nsystems)])
        return sigma_z
    
    def gen_jy(self):
        """
        Gives Jy
        (Nori definition)
        """
        jy = qzero(self.dims)
        for i in range(self.nsystems):
            jy += self.gen_sigmay(sys_ind)
        return jy / 2 
    
    def gen_sigmay(self, sys_ind):
        """
        Gives sigma_y for a TLS
        """
        lower_sys_full = self.gen_trans_system(sys_ind, 1, 0)
        raise_sys_full = self.gen_trans_system(sys_ind, 0, 1)
        return raise_sys_full * complex(0, -1) + lower_sys_full * complex(0, 1)

    def gen_jx(self):
        """
        Gives Jx 
        (Nori definition)
        """
        jx = qzero(self.dims)
        for i in range(self.nsystems):
            jx += self.gen_sigmax(sys_ind)
        return jx / 2       
    
    def gen_sigmax(self, sys_ind):
        """
        Gives sigma_x for a TLS
        """
        lower_sys_full = self.gen_trans_system(sys_ind, 1, 0)
        raise_sys_full = self.gen_trans_system(sys_ind, 0, 1)
        return lower_sys_full + raise_sys_full

    def gen_trans_system(self, sys_ind, start, end):
        """
        Gives the transition operator from start -> end states
        """
        sys = self.systems[sys_ind]
        trans_sys = tensor(*[basis(sys.dims, end) * basis(sys.dims, start).dag() 
                             if sys_ind==j
                             else qeye(self.dims[j])
                             for j in range(self.nsystems)])
        return trans_sys


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


