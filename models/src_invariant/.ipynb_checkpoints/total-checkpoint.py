# total.py

import numpy as np
from cavity import Cavity
from systems import MultipartiteSystem
from qutip import qzero, tensor, basis, destroy, num, qeye, sigmaz

class TotalSystem:
    """
    Class to define a multipartite system coupled to a cavity
    which can be multimode and multiphoton
    """

    def __init__(self, system_e_levels, photon_freqs, max_photon_nums,
                 couplings_dict, system_starts, photon_starts, model=""):
        """
        Creates the class
        
        Args:
            system_e_levels, list to specify the energy of levels in 
                a system. 
                [system][level]

            photon_freqs, list of cavity modes
                [mode1, mode2, ...]

            max_photon_nums, list of the maximum photons allowed in
                a mode. 
                [max pnum mode1, max pnum mode2, ...]
            
            couplings_dict, dictionary containing all the coupling
                strengths between a mode and a systems energy level
                separations. 
                {mode1: 
                    {system1: 
                        {elevel1: 
                            {elevel0: coupling strength}
                        }
                    }
                }
            system_starts, list of energy level that system starts in
                [system1 state, ...]

            photon_starts, list of photon numbers to start in
                [mode1 pnum, ...]
        """
        self.model = model
        self.couplings_dict = couplings_dict
        self.cavity = Cavity(photon_freqs, max_photon_nums)
        self.mp_systems = MultipartiteSystem(system_e_levels)

        self.total_hamiltonian = qzero(self.cavity.dims + self.mp_systems.dims)
        self.identity = tensor(self.cavity.identity, self.mp_systems.identity)

        # calculate diagonal multipartite system energy terms
        if model == "tc" or model == "dicke" or "invariant_dipole" in model:
            self.total_hamiltonian += tensor(self.cavity.identity, self.mp_systems.gen_sys_hamiltonian())

        # calculate diagonal cavity energy terms
        self.total_hamiltonian += tensor(self.cavity.gen_cavity_hamiltonian(), self.mp_systems.identity)

        # calculate off-diagonal terms
        # we assume that there are no terms in which relaxation between
        # different systems is coupled i.e. one raised and one lowered without photon mediation
        # note that this does not mean that only one changes at a time
        # we also only consider one mode annihilation/creation events at a time
        # this does not mean that only one mode changes at a time
        # multiphoton events in a single mode also occur
        self.lamb_mu_sqrd = qzero(self.mp_systems.dims)
        for mode in range(self.cavity.nmodes):
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys = self.mp_systems.systems[sys_id]

                # iterate over all possible energy level transitions for a system
                # levels are sorted in order of highest to lowest
                for level_start in range(trans_sys.dims):
                    for level_end in range(level_start + 1, trans_sys.dims): 
                        
                        coupling_g = couplings_dict[mode][sys_id][level_start][level_end - level_start - 1]

                        # get transition terms and add to hamiltonian
                        down_system = tensor(self.cavity.identity, self.mp_systems.gen_trans_system(sys_id, level_start, level_end))
                        sigma_x = tensor(self.cavity.identity, self.mp_systems.gen_sigmax(sys_id))
                        ann_cavity = tensor(self.cavity.gen_ann_op(mode), self.mp_systems.identity)
                 
                        if model == "jc":
                            self.total_hamiltonian += coupling_g * ann_cavity.dag() * down_system
                            self.total_hamiltonian += coupling_g * ann_cavity * down_system.dag()
                        
                        if model == "dicke":
                            self.total_hamiltonian += coupling_g * (ann_cavity.dag() + ann_cavity) * sigma_x

                        if "invariant_dipole" in model:
                            # we take coupling_g to be eta in Nori's paper instead (eta = g_D/omega_c = g_C/omega_10) 
                            omega_c = self.cavity.freqs[mode]
                            self.total_hamiltonian += complex(0, omega_c) * coupling_g * (ann_cavity.dag() - ann_cavity) * sigma_x
                            # print(complex(0, omega_c) * coupling_g * (ann_cavity.dag() - ann_cavity) * sigma_x)
                            
                            for sys_j in range(sys_id + 1, self.mp_systems.nsystems):
                                trans_sys_j = self.mp_systems.systems[sys_j]
                                for level_start_j in range(trans_sys_j.dims):
                                    for level_end_j in range(level_start_j + 1, trans_sys_j.dims):
                                        coupling_g_j = couplings_dict[mode][sys_j][level_start_j][level_end_j - level_start_j - 1]
                                        sigma_x_j = tensor(self.cavity.identity, self.mp_systems.gen_sigmax(sys_j))
                                        self.total_hamiltonian += 2 * omega_c * coupling_g * coupling_g_j * sigma_x * sigma_x_j
                                        self.lamb_mu_sqrd += omega_c * coupling_g * coupling_g_j * self.mp_systems.gen_sigmax(sys_id) * self.mp_systems.gen_sigmax(sys_j)

                        if model == "invariant_coulomb":
                            delta_e_sys = np.abs(self.mp_systems.get_sys_energy(sys_id, level_start) - self.mp_systems.get_sys_energy(sys_id, level_end))
                            
                            # using the shifted operator sigma'z does not work correctly
                            # energies scale incorrectly leading to errors so use standard sigma z def
                            sigma_z = tensor(self.cavity.identity, 
                                             tensor(*[sigmaz() if sys_id == j 
                                                      else qeye(self.mp_systems.systems[j].dims)
                                                      for j in range(self.mp_systems.nsystems)]))
                            sigma_y = tensor(self.cavity.identity, self.mp_systems.gen_sigmay(sys_id))
                            cos_part = (2 * coupling_g * (ann_cavity + ann_cavity.dag())).cosm()
                            sin_part = (2 * coupling_g * (ann_cavity + ann_cavity.dag())).sinm()
                            self.total_hamiltonian += 0.5 * delta_e_sys * sigma_z * cos_part
                            self.total_hamiltonian += 0.5 * delta_e_sys * sigma_y * sin_part               
                           
                            #     sigma_z = tensor(self.cavity.identity, self.mp_systems.gen_sigmaz(sys_id))
                            #     self.total_hamiltonian += sigma_z * cos_part
                            #     self.total_hamiltonian += 0.5 * delta_e_sys * sigma_y * sin_part                     

    def gen_cavity_operators(self):
        """
        Gives operators to get photon number for every photon mode
        """
        operators = []
        
        for mode in range(self.cavity.nmodes):
            ann_cav = tensor(*[qeye(self.cavity.dims[m]) if mode != m
                                    else destroy(self.cavity.dims[mode])
                                    for m in range(self.cavity.nmodes)], self.mp_systems.identity)
            
            if "minus" in self.model or "plus" in self.model:
                ann_cav = self.gen_ann_shift(ann_cav)

            num_op = ann_cav.dag() * ann_cav

            if "pzw" in self.model:
                u = self.gen_pzw()
                # num_op = u * num_op * u.dag()
                num_op = u.dag() * num_op * u
            
            operators.append(num_op)            
        return operators

    def gen_ann_shift(self, ann_cav):
        # a -> a' = a + i sum_i (g^i * sigma_x^i)
        # ann_cav = tensor(self.cavity.gen_ann_op(mode), self.mp_systems.identity)
        # we take coupling_g to be eta in Nori's paper instead (eta = g_D/omega_c = g_C/omega_10) 
        for mode in range(self.cavity.nmodes):
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys_i = self.mp_systems.systems[sys_id]
                for level_start_i in range(trans_sys_i.dims):
                    for level_end_i in range(level_start_i + 1, trans_sys_i.dims):
                        coupling_g_i = self.couplings_dict[mode][sys_id][level_start_i][level_end_i - level_start_i - 1]
                        sigma_x_i = tensor(self.cavity.identity, self.mp_systems.gen_sigmax(sys_id))
                        
                        if "minus" in self.model:
                            ann_cav -= complex(0, 1) * coupling_g_i * sigma_x_i
                        if "plus" in self.model: # this is the correct version with no change to coulomb
                            ann_cav += complex(0, 1) * coupling_g_i * sigma_x_i
        return ann_cav   

    def gen_pzw(self):
        # gives the power zienau woolley 
        # tranformation unitary transformation
        # U = exp(-i eta (a + a.dag() sigmax)
        terms = []
        for mode in range(self.cavity.nmodes):
            sigma_term = qzero(self.mp_systems.dims)
            ann_cav = tensor([qeye(self.cavity.dims[m]) if mode != m
                                    else destroy(self.cavity.dims[mode])
                                    for m in range(self.cavity.nmodes)])
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys_i = self.mp_systems.systems[sys_id]
                for level_start_i in range(trans_sys_i.dims):
                    for level_end_i in range(level_start_i + 1, trans_sys_i.dims):
                        coupling_g_i = self.couplings_dict[mode][sys_id][level_start_i][level_end_i - level_start_i - 1]
                        sigma_x_i = self.mp_systems.gen_sigmax(sys_id)
                        sigma_term += complex(0, 1) * coupling_g_i * sigma_x_i 
            terms.append(tensor(ann_cav.dag() + ann_cav, sigma_term))
        U = sum(terms).expm()
        return U

    def gen_total_state(self, systems_dirac, photons_dirac):
        """
        Gives the ket for dirac representations
        """
        systems_part = self.mp_systems.gen_systems_state(systems_dirac)
        photons_part = self.cavity.gen_cavity_state(photons_dirac)
        return tensor(photons_part, systems_part)

    def gen_joint_operator(self, systems_dirac, photons_dirac):   
        state = self.gen_total_state(systems_dirac, photons_dirac)
        proj = state * state.dag()
        if "pzw" in self.model:
            u = self.gen_pzw()
            proj = u.dag() * proj * u
        return proj

    def gen_joint_label(self, systems_dirac, photons_dirac):
        """
        Gives string for the operators 
        """
        return r"$|{}\rangle |{}\rangle$".format(str(systems_dirac)[1:-1], str(photons_dirac)[1:-1])
