# total.py

import numpy as np
from cavity import Cavity
from systems import MultipartiteSystem, MultilevelSystem
from qutip import qzero, tensor, basis, destroy, num, qeye, sigmaz, Qobj
import math


def calc_spatial_coupling(system_positions, lambdas, photon_freqs):
    """
    Modify the coupling dict taking into account the system positions within
    the photon mode (breaking long wave approximation)
    """
    lambdas_spatial = []
    for system in range(len(system_positions)):
        lambdas_cur_sys = []
        for lamb, freq in zip(lambdas, photon_freqs):
            factor = 1
            if system_positions[system]:
                factor *= np.abs(np.sin(2 * math.pi * system_positions[system] * photon_freqs[n]))
            lambdas_cur_sys.append(factor * np.array(lamb))
        lambdas_spatial.append(lambdas_cur_sys)
    return lambdas_spatial

    
def check_total_params(systems, photon_freqs, max_photon_nums, lambdas, mus, positions):
    """
    Function to check a multipartite system in cavity is properly specified
    """
    # check modes all have a frequency and counts
    if len(photon_freqs) != len(max_photon_nums):
        print("Photon number {} != listed frequencies {}".format(len(photon_freqs), len(max_photon_nums)))
        raise

    # check that each mode has a vector dir
    if len(lambdas) != len(photon_freqs):
        print("Photon number {} != Lambdas {}".format(len(photon_freqs), len(max_photon_nums)))
        raise        
    
    # check that each system has a position
    if len(positions) != len(systems):
        print("Positions {} != Systems {}".format(len(positions), len(systems)))
        raise

    # check that each system has mus
    if len(mus) != len(systems):
        print("Mus {} != Systems {}".format(len(mus), len(systems)))
        raise
    # for mu, system in zip(mus, systems):  

class TotalSystem:
    """
    Class to define a multipartite system coupled to a cavity
    which can be multimode and multiphoton
    """
    def __init__(self, systems, photon_freqs, max_photon_nums,
                 lambdas, mus, positions=[], model=""):
        """
        Creates the class
        
        Args:
            systems, list to specify the systems involved 
                can be a string for molecule
                [system] "name"
                or list of energy levels
                [system][levels]

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
        """
        self.model = model
        if not positions:
            positions = [None] * len(systems)
        self.lambdas_spatial = calc_spatial_coupling(positions, lambdas, photon_freqs)
            
        self.cavity = Cavity(photon_freqs, max_photon_nums)
        self.mp_systems = MultipartiteSystem(systems, self.lambdas_spatial, mus)

        self.total_hamiltonian = qzero(self.cavity.dims + self.mp_systems.dims)
        self.identity = tensor(self.cavity.identity, self.mp_systems.identity)

        # calculate diagonal energy terms (elec + dse + omega)
        self.total_hamiltonian += tensor(self.cavity.identity, self.mp_systems.gen_sys_hamiltonian())
        self.total_hamiltonian += tensor(self.cavity.gen_cavity_hamiltonian(), self.mp_systems.identity)

        # calculate off-diagonal terms
        for mode in range(self.cavity.nmodes):
            ann = self.cavity.gen_ann_op(mode)
            cav_blc = tensor(*[ann + ann.dag() if n==mode
                               else qeye(self.cavity.dims[n])
                               for n in range(self.cavity.nmodes)])
            blc_total = qzero(self.mp_systems.dims)
            for sys_id in range(self.mp_systems.nsystems):
                # term -sqrt(omega/2) lambda cdot mu (a + a dagger)
                lambas_cur_sys = self.lambdas_spatial[sys_id][mode]   
                trans_sys = self.mp_systems.systems[sys_id]
                if isinstance(trans_sys, MultilevelSystem):
                    lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(mode=mode, sqrd=False)
                else:
                    lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
                                                                        lamb=np.linalg.norm(lambas_cur_sys), 
                                                                        sqrd=False)
                prefactor = -np.sqrt(self.cavity.freqs[mode] / 2)
                blc_total += prefactor * tensor([lambda_dot_mu_sys if sys_id==i 
                                                else qeye(self.mp_systems.dims[i])
                                                for i in range(self.mp_systems.nsystems)])                    
            self.total_hamiltonian += tensor(cav_blc, blc_total)

    def gen_cavity_operators(self, mode=None):
        """
        Gives operators to get photon number for every photon mode
        """
        operators = []
        if not mode:
            modes = [_ for _ in range(self.cavity.nmodes)]   
        for mode in modes:
            ann_cav = tensor(*[qeye(self.cavity.dims[m]) if mode != m
                                    else destroy(self.cavity.dims[mode])
                                    for m in range(self.cavity.nmodes)], self.mp_systems.identity)
            # if "minus" in self.model or "plus" in self.model:
            #     ann_cav = self.gen_ann_shift(ann_cav)
            num_op = ann_cav.dag() * ann_cav
                           
            operators.append(num_op)          
        
        return operators

    def gen_ann_shift(self, ann_cav):
        # a -> a' = a + i sum_i (eta * sigma_x^i)
        # note that we rotate by Udag a U = ia, Udag adag U = -iadag 
        # a -> a' = ia - wc sum A_i cdot mu_i
        # a -> a' = ia - sum sqrt(w_c/2) lambda cdot mu 
        # ann_cav *= complex(0, 1)
        # for mode in range(self.cavity.nmodes):
        #     prefactor =  1 / np.sqrt(2 * self.cavity.freqs[mode])
        #     for sys_id in range(self.mp_systems.nsystems):
        #         trans_sys = self.mp_systems.systems[sys_id]
        #         if isinstance(trans_sys, MultilevelSystem):
        #             lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(mode=mode, sqrd=False)
        #         else:
        #             lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
        #                                                                 lamb=np.linalg.norm(lambdas_cur_sys), 
        #                                                                 sqrd=False)
        #         lambda_dot_mu_total = tensor(*[lambda_dot_mu_sys if sys_id==j 
        #                                       else qeye(self.mp_systems.dims[j])
        #                                       for j in range(self.mp_systems.nsystems)])
        #         ann_cav -= prefactor * tensor(self.cavity.identity, lambda_dot_mu_total)
        #         print( prefactor * tensor(self.cavity.identity, lambda_dot_mu_total))
        return ann_cav   

    def gen_total_state(self, systems_dirac, photons_dirac):
        """
        Gives the ket for dirac representations
        """
        systems_part = self.mp_systems.gen_systems_state(systems_dirac)
        photons_part = self.cavity.gen_cavity_state(photons_dirac)
        return tensor(photons_part, systems_part)

    # def gen_pzw(self):
    #     # TODO EDIT TO APPLY FOR CI MODEL           
    #     terms = []
    #     for mode in range(self.cavity.nmodes):
    #         sigma_term = qzero(self.mp_systems.dims)
    #         ann_cav = tensor([qeye(self.cavity.dims[m]) if mode != m
    #                                 else destroy(self.cavity.dims[mode])
    #                                 for m in range(self.cavity.nmodes)])
    #         for sys_id in range(self.mp_systems.nsystems):
    #             trans_sys_i = self.mp_systems.systems[sys_id]
    #             for level_start_i in range(trans_sys_i.dims):
    #                 for level_end_i in range(level_start_i + 1, trans_sys_i.dims):
    #                     coupling_g_i = self.couplings_dict[mode][sys_id][level_start_i][level_end_i - level_start_i - 1]
    #                     sigma_x_i = self.mp_systems.gen_sigmax(sys_id)
    #                     sigma_term -= complex(0, 1) * coupling_g_i * sigma_x_i 
    #         terms.append(tensor(ann_cav.dag() + ann_cav, sigma_term))
    #     U = sum(terms).expm()
    #     return U

    
    def gen_joint_operator(self, systems_dirac, photons_dirac, correction=False):
        # if correction:
        #     corrected_op = 
        #     return corrected_op 
        return self.gen_total_state(systems_dirac, photons_dirac) * self.gen_total_state(systems_dirac, photons_dirac).dag()

    def gen_joint_label(self, systems_dirac, photons_dirac):
        """
        Gives string for the operators 
        """
        return r"$|{}\rangle |{}\rangle$".format(str(systems_dirac)[1:-1], str(photons_dirac)[1:-1])

        