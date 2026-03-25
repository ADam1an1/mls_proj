# total.py

import numpy as np
from cavity import Cavity
from systems import *
from qutip import qzero, tensor, basis, destroy, num, qeye, sigmaz, Qobj, qdiags

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
            if system_positions[system] != None:
                # magnitude of field sin(2pi k x)
                factor *= np.abs(np.sin(2 * np.pi * system_positions[system] * freq))
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
        print("Photon number {} != Lambdas {}".format(len(photon_freqs), len(lambdas)))
        raise        
    
    # check that each system has a position
    if len(positions) != len(systems):
        print("Positions {} != Systems {}".format(len(positions), len(systems)))
        raise


class TotalSystem:
    """
    Class to define a multipartite system coupled to a cavity
    which can be multimode and multiphoton
    """

    def __init__(self, systems, photon_freqs, max_photon_nums,
                 lambdas, mus, positions=[], model="", filepath="/scratch/avd383/qed-ci/H2_chain"):
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
        if not positions:
            positions = [None for _ in range(len(systems))]
        check_total_params(systems, photon_freqs, max_photon_nums, lambdas, mus, positions)
        
        self.lambdas_spatial = calc_spatial_coupling(positions, lambdas, photon_freqs) 
        self.cavity = Cavity(photon_freqs, max_photon_nums)
        self.mp_systems = MultipartiteSystem(systems, self.lambdas_spatial, mus, positions, filepath)
        self.identity = tensor(self.cavity.identity, self.mp_systems.identity)

        # calculate diagonal multipartite system energy terms
        self.total_elec = tensor(self.cavity.identity, self.mp_systems.gen_sys_hamiltonian())
        self.total_photon = tensor(self.cavity.gen_cavity_hamiltonian(), self.mp_systems.identity)

        # calculate off-diagonal terms
        # we assume that there are no terms in which relaxation between
        # different systems is coupled i.e. one raised and one lowered without photon mediation
        # note that this does not mean that only one changes at a time
        # we also only consider one mode annihilation/creation events at a time
        # this does not mean that only one mode changes at a time
        # multiphoton events in a single mode also occur
        self.total_dse = qzero(self.cavity.dims + self.mp_systems.dims)
        self.total_blc = qzero(self.cavity.dims + self.mp_systems.dims)
        for mode in range(self.cavity.nmodes):
            ann = self.cavity.gen_ann_op(mode)
            cav_blc = ann + ann.dag()
            lambda_dot_mu_total = qzero(self.mp_systems.dims)
            lambda_dot_mu_sqrd_total = qzero(self.mp_systems.dims)
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys = self.mp_systems.systems[sys_id]
                lambdas_cur_sys = self.lambdas_spatial[sys_id][mode]
                prefactor = -np.sqrt(self.cavity.freqs[mode] / 2)
                if isinstance(trans_sys, MultilevelSystem):
                    lambda_dot_mu_sys = prefactor * self.mp_systems.systems[sys_id].lambdadotmus[mode]
                    lambda_dot_mu_sqrd = self.mp_systems.systems[sys_id].lambdadotmus[mode]
                else:
                    lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
                                                                        lamb=np.round(np.linalg.norm(lambdas_cur_sys), 3), 
                                                                        sqrd=False)
                    if "Gsqrd" in model:
                        lambda_dot_mu_sqrd = lambda_dot_mu_sys / prefactor
                    else:
                        lambda_dot_mu_sqrd = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
                                                                        lamb=np.round(np.linalg.norm(lambdas_cur_sys), 3), 
                                                                        sqrd=True)
                lambda_dot_mu_total += tensor(*[lambda_dot_mu_sys if sys_id==j 
                                              else qeye(self.mp_systems.dims[j])
                                              for j in range(self.mp_systems.nsystems)])
                lambda_dot_mu_sqrd_total += tensor(*[lambda_dot_mu_sqrd if sys_id==j 
                                              else qeye(self.mp_systems.dims[j])
                                              for j in range(self.mp_systems.nsystems)])

            # term -sqrt(omega / 2) (a + adagger) (lambda dot mu)
            self.total_blc += tensor(cav_blc, lambda_dot_mu_total)

            # term 1/2 (lambda dot mu)^2 
            if "nodse" not in model:
                self.total_dse += 1/2 * tensor(self.cavity.identity, lambda_dot_mu_sqrd_total @ lambda_dot_mu_sqrd_total)
                if "dseH2" in self.model:
                    dse_part = [0.16251041, 0.50338401, 0.50333195, 0.84420554]
                    dse_part_2 = basis(4, 0) * basis(4, 3).dag() * 0.07835277
                    self.total_dse += tensor(self.cavity.identity, qdiags(dse_part) * self.lambdas_spatial[0][0][2]**2)
                    self.total_dse += tensor(self.cavity.identity, (dse_part_2 + dse_part_2.dag()) * self.lambdas_spatial[0][0][2]**2)

        self.total_hamiltonian = self.total_elec + self.total_photon + self.total_blc + self.total_dse

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
            
            if "minus" in self.model or "plus" in self.model:
                ann_cav = self.gen_ann_shifted(ann_cav)
            
            num_op = ann_cav.dag() * ann_cav

            if "pzw" in self.model:
                u = self.gen_pzw()
                num_op = u * num_op * u.dag()
                           
            operators.append(num_op)              
        return operators

    def gen_ann_shifted(self, ann_cav):
        # a -> a' = a + i sum_i (eta * sigma_x^i) 
        # a -> a' = ia - i wc sum A_i cdot mu_i
        # a -> a' = ia - i sum 1/sqrt(w_c/2) lambda cdot mu
        ann_cav *= complex(0, 1)
        for mode in range(self.cavity.nmodes):
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys = self.mp_systems.systems[sys_id]
                lambdas_cur_sys = self.lambdas_spatial[sys_id][mode]
                if isinstance(trans_sys, MultilevelSystem):
                    lambda_dot_mu_sys = self.mp_systems.systems[sys_id].lambdadotmus[mode]
                else:
                    lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
                                                                        lamb=np.linalg.norm(lambdas_cur_sys), 
                                                                        sqrd=False)
                lambda_dot_mu_total = tensor(*[lambda_dot_mu_sys if sys_id==j 
                                          else qeye(self.mp_systems.dims[j])
                                          for j in range(self.mp_systems.nsystems)])
                prefactor = complex(0, 1) / np.sqrt(2 * self.cavity.freqs[mode])
                if self.model == "minus":
                    ann_cav -= prefactor * tensor(self.cavity.identity, lambda_dot_mu_total)
                elif self.model == "plus":
                    ann_cav += prefactor * tensor(self.cavity.identity, lambda_dot_mu_total)
        return ann_cav   

    def gen_sys_operators(self):
        """
        Gives state operators for every systems' states
        """
        operators = []
        for i in range(self.mp_systems.nsystems):
            for level in range(self.mp_systems.dims[i]):
                sys_op = tensor(*[qeye(self.mp_systems.dims[j]) if i!=j
                                 else basis(self.mp_systems.dims[i], level) * basis(self.mp_systems.dims[i], level).dag()
                                 for j in range(self.mp_systems.nsystems)])
                operators.append(tensor(self.cavity.identity, sys_op))
        return operators

    def gen_pol_operators(self, eigs):
        """
        Gives state operators for every systems' states
        """
        operators = []
        eigengs, eigvecs = self.total_hamiltonian.eigenstates()
        for eigenstate in eigvecs[eigs]:
            operators.append(eigenstate * eigenstate.dag())
        return operators
        
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
            U = self.gen_pzw()
            proj = U * proj * U.dag()
        return proj
        
    def gen_joint_label(self, systems_dirac, photons_dirac):
        """
        Gives string for the operators 
        """
        return r"$|{}\rangle |{}\rangle$".format(str(systems_dirac)[1:-1], str(photons_dirac)[1:-1])

    def gen_pzw(self):
        # gives the power zienau woolley 
        # tranformation unitary transformation
        # U = exp(-i eta (a + a.dag()) sigmax)
        # U = exp(-(a - a.dag()) sqrt(1 / 2 omega_c) lambda dot mu)
        terms = []
        for mode in range(self.cavity.nmodes):
            ann_cav = tensor(*[qeye(self.cavity.dims[m]) if mode != m
                                    else destroy(self.cavity.dims[mode])
                                    for m in range(self.cavity.nmodes)])
            lambda_dot_mu_total = qzero(self.mp_systems.dims)
            for sys_id in range(self.mp_systems.nsystems):
                trans_sys = self.mp_systems.systems[sys_id]
                lambdas_cur_sys = self.lambdas_spatial[sys_id][mode]
                if isinstance(trans_sys, MultilevelSystem):
                    lambda_dot_mu_sys = self.mp_systems.systems[sys_id].lambdadotmus[mode]
                else:
                    lambda_dot_mu_sys = trans_sys.gen_lambdadotmu_terms(freq=self.cavity.freqs[mode], 
                                                                        lamb=np.linalg.norm(lambdas_cur_sys), 
                                                                        sqrd=False)
                lambda_dot_mu_total += tensor(*[lambda_dot_mu_sys if sys_id==j 
                                          else qeye(self.mp_systems.dims[j])
                                          for j in range(self.mp_systems.nsystems)])
            prefactor = - 1 / np.sqrt(2 * self.cavity.freqs[mode])
            terms.append(prefactor * tensor(ann_cav - ann_cav.dag(), lambda_dot_mu_total))
        U = sum(terms).expm()
        return U

    def gen_gamma_losses(self, gamma):
        # sqrt(2 gamma) * sigma_-, spontaneuos lossy emission from excited system
        jumps = []
        for sys_id in range(self.mp_systems.nsystems):
            prefactor = np.sqrt(2 * gamma)
            sigma_lower = self.mp_systems.gen_system_lower(sys_id)
            jumps.append(prefactor * tensor(self.cavity.identity, sigma_lower))
        return jumps

    def gen_kappa_losses(self, kappa):
        # sqrt(2 kappa) * a, sponatneuos lossy cavity emission
        jumps = []
        for mode in range(self.cavity.nmodes):
            prefactor =  np.sqrt(2 * kappa)  
            ann_cav =  tensor(*[qeye(self.cavity.dims[m]) if mode != m
                                    else destroy(self.cavity.dims[mode])
                                    for m in range(self.cavity.nmodes)], self.mp_systems.identity)
            shifted_ann = self.gen_ann_shifted(ann_cav)
            jumps.append(prefactor * shifted_ann)
        return jumps


