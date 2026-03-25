# cavity.py

import numpy as np
from qutip import Qobj, tensor, qeye, basis, num, destroy

class Cavity:
    """
    Class to construct a cavity with specified 
    frequency and specified photons allowance
    """

    def __init__(self, freqs, max_photon_nums):
        """
        Initialize the cavity
        Args:
            freqs, frequencies of modes
            max_photon_nums, maximum count allowed in a mode
        """
        self.freqs = freqs
        self.max_photon_nums = max_photon_nums
        self.nmodes = len(freqs)
        self.dims = [max_photon_nums[n] + 1 for n in range(self.nmodes)]
        self.identity = tensor(*[qeye(dim) for dim in self.dims])

        return

    def gen_cavity_state(self, photon_dirac):
        """
        Makes ket based on dirac reprsentation
        """
        cavity_state = tensor(*[basis(self.dims[alpha], photon_dirac[alpha])
                                if isinstance(photon_dirac[alpha], int)
                                else qeye(self.dims[alpha])
                                for alpha in range(self.nmodes)])
        return cavity_state

    def gen_cavity_hamiltonian(self):
        """
        Makes hamiltonian for energies of frequencies and 
        max allowance
        """
        return tensor(*[self.freqs[n] * num(self.dims[n]) for n in range(self.nmodes)])

    def gen_ann_op(self, mode):
        trans_mode = destroy(self.dims[mode])
        ann = tensor(*[trans_mode if mode==m
                       else qeye(self.dims[m])
                       for m in range(self.nmodes)])      
        return ann


    def gen_cavity_operators(self, mp_dims):
        """
        Makes operators to get photon number
        for every cavity mode
        """
        operators = []
        labels = []
        for mode in range(self.nmodes):
            ann = destroy(self.dims[mode])
            cur_mode = ann.dag() * ann
            operators.append(tensor(*[cur_mode if mode == m
                                      else qeye(self.dims[m])
                                      for m in range(self.nmodes)], qeye(mp_dims)))
            labels.append("Photons of Freq_{}".format(self.freqs[n]))
        return operators, labels



