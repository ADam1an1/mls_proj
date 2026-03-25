import os, sys
from qutip import *
import numpy as np
from tqdm import tqdm

def calc_mf(f):
    return f * 2 + 1
    
def calc_mf_ind(f_list, f_ind, mf):
    mf_ind = np.sum([calc_mf(f) for f in f_list[:f_ind]])
    return int(mf_ind + f_list[f_ind] + mf)

def calc_num_trans(f_list, f_ind1, f_ind2, typ):
    mf1 = calc_mf(f_list[f_ind1])
    mf2 = calc_mf(f_list[f_ind2])
    if typ == "+":
        return int(min(mf1 - 1, mf2))
    elif typ == "pi":
        return int(min(mf1, mf2))
    elif typ == "-":
        return int(min(mf1 - 1, mf2))

def populate_transition(arr, f_list, f_start, f_end, typ, mf_start, elem):
    start = calc_mf_ind(f_list, f_start, mf_start)
    if typ == "+":
        end = calc_mf_ind(f_list, f_end, mf_start + 1)
    elif typ == "pi":
        end = calc_mf_ind(f_list, f_end, mf_start)
    elif typ == "-":
        end = calc_mf_ind(f_list, f_end, mf_start - 1)
        
    arr[start][end] = arr[end][start] = elem
    return arr

