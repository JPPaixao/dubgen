# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:27:06 2022

@author: João Paixão

@brief: Arctangent Audio Compressor. Inspired in compressor by: Frank Zalkow
"""

import numpy as np
from scipy.interpolate import interp1d

#apply arctan transfer functiin
def apply_transfer(signal, transfer, interpolation='linear'):
    constant = np.linspace(-1, 1, len(transfer))
    interpolator = interp1d(constant, transfer, interpolation, fill_value="extrapolate")
    return interpolator(signal)

# hard limiting
def limiter(x, treshold=0.8):
    transfer_len = 1000
    transfer = np.concatenate([ np.repeat(-1, int(((1-treshold)/2)*transfer_len)),
                                np.linspace(-1, 1, int(treshold*transfer_len)),
                                np.repeat(1, int(((1-treshold)/2)*transfer_len)) ])
    return apply_transfer(x, transfer)

# smooth compression: if factor is small, its near linear, the bigger it is the
# stronger the compression
def arctan_compressor(x, factor=2):
    constant = np.linspace(-1, 1, 1000)
    transfer = np.arctan(factor * constant)
    transfer /= np.abs(transfer).max()
    return apply_transfer(x, transfer)