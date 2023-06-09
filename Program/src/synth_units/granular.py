"""
Created on Mon Oct 17 11:21:50 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Granular Synthesizer
"""

import numpy as np
from random import normalvariate
import random
from scipy import signal

from PLAY_MIDI_AND_SAMPLES_j import Unclipping

# create grains from a sample and store in an array
def Create_Grains(samples, grain_size):
    n_grains = int(len(samples)/grain_size)+1
    grain_table = np.array_split(samples, n_grains)
    return grain_table

# Smooth audio signal through convolution
def Smoothing(out_samples, window):
    out_samples = np.convolve(out_samples, 100 , mode = 'same')
    return out_samples

# returns index of random choice of grain, within a normal distribution
def normal_choice(lst, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2
        
    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return index


# order grains using a sliding normal distribution probability window
def Order_Grains(n_grains, grain_center):
    lst = [0]*n_grains
    #init list with grain order
    grain_order = np.ones(n_grains)*-1
    
    for grain in range(n_grains):
        grain_order[grain] = int(normal_choice(lst, mean = grain_center[grain]))
    
    return grain_order


# construct output signal by joining grains together
def Arrange_Grains(grain_table, grain_space, grain_order, sync, sr, length, grain_size):
    out_samples = np.array([])
    grain_order=list(grain_order)
    
    if grain_space >-2 and grain_space <2: grain_space=2
    
    if grain_space > 0:
        if sync == False:
            space_list = list(np.arange(0, grain_space, min((grain_space/2), 10)))
        
        for i in range(len(grain_table)):
            #append grain
            out_samples = np.append(out_samples, grain_table[int(grain_order[i])])
            
            #append space between grains
            if sync == True:
                out_samples = np.append(out_samples, np.zeros(grain_space))
            else:
                out_samples = np.append(out_samples, np.zeros(int(random.sample(space_list, 1)[0])))
    else:
        ########################### COMPENSATE LENGTH ##########################################
        #what each grain will occupy in the out_samples when overlap happens
        grain_length = grain_size+grain_space
        
        #how many times the signal has to be extended to reach the original sample's length
        compensate = int(np.ceil((length/grain_length)/len(grain_table)))
        
        #copy grain list untill reaching desired length
        for double in range(1,compensate):
            grain_table+=grain_table
            grain_order+=grain_order
        ########################################################################################
        
        if sync == False:
            space_list = list(range(grain_space, 0, min(abs(int(grain_space/2)), 10)))
            
        out_samples = np.append(out_samples, grain_table[0])
            
        for i in range(1, len(grain_table)):
            grain = grain_table[int(grain_order[i])]
            
            if sync==False: grain_space = random.sample(space_list, 1)[0]
                                     
            grain_over = grain[:-grain_space]
            grain_under = out_samples[grain_space:]
            
            grain_over += grain_under
            grain_over /=2
            
            out_samples[grain_space:] = grain_over
            out_samples = np.append(out_samples, grain[-grain_space:])
    
    return out_samples


# applies velocity attack attenuation and cuts input sound to desired duration 
def envelope(sample, velocity, duration, sr):
    if velocity<127:
        # Calculate the velocity scaling factor and attack time
        scaling_factor = velocity / 127
        attack_time = 0.001 + (1 - scaling_factor) * 0.009  # range [0.01, 0.1]
    
        # Apply the attack modulation to the sound sample
        num_samples = len(sample)
        attack_samples = int(attack_time * sr)
        if attack_samples<len(sample):
            attack = np.linspace(0, 1, attack_samples)
            envelope = np.concatenate((attack, np.ones(num_samples - attack_samples)))
            sample = sample * envelope
    
    cut_sample=sample[:duration]
    return cut_sample


# apply envelope and unclipping to granular signal
def apply_envelope(grain_table, sr):
    for idx in range(len(grain_table)):  
        grain = grain_table[idx]
        grain = envelope(grain, 70, len(grain), sr)
        grain_table[idx] = Unclipping(grain, offset=0, smoothing_level=0.2)
    
    return grain_table
    

# apply granular su«ynthesis to input signal
def Granular_Synthesis(samples, grain_size, grain_space, window, sr, order= False, smoothing = False,
                       sync= True):
    grain_table = Create_Grains(samples, grain_size)
    grain_table = apply_envelope(grain_table, sr)
    grain_center = list(range(len(grain_table)))  
    
    if order == False: 
        #for gaussian distribution grain picking
        random.shuffle(grain_center)
    
    # order == true means that the mean center shifts accordingly to the signal evolution over time
    grain_order = Order_Grains(len(grain_table), grain_center)
    #order the grains
    out_samples = Arrange_Grains(grain_table, grain_space, grain_order, sync, sr, len(samples), grain_size)
    
    #normalize
    norm_coef = max(abs(out_samples))
    out_samples_norm = out_samples/norm_coef
    
    #smoothen signal
    if smoothing == True:
        out_samples_norm = Smoothing(out_samples_norm, window)
    
    #denormalize
    out_samples = out_samples_norm*norm_coef
    
    return out_samples[:len(samples)]


# convert normalized grain space parameter to "sampling time"
def get_grain_space(grain_space_norm, grain_size, sr):
    grain_space = int(((grain_space_norm-1)*grain_size)/2) 
    return grain_space


# granular synth
def granular_synth(input_signal, sr, param_dict , automation_beginning,
                      automation_ending, section, automation, pitch):
    
    if len(input_signal)<2:
        output=input_signal
    else:
        parameters = param_dict[section]
        
        # from 10 milisseconds to 100 milisseconds
        grain_size = int(parameters['grain_size']*sr)
        
        #space between grains
        grain_space = get_grain_space(parameters['grain_space'], grain_size, sr)
    
        #smoothing window
        window = signal.windows.hamming(int(0.007*sr))
        
        output = Granular_Synthesis(input_signal, grain_size, grain_space, window, sr,
                                    order=parameters['order'], smoothing = parameters['smoothness'],
                                    sync = parameters['sync'])
    
    return output
