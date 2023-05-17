# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:09:58 2022

@author: JP
"""

#interpolator:
    #given two signals and a interpolation window, it interpolates between the
    #two signals given the current interpolation coefficcient
    
    #interpolation window is the evolution of this coef. throughout the song

def linear_positive(x, coef):
    return coef*x
    

#create iteration window with size of specific sound. "type" determines automation
def create_ite_window(length, type=0):
    if type==0:
        func = linear_positive
        coef=1
    
    window = np.zeros(length)
    
    for s in range(length):
        window[s] = func(s/length, coef)
        
    return window

def Pitch_MODA(f0):
    f0_rounded = list(np.round(f0, -1)) #round pitch by "tenths" (to avoid noise and very different frequencies)
    
    moda_f0 = max(f0_rounded, key = f0_rounded.count) #find moda
    
    idx_moda = np.where(np.array(f0_rounded)==moda_f0) #use the idxs of elements that correspond to the moda (rounded)
    
    avg_f0 = np.sum(f0[idx_moda])/len(f0[idx_moda]) #compute average of those elements in the original pitch array
    
    
    return avg_f0

import soundfile as sf
from scipy.io.wavfile import read, write
import numpy as np
from scipy import signal
from scipy.signal import correlate
#from PLAY_MIDI_AND_SAMPLES_j import Unclipping

def create_modulator(x_plot, shape, hz): #falta o random...
    if shape == 'sine':
        y = np.sin(2 * np.pi * hz * x_plot)
        
    elif shape == 'square':
        y = signal.square(2 * np.pi * hz * x_plot)
        
    elif shape == 'triangle':
        y = 2*np.abs(signal.sawtooth(2 * np.pi * hz* x_plot))-1
        
    elif shape == 'sawr':
        y = signal.sawtooth(2 * np.pi* hz * x_plot)
        
    elif shape == 'sawl':
        y = signal.sawtooth(2 * np.pi * hz * x_plot)*-1
        
    return y

def create_evo_window(N, mod, ceiling, floor):
    N=int(N)
    e = 0.00001
    if mod == 'constant':
        evo = np.ones(N)*floor
        
    elif mod == 'linear_up':
        m = (ceiling-floor+e)/N #slope
        b = floor #bias
        evo = np.arange(N)*m +b
        
    elif mod == 'linear_down':
        m = (floor-ceiling+e)/N #slope
        b = ceiling #bias
        evo = np.arange(N)*m +b
        
    elif mod == 'exp_up':
        evo = np.geomspace(floor+e, ceiling+e, N)
        
    elif mod == 'exp_down':
        evo = np.geomspace(ceiling+e, floor+e, N)
        
    elif mod == 'random':
        evo = np.random.uniform(low=floor, high=ceiling, size=(N,))
        
    return evo

def interpolator_synth(input_signal, sr, param_dict , automation_beginning,
                      automation_ending, section, automation, pitch):

    if len(input_signal)<2:
        print('length of signal in interpolator:', len(input_signal))
        output=input_signal
    else:
    
        parameters = param_dict[section]
        
        shape = parameters['shape']
        
        ######################### verificar isto########################
        T=1/sr
        t = np.arange(len(input_signal))* float(T) # ou t = np.linspace(0, time, sr*time)? time being in seconds
        modulator = create_modulator(t, shape, pitch)
        ################################################################
        
        input_signal -= input_signal.mean(); input_signal /= input_signal.std()
        modulator -= modulator.mean(); modulator /= modulator.std()
    
        nsamples = input_signal.size
    
        ########################## find phase difference ########################
        # Find cross-correlation
        xcorr = correlate(input_signal, modulator)
        
        # delta time array to match xcorr
        dt = np.arange(1-nsamples, nsamples)
    
        recovered_time_shift = dt[xcorr.argmax()]
        ################################################################################################
        #adjust modulator to og signal phase (isto irá criar pops! pensar em melhor opcoes de ajustar fase)
        # alternativa: gerar parte inicial que falta da modulator em fase com o que levou roll...
        ################################################################################################
        modulator = np.roll(modulator, -recovered_time_shift)
        
        out_simple = np.zeros(nsamples)
    
    
        #window = automation[automation_beginning:automation_ending+1]
        window = automation[automation_beginning:automation_ending]
        
        
        for s in range(nsamples):
            coef = window[s]
            out_simple[s] = ( input_signal[s]*(1-coef) )+ ( modulator[s]*coef )
    
        output = out_simple.T
        
        #print('check sizes: in=',len(input_signal),', freq=',len(window),', out=',len(output))
        ######################################################
        #é preciso normalizar final????
        #####################################################
        
    return output
