"""
Created on Mon Oct 10 17:09:58 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Interpolator:
    1. Given two signals and a interpolation window, it interpolates between signals given the current interpolation coefficcient;
    2. "Interpolation Window" is the evolution of this coefficient throughout the song.
"""
import numpy as np
from scipy import signal
from scipy.signal import correlate


# get linear transformation
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


# takes pitch evolution thorugh the signal and determines the most proeminent pitch value (through "MODA")
def Pitch_MODA(f0):
    #round pitch by "tenths" (to avoid noise and very different frequencies)
    f0_rounded = list(np.round(f0, -1))
    #find moda
    moda_f0 = max(f0_rounded, key = f0_rounded.count)
    #use the idxs of elements that correspond to the moda (rounded)
    idx_moda = np.where(np.array(f0_rounded)==moda_f0)
    #compute average of those elements in the original pitch array
    avg_f0 = np.sum(f0[idx_moda])/len(f0[idx_moda])
    
    return avg_f0


#create modulator signal
def create_modulator(x_plot, shape, hz):
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


# create "evo" interpolation window
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


# Interpolator Synth
def interpolator_synth(input_signal, sr, param_dict , automation_beginning,
                      automation_ending, section, automation, pitch):
    if len(input_signal)<2:
        print('length of signal in interpolator:', len(input_signal))
        output=input_signal
    else:
        parameters = param_dict[section]
        shape = parameters['shape']
        
        T=1/sr
        t = np.arange(len(input_signal))* float(T)
        modulator = create_modulator(t, shape, pitch)
        
        input_signal -= input_signal.mean(); input_signal /= input_signal.std()
        modulator -= modulator.mean(); modulator /= modulator.std()
    
        nsamples = input_signal.size
    
        ########################## find phase difference ######################
        # Find cross-correlation
        xcorr = correlate(input_signal, modulator)
        # delta time array to match xcorr
        dt = np.arange(1-nsamples, nsamples)
        
        recovered_time_shift = dt[xcorr.argmax()]
        #######################################################################
        
        #adjust modulator to original signal phase
        modulator = np.roll(modulator, -recovered_time_shift)
        
        out_simple = np.zeros(nsamples)
        window = automation[automation_beginning:automation_ending]
        
        for s in range(nsamples):
            coef = window[s]
            out_simple[s] = ( input_signal[s]*(1-coef) )+ ( modulator[s]*coef )
    
        output = out_simple.T
    return output
