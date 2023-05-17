# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:40:06 2022

@author: JP
"""

import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks
import math
import librosa

def freq_to_midi(freq):
    midi_note = np.round(( ( math.log2( (32*freq)/440 ) )*12 ) + 9)
    
    #blindar de forma a nota estar sempre no intervalo 0 a 127
    return midi_note

def Pitch_Detection_ACF(data, sampling_frequency):
    
    auto = sm.tsa.acf(data, nlags=2000) #compute autocorreltion
    
    peaks = find_peaks(auto)[0] # Find peaks of the autocorrelation
    lag = peaks[0] # Choose the first peak as our pitch component lag
    
    pitch = sampling_frequency / lag # Transform lag into frequency
    
    # ACF Pitch Detection
    midi_note = freq_to_midi(pitch)
    
    if midi_note<0:midi_note=0
    elif midi_note>127:midi_note=127
    
    return int(midi_note), pitch

def Pitch_MODA(f0):
    
    if len(f0) > 1:
        f0=np.array(f0)
        #round pitch by "tenths" (to avoid noise and very different frequencies)
        f0_rounded = list(np.round(f0, -1))
        
        #find moda
        moda_f0 = max(f0_rounded, key = f0_rounded.count)
        
        #use the idxs of elements that correspond to the moda (rounded)
        idx_moda = np.array(np.where(np.array(f0_rounded)==moda_f0)[0])
        
        #compute average of those elements in the original pitch array
        avg_f0 = np.sum(f0[idx_moda])/len(f0[idx_moda])
    else:
        avg_f0=np.nan
        
    
    return avg_f0

def Pitch_Detection(data, sampling_frequency):

    
    f0_vec, voiced_flag, voiced_probs = librosa.pyin(data,
                                             fmin=librosa.note_to_hz('C0'),
                                             fmax=librosa.note_to_hz('C7')) #talvez pôr C2 no limite inferior
    
    pitch = Pitch_MODA(f0_vec)
    #print('pitch detection:', pitch)
    
    if pitch<0:
        print('pitch negativo:', pitch,'(?!)')
        pitch=0.1
        
    if type(pitch)!= float and type(pitch)!= int and type(pitch)!= type(np.float64(0)):
        print('pitch com formato inválido', pitch,'(', type(pitch),'?!)')
        pitch=0.1
        
    if math.isinf(pitch):
        print('pitch com valor infinito:', pitch,'(?!)')    
        pitch=0.1
        
    if np.isnan(pitch):
        print('pitch nulo (NaN):', pitch,'(?!)')    
        pitch=0.1        
    
    ##talvez isto não é necessário###########################
    target_note = librosa.hz_to_note(pitch)
    target_hz = librosa.note_to_hz(target_note)
    
    steps_to_A_target = librosa.core.pitch_tuning(target_hz)
    ########################################################
    
    steps_to_A_sample = librosa.core.pitch_tuning(pitch)
    
    steps_tuning = steps_to_A_sample-steps_to_A_target #desvio entre a nota da sample e a nota pretendida (mais próxima)
    #print('steps_tuning', steps_tuning )
    
    #print('pitch: ', pitch)
    note = librosa.hz_to_note(pitch)
    # Porque se fizessemos só hz to midi, ele iria aproximar a nota sempre por um valor mais baixo (floor(hz)->midi)
    # enquanto que o hz_to_note aproxima pelo valor mais proximo!! round(hz)->note->midi
    midi_note = librosa.note_to_midi(note)
    
    if midi_note<0:midi_note=0
    elif midi_note>127:midi_note=127
    
    return int(midi_note), f0_vec, steps_tuning
