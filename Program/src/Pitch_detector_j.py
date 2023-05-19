"""
Created on Tue May 31 15:40:06 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Determines pitch of input audio signal
"""
import numpy as np
import math
import librosa

#converts Hz to MIDI note
def freq_to_midi(freq):
    midi_note = np.round(( ( math.log2( (32*freq)/440 ) )*12 ) + 9)
    return midi_note


# takes pitch evolution through the signal and determines the most proeminent pitch value (through "MODA")
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


# Pitch detecion algorithm. Determines pitch of input audio signal
def Pitch_Detection(data, sampling_frequency):
    f0_vec, voiced_flag, voiced_probs = librosa.pyin(data,
                                             fmin=librosa.note_to_hz('C0'),
                                             fmax=librosa.note_to_hz('C7'))
    pitch = Pitch_MODA(f0_vec)
    
    if pitch<0 or math.isinf(pitch) or np.isnan(pitch):
        pitch=0.1   
    if type(pitch)!= float and type(pitch)!= int and type(pitch)!= type(np.float64(0)):
        pitch=0.1
     
    target_note = librosa.hz_to_note(pitch)
    target_hz = librosa.note_to_hz(target_note)
    steps_to_A_target = librosa.core.pitch_tuning(target_hz)
    steps_to_A_sample = librosa.core.pitch_tuning(pitch)
    steps_tuning = steps_to_A_sample-steps_to_A_target
    note = librosa.hz_to_note(pitch)
    midi_note = librosa.note_to_midi(note)
    
    if midi_note<0:midi_note=0
    elif midi_note>127:midi_note=127
    
    return int(midi_note), f0_vec, steps_tuning
