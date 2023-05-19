"""
Created on Wed Oct 19 15:14:53 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Auto-Filter: Low/High Pass Filter modulated by LFO.

All-Pass Filter based on Jan Wilczek Implementation:
    https://thewolfsound.com/allpass-based-lowpass-and-highpass-filters/
"""
import numpy as np


#compute a1 coefficient
def a1_coefficient(break_frequency, sampling_rate):
    tan = np.tan(np.pi * break_frequency / sampling_rate)
    return (tan - 1) / (tan + 1)


# compute all-pass filter output signal
def allpass_filter(input_signal, break_frequency, sampling_rate):
    # Initialize the output array
    allpass_output = np.zeros_like(input_signal)
    # Initialize the inner 1-sample buffer
    dn_1 = 0

    for n in range(input_signal.shape[0]):
        # The allpass coefficient is computed for each sample
        # to show its adaptability
        a1 = a1_coefficient(break_frequency[n], sampling_rate)

        # The allpass difference equation
        # Check the article on the allpass filter for an 
        # in-depth explanation
        allpass_output[n] = a1 * input_signal[n] + dn_1

        # Store a value in the inner buffer for the 
        # next iteration
        dn_1 = input_signal[n] - a1 * allpass_output[n]
        
    return allpass_output


# compute high/low-pass filter output signal (using an all-pass filter)
def allpass_based_filter(input_signal, cutoff_frequency, \
    sampling_rate, highpass=False, amplitude=1.0):
    # Perform allpass filtering
    allpass_output = allpass_filter(input_signal, \
        cutoff_frequency, sampling_rate)

    # If we want a highpass, we need to invert 
    # the allpass output in phase
    if highpass:
        allpass_output *= -1

    # Sum the allpass output with the direct path
    filter_output = input_signal + allpass_output
    # Scale the amplitude to prevent clipping
    filter_output *= 0.5
    # Apply the given amplitude
    filter_output *= amplitude

    return filter_output


#auto-filter synthesizer
def auto_filter_synth(input_signal, sr, param_dict , automation_beginning,
                      automation_ending, section, automation, pitch):
    
    if len(input_signal)<2:
        print('length of signal in autofilter:', len(input_signal))
        filter_output=input_signal
    else:
        parameters = param_dict[section]
        high_pass = parameters['high_pass']
        cutoff_frequency = automation[automation_beginning:automation_ending+1]
    
        if len(cutoff_frequency)==0:
            print('error in beg_sample or end_sample')
            
        # Actual filtering
        filter_output = allpass_based_filter(input_signal, \
            cutoff_frequency, sr, highpass=high_pass, amplitude=1)

    return filter_output
