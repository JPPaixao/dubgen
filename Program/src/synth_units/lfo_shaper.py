# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:57:15 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: "lfo_shaper_func" creates LFO and modifies its speed over time
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import random

shapes = ['sine', 'square', 'triangle', 'sawr', 'sawl', 'random']
lfo_mod = ['cte', 'linear_up', 'linear_down', 'exp_up', 'exp_down']


# create LFO frequency "evo"lution window
def create_lfo_evo(N, mod, ceiling, floor):
    if mod == 'cte':
        evo = np.array(N)*floor
    elif mod == 'linear_up':
        m = (ceiling-floor)/N #slope
        b = floor #bias
        evo = np.arange(N)*m +b
    elif mod == 'linear_down':
        m = (floor-ceiling)/N #slope
        b = ceiling #bias
        evo = np.arange(N)*m +b
    elif mod == 'exp_up':
        evo = np.geomspace(floor, ceiling, N)
    elif mod == 'exp_down':
        evo = np.geomspace(ceiling, floor, N)
    else:
        evo = random_evo(ceiling, floor, N)   
    return evo


# random evolution of LFO frequecny (within a range)
def random_evo(ceiling, floor, N): 
    rand_n_points = random.randrange(2,min(10,N))
    
    if ceiling-floor<10:
        range_freq = list(np.linspace(floor, ceiling, 2*(rand_n_points+2)))
    else: range_freq = range(int(floor), int(ceiling))
        
    rand_freq = random.sample(range_freq, rand_n_points+2)       
    rand_x = random.sample(range(-10,N+10), rand_n_points)
    
    rand_x.sort()
    rand_x.insert(0, -1)
    rand_x.append(N+1)
    
    interpolation = interp1d(np.array(rand_x), np.array(rand_freq),
                             kind='linear', fill_value="extrapolate")
    
    x = np.arange(0, N)
    evo = interpolation(x)
    window_size=min(51, N)
    smooth_evo = savgol_filter(evo, window_size, 3)
    return smooth_evo


# triangle shaped signal
def tri(x):
    sigma = 0.01
    return 1 - 2*np.arccos((1 - sigma)*np.sin(2*np.pi*x))/np.pi


# square shaped signal
def square(x):
    sigma = 0.01
    return 2*np.arctan(np.sin(2*np.pi*x)/sigma)/np.pi


# saw shaped signal
def saw(x):
    return (1 + tri((2*x - 1)/4)*square(x/2))/2


# vectorize signal shape function ("tri", "square" or "saw") to get an output array.
# "x_array" modulates the speed and "function" is the signal's shape
def custom_wave_function(x_array, function):
    wave_function = np.vectorize(function)
    return wave_function(x_array)


# creates final LFO signal
def create_lfo_shape(x_plot, shape):
    if shape == 'sine':
        y = np.sin(x_plot)
    elif shape == 'square':
        y =  custom_wave_function(x_plot, square)     
    elif shape == 'triangle':
        y =  custom_wave_function(x_plot, tri)     
    elif shape == 'sawr':
        y =  custom_wave_function(x_plot, saw)   
    elif shape == 'sawl':
        y =  custom_wave_function(x_plot, saw)*-1 
    return y


# create LFO and modify its speed over time
def lfo_shaper_func(shape, lfo_mod, n_samples, lfo_speed, freq_range, sr):
    sr = 44100 #sampling rate
    T=1/sr #period
    
    #freq_range of lfo speed
    floor = lfo_speed[0]
    ceiling = lfo_speed[1]
    
    #freq_range which is the amplitude range of the lfo
    amp_floor = freq_range[0]
    amp_ceil = freq_range[1]
    
    #create evolution of lfo freqs
    lfo_evo = create_lfo_evo(n_samples, lfo_mod, ceiling, floor)

    x = np.arange(0, n_samples, 1)
    #dx = x_{n}-x{n-1}: space between every x sample (in seconds)
    dx = np.ones(len(x))* float(T) 
    
    freq = lfo_evo
    # Cumsum freq * change in x
    x_plot = (freq * dx ).cumsum()
    
    y = create_lfo_shape(x_plot, shape)
    
    amplitude = (amp_ceil-amp_floor)/2
    bias = amp_floor + amplitude
    y = amplitude*y + bias
    
    return y

