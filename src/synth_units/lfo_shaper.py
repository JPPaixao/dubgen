# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:57:15 2022

@author: Tls Chris
"""

#Create lfo and modify its speed

shapes = ['sine', 'square', 'triangle', 'sawr', 'sawl', 'random']

lfo_mod = ['cte', 'linear_up', 'linear_down', 'exp_up', 'exp_down']

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal

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
    else: #mod is random
        evo = random_evo(ceiling, floor, N)
        
    return evo

def random_evo(ceiling, floor, N): #ainda não está bem!!
# e interpolação também só pode estar dentro dos limites!!!
    import random
    rand_n_points = random.randrange(2,min(10,N))
    
    if ceiling-floor<10:
        range_freq = list(np.linspace(floor, ceiling, 2*(rand_n_points+2)))
    else: range_freq = range(int(floor), int(ceiling))
        
    rand_freq = random.sample(range_freq, rand_n_points+2)
        
    rand_x = random.sample(range(-10,N+10), rand_n_points)
    #x_samples = np.arange(-100, 3100, 50)
    
    rand_x.sort()
    rand_x.insert(0, -1)
    rand_x.append(N+1)
    
    interpolation = interp1d(np.array(rand_x), np.array(rand_freq),
                             kind='linear', fill_value="extrapolate")
    #quadratic interpolation often leads to some negative freq values,
    #which doesnt work with the rest of the process inside the synth
    
    #plt.title("lfo evo random points")
    #plt.plot(rand_x, rand_freq)
    
    x = np.arange(0, N)
    evo = interpolation(x)
    '''
    plt.plot(evo)
    plt.title("lfo evo")
    plt.show()
    '''
    
    from scipy.signal import savgol_filter
    
    window_size=min(51, N)
    smooth_evo = savgol_filter(evo, window_size, 3)
    '''
    plt.plot(smooth_evo)
    plt.title("lfo smooth evo")
    plt.show()
    '''
    return smooth_evo


def create_lfo_shape(x_plot, shape): #falta o random...
    if shape == 'sine':
        y = np.sin(x_plot)
        
    elif shape == 'square':
        #y = signal.square(2 * np.pi * x_plot)
        y =  custom_wave_function(x_plot, square)
        
    elif shape == 'triangle':
        #y = 2*np.abs(signal.sawtooth(2 * np.pi * x_plot))-1
        y =  custom_wave_function(x_plot, tri)
        
    elif shape == 'sawr':
        #y = signal.sawtooth(2 * np.pi * x_plot)
        y =  custom_wave_function(x_plot, saw)
        
    elif shape == 'sawl':
        #y = signal.sawtooth(2 * np.pi * x_plot)*-1
        y =  custom_wave_function(x_plot, saw)*-1
        
    return y


def lfo_shaper_func(shape, lfo_mod, n_samples, lfo_speed, freq_range, sr):
    sr = 44100 #sampling rate
    T=1/sr #period
    
    #freq_range of lfo
    floor = lfo_speed[0]
    ceiling = lfo_speed[1]
    
    #freq_range which is the amplitude range of the lfo
    amp_floor = freq_range[0]
    amp_ceil = freq_range[1]
    
    #create evolution of lfo freqs
    lfo_evo = create_lfo_evo(n_samples, lfo_mod, ceiling, floor)
    '''
    plt.figure()
    plt.plot(lfo_evo)
    plt.title('lfo_evo')
    plt.show()
    '''
    x = np.arange(0, n_samples, 1)
    dx = np.ones(len(x))* float(T) #dx = x_{n}-x{n-1}: espaço entre cada unidade de x em SEGUNDOS!!!
    
    freq = lfo_evo
    
    x_plot = (freq * dx ).cumsum()    # Cumsum freq * change in x
    
    
    y = create_lfo_shape(x_plot, shape)
    
    '''
    plt.figure()
    plt.plot(x, y, label="sin(freq(x) * x)")
    plt.plot(x, freq, label="freq(x)")
    plt.show()
    
    plt.figure()
    plt.plot(x, y, label="sin(freq(x) * x)")
    plt.show()
    '''
    amplitude = (amp_ceil-amp_floor)/2
    bias = amp_floor + amplitude
    y = amplitude*y + bias
    
    return y

def tri(x):
    sigma = 0.01 #0.01 works too (closer to zero, the more pointy)
    return 1 - 2*np.arccos((1 - sigma)*np.sin(2*np.pi*x))/np.pi

def square(x):
    sigma = 0.01 #0.01 works too (closer to zero, the more pointy)
    return 2*np.arctan(np.sin(2*np.pi*x)/sigma)/np.pi

def saw(x):
    return (1 + tri((2*x - 1)/4)*square(x/2))/2

def custom_wave_function(x_array, function):
    
    wave_function = np.vectorize(function)
    
    return wave_function(x_array)

'''
# Original:
###############################################################################################
# for the random function!!!
n_samples = 350
x_samples = np.arange(-10, n_samples, 50)
freq_samples = np.random.random(x_samples.shape) * 1.5 + 0.5

#linear
#x_samples = np.array([0, 300])
#freq_samples = (np.arange(int(x_samples.shape[0]))/int(x_samples.shape[0]))* 1.5 + 0.5

x = np.arange(0, 300, 0.1)

dx = np.full_like(x, 0.1 )       # Change in x

interpolation = interp1d(x_samples, freq_samples, kind='quadratic')
freq= interpolation(x)

x_plot = (freq * dx ).cumsum()    # Cumsum freq * change in x

y = np.sin(x_plot)
#y = signal.sawtooth(2 * np.pi * x_plot)

plt.plot(x, y, label="sin(freq(x) * x)")
plt.plot(x, freq, label="freq(x)")
plt.legend()
plt.show()

##############################################################################################
'''
