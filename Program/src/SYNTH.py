"""
Created on Fri Oct 28 14:53:49 2022

@author: JP
"""
import sys
import os
import numpy as np

current_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path.replace("\\", "/")+'/synth_units')

from lfo_shaper import lfo_shaper_func as shape_lfo
from interpolator import create_evo_window

import warnings
warnings.filterwarnings("ignore")

def scale_one_parameter(parameter, norm_value, scale_list):
    scaling_floor = scale_list[0][parameter]
    scaling_ceiling = scale_list[1][parameter]
    
    scaling = [scaling_floor, scaling_ceiling]
    
    scale_range = scaling[1]-scaling[0]
    
    value = scaling[0] + scale_range*norm_value
    return value

def scale_all_parameters(parameters, scale_param, synth, inst):
    scaled_parameters = dict()
    
    for key, value in parameters.items():
        if type(value) == type(float()) and synth!=2:
            new_value = scale_one_parameter(key, value, scale_param[synth][inst])
            #new_value = scale_one_parameter(key, value, scale_param)
            scaled_parameters[key] = new_value
        else:
            scaled_parameters[key] = value
    return scaled_parameters

def Create_automation(self):
    automation = np.array([])
    
    #create list with parameters for every section (for this synth)   
    for sec in range(self.n_sections):
        param_dict=dict()
        idx_param=0
        for key in self.parameter_types:
            param_dict[key] = self.synth_params[sec][idx_param]
            idx_param += 1
        
        scaled_param_dict = scale_all_parameters(param_dict, self.parameters_range, self.synth, self.inst)
        
        self.param_dict_list.append(scaled_param_dict)
    
    
    automation = np.append(automation,np.zeros(int(self.beg_silence)))
       
    duration = int(self.window_size)
    
    if self.synth == 0: #autofilter
        for sec in range(self.n_sections):
            
            cutoff_floor = self.param_dict_list[sec]['cutoff_floor']
            cutoff_ceiling = self.param_dict_list[sec]['cutoff_ceiling']
            
            lfo_floor = self.param_dict_list[sec]['lfo_floor']
            lfo_ceiling = self.param_dict_list[sec]['lfo_ceiling']
            
            lfo_shape = self.param_dict_list[sec]['lfo_shape']
            lfo_evo = self.param_dict_list[sec]['lfo_evo']
            
            lfo_speed = [lfo_floor, lfo_ceiling] #frequency range of lfo
            freq_range = [cutoff_floor, cutoff_ceiling] #frequency range of cutoff frequency (amplitude range of lfo)
            
            section_aut = shape_lfo(lfo_shape, lfo_evo, duration,
                                         lfo_speed, freq_range, self.sr)
            
            automation = np.append(automation, section_aut)
    elif self.synth == 2: #interp
        for sec in range(self.n_sections):
            evo_type = self.param_dict_list[sec]['evo']
            window = create_evo_window(duration, evo_type, 1, 0)
            automation = np.append(automation, window)
    #print('debug automation')    
    return automation

class SYNTH:
    def __init__(self, synth, sr, user_info, parameter_types, parameters_range, 
                 synth_params, window_size, midi_info, beg_compass_samples, inst):

        synth_path=user_info.main_path +'/src/synth_units'
        if synth_path not in sys.path:
            sys.path.append(synth_path) #talvez é preciso pôr /src/

        from auto_filter import auto_filter_synth
        from granular import granular_synth
        from interpolator import interpolator_synth
        synth_functions = [auto_filter_synth, granular_synth, interpolator_synth] #types of synths
        
        self.synth = synth #synth type
        self.sr = sr
        
        # size of each section
        self.window_size = window_size
        self.n_sections = midi_info.n_sections
        # size in samples of the silence in the beggining of the song
        self.beg_silence = beg_compass_samples
        
        # parameter values for this synth (every section)
        self.synth_params = synth_params
        # parameter variable type for this synth
        self.parameter_types = parameter_types
        # parameter range for this type of synth
        self.parameters_range = parameters_range

        #list with dictionary of parameters for every section (for each inst)
        self.param_dict_list = []   
        
        self.inst=inst
        
        #create array with automation for this synth
        self.automation = Create_automation(self)
        
        self.synthesis = synth_functions[synth] #select synth function
        
        
        
    def synthetize(self, signal, automation_beginning,
                   automation_ending, section, pitch):
        #sec_beg: beggining of that section
        #sec_end: end of that section
        
        synth_signal = self.synthesis(signal, self.sr, self.param_dict_list, automation_beginning,
                                       automation_ending, section, self.automation, pitch)
        
        return synth_signal
        