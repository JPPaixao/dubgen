"""
Created on Fri May 27 15:48:35 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Playback function that generates wav/mp3 files of MIDI Arrangement
"""
import MIDI_PROCESS_j as mp
from SYNTH import SYNTH
from Pitch_detector_j import Pitch_Detection
import Compressor_j as comp
from pedalboard_script import Pedalboard_func

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import math
import os
import sys
import pydub

import warnings
warnings.filterwarnings("ignore")


# shift pitch of sound
def pitch_shift(sample, note_target, sr, note_original):
    sample = librosa.effects.pitch_shift(sample, sr=sr,
                n_steps=note_target-note_original, bins_per_octave=12)
    
    return sample


#applies velocity attack attenuation and cuts input sound to desired duration
def envelope(sample, velocity, duration, sr):
    if velocity<127:
        # Calculate the velocity scaling factor and attack time
        scaling_factor = velocity / 127
        attack_time = 0.01 + (1 - scaling_factor) * 0.09  # range [0.01, 0.1]
    
        # Apply the attack modulation to the sound sample
        num_samples = len(sample)
        attack_samples = int(attack_time * sr)
        
        if attack_samples<len(sample):
            attack = np.linspace(0, 1, attack_samples)
            envelope = np.concatenate((attack, np.ones(num_samples - attack_samples)))
            sample = sample * envelope
    
    cut_sample=sample[:duration]
    
    return cut_sample, (duration-len(sample))


#this is where the synth parameters are defined
def get_range_parameters():
    
    #auto_filter
    Af_bass_floor = {'cutoff_floor': 8.0, 'cutoff_ceiling': 100.0, 'lfo_floor':0.01,
                  'lfo_ceiling': 2.1}
    Af_bass_ceiling = {'cutoff_floor': 30.0, 'cutoff_ceiling': 1_000.0, 'lfo_floor':2.0,
                  'lfo_ceiling': 4.0}
    
    Af_harm_floor = {'cutoff_floor': 20.0, 'cutoff_ceiling': 300.0, 'lfo_floor':0.01,
                  'lfo_ceiling': 3.1}
    Af_harm_ceiling = {'cutoff_floor': 200.0, 'cutoff_ceiling': 10_000.0, 'lfo_floor':3.0,
                  'lfo_ceiling': 5.0}
    
    Af_melo_floor = {'cutoff_floor': 20.0, 'cutoff_ceiling': 300.0, 'lfo_floor':0.01,
                  'lfo_ceiling': 3.1}
    Af_melo_ceiling = {'cutoff_floor': 200.0, 'cutoff_ceiling': 10_000.0, 'lfo_floor':3.0,
                  'lfo_ceiling': 5.0}
    
    #granular
    Gr_bass_floor = {'grain_size': 0.001, 'grain_space': 0}
    Gr_bass_ceiling = {'grain_size': 0.01, 'grain_space': 1}
    
    Gr_harm_floor = {'grain_size': 0.001, 'grain_space': 0}
    Gr_harm_ceiling = {'grain_size': 0.01, 'grain_space': 1}
    
    Gr_melo_floor = {'grain_size': 0.001, 'grain_space': 0}
    Gr_melo_ceiling = {'grain_size': 0.01, 'grain_space': 1}
    
    af_bass=[Af_bass_floor, Af_bass_ceiling]
    af_harm=[Af_harm_floor, Af_harm_ceiling]
    af_melo=[Af_melo_floor, Af_melo_ceiling]
    
    gr_bass=[Gr_bass_floor, Gr_bass_ceiling]
    gr_harm=[Gr_harm_floor, Gr_harm_ceiling]
    gr_melo=[Gr_melo_floor, Gr_melo_ceiling]
    
    
    af_range = [af_bass, af_harm, af_melo]
    gr_range = [gr_bass, gr_harm, gr_melo]
    
    ip_range = [list(), list(), list()] #just to init
    
    # list for every synth type. every synth type has specific ranges for each instruments.
    # each instrument has an upper and down limit (ceiling, floor) 
    param_range = [af_range, gr_range, ip_range]
    
    return param_range


# gets original pitch of every sample in the Arrangement 
def Get_MIDI_pitch(Ind, sr, Signals, sample_info):
    #visited_info: pitch and tuning of visited samples   
    Ind_pitch = np.zeros(np.shape(Ind))
    Ind_tuning = np.zeros(np.shape(Ind))
    
    for inst in range(np.shape(Ind)[0]):
        for sec in range(np.shape(Ind)[1]):
            if Ind[inst][sec] in sample_info.visited_samples:
                samp_idx = int(np.where(np.array(sample_info.visited_samples)==Ind[inst][sec])[0])
                Ind_pitch[inst][sec] = sample_info.visited_info[samp_idx][0]
                Ind_tuning[inst][sec] = sample_info.visited_info[samp_idx][1]
                
            else:        
                sample = Signals[Ind[inst][sec]]/sample_info.norm_coef[sample_info.sample_idx[Ind[inst][sec]][0]]
                Ind_pitch[inst][sec], og_freq, Ind_tuning[inst][sec] = Pitch_Detection(sample, sr)
                
                sample_info.visited_samples.append(Ind[inst][sec])
                sample_info.visited_info.append([Ind_pitch[inst][sec] , Ind_tuning[inst][sec]])

    return Ind_pitch, Ind_tuning


# exponentially decrescent function
def Exp_func_down(x):
    y = (1-(math.exp(1-x)))/(1-math.exp(1))
    return y


# exponentially crescent function
def Exp_func_up(x):
    y = ((math.exp(x))-1)/(math.exp(1)-1)
    return y


# smoothen sustain of sample by applying a function to the last "smoothing level"-percentage of the signal
def Unclipping(sample, offset, smoothing_level=0.2):
    #for the sound's attack:
    ending_point = round(len(sample)*(smoothing_level/2))
    for s in range(0,ending_point):
        x = s/ending_point
        sample[s]=Exp_func_up(x)*sample[s]
        
    #for the sound's release:
    starting_point = round(len(sample)*(1-smoothing_level))    
    for s in range(starting_point,len(sample)):
        x = (s-starting_point)/(len(sample)-starting_point)
        sample[s]=Exp_func_down(x)*sample[s]

    if offset>0: #in case the sample is shorter than the note duration needed
        sample = np.append(sample, np.zeros(offset))

    return sample
            

# adjust IND number of sections to desired length
def adjust_IND(Ind, midi_info_list):
    desired_length=0
    
    for info_midi in midi_info_list:
        desired_length = max(desired_length, info_midi.n_sections)
    
    adjusted_sequences = []
    for sequence in Ind:
        length = len(sequence)
        sequence=list(sequence)
        if length >= desired_length:
            adjusted_sequence = sequence[:desired_length]
        else:
            repetitions = desired_length // length
            remainder = desired_length % length
            adjusted_sequence = sequence * repetitions + sequence[:remainder]
        adjusted_sequences.append(np.array(adjusted_sequence))
    return np.array(adjusted_sequences)


#write mp3 file from array
def write_mp3(f, sr, x, normalized=False):

    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1]
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
    

#playback function: create wav and mp3 files for MIDI Arrangement with chosen samples and synth parameters(optional)
def PLAYBACK_midi_samples(Ind, user_info, midi_info_list, 
                          sample_info, midi_files, sr=22050, BPM=120, ex=-1, id_name=str(), first_sec = 0, 
                          synth = -1, synths_param=list(), synths=list(),
                          Parameters_types = list(), tkinter = None):
    
    sys.path.append(user_info.sound_folder)
    sys.path.append(user_info.midi_folder)
    
    #adjust from 5 sections to the original number of sections
    Ind = adjust_IND(Ind, midi_info_list)
    
    n_sections = np.shape(Ind)[1]
    inst_name = ['Bass', 'Harmony', 'Melody']
    
    Signals = sample_info.flat_samp
    Ind_pitch, Ind_tuning = Get_MIDI_pitch(Ind, sr, Signals, sample_info)
    
    Inst_signal = [[]]*len(midi_files) #saving signal for every instrument
    Inst_signal_synth = [[]]*len(midi_files) #saving signal for every instrument
    sound_filenames=[]
    sound_filenames_synth = []
    
    #for the synth (numeric range of parameters)
    parameters_range_list = get_range_parameters()
    
    for inst in range(0,len(midi_files)):

        text = 'Generating Arrangement... (Instrument ' + str(inst+1) + '/' + str(3)+ ' )'
        if tkinter !=None: tkinter.config(text=text)
    
        csv_strings = midi_info_list[inst].csv_strings
        velocity = midi_info_list[inst].velocity
        ppq = midi_info_list[inst].ppq
        window_size = user_info.section_size*4*ppq 
        n_sections = np.shape(Ind)[1]
        beg_compass = midi_info_list[inst].beg_compass #this is in midi ticks!!
    
        # for the case where IND has less sections than 5
        if first_sec > 0:
            BEG = beg_compass + (window_size*first_sec)
        else: BEG = 0
        END = max(BEG , beg_compass) + (window_size*n_sections)
        
        note_on = 'Note_on_c'
        note_off = 'Note_off_c'
        msg_beg = mp.first_substring(csv_strings, note_on)
        msg_end= mp.first_substring(csv_strings, 'End_track')-1
        
        delta_time = 60/(BPM*ppq) #time of each tick
        samples_per_tick = delta_time*sr
        sec=0
        silence=0
        msg_f=0
        aux_beg=0 #just to flag the beggining of the song
        out_path = os.path.dirname(user_info.main_path)+'/output'
    
        if synth ==1:
            #get parameters range for said synth and inst
            parameters_range = parameters_range_list
            
            #init synth class and choose type of synth
            inst_synth = SYNTH(synths[inst], sr, user_info, Parameters_types[synths[inst]],
                               parameters_range, synths_param[inst], 
                               round(window_size*samples_per_tick),
                               midi_info_list[inst], round(beg_compass*samples_per_tick), inst)
            
            #function that applies the synthesis
            Synthesize  = inst_synth.synthetize
        
        #silencio inicial para sinal começar no mesmo sitio que o midi (pode se retirar caso seja silencio muito grande...)
        Inst_signal[inst] = np.append(Inst_signal[inst], np.zeros(round(beg_compass*samples_per_tick)))
        
        if synth ==1:
            Inst_signal_synth[inst] = np.append(Inst_signal_synth[inst], np.zeros(round(beg_compass*samples_per_tick)))
        
        for msg in csv_strings[msg_beg:msg_end+1]:
            msg_list = msg.split(', ')
            
            #if msg in midi surpasses the number of sections, stop "for" loop
            if int(msg_list[1])>END:
                break
            elif msg_list[2] == note_on:
                note = int(msg_list[4])
                velocity = int(msg_list[-1])
                
                if aux_beg==0:
                    msg_i = max(round(beg_compass*samples_per_tick),
                                round(int(msg_list[1])*samples_per_tick))
                    
                    silence= msg_i-round(beg_compass*samples_per_tick)
                    aux_beg=1
                else:
                    msg_i = round(int(msg_list[1])*samples_per_tick) #convert to sampling 'time'
                    silence = msg_i-msg_f

            elif msg_list[2] == note_off:
                if (int(msg_list[1]) > (window_size*(sec+1)) + beg_compass) and (sec+1<n_sections):
                    sec+=1 #new section
        
                msg_f = round(int(msg_list[1])*samples_per_tick) #convert to sampling 'time'
                duration = msg_f-msg_i
                
                if sec<first_sec:
                    idx = 0
                else:
                    idx = sec - first_sec

                sample = Signals[Ind[inst][idx]]/sample_info.norm_coef[sample_info.sample_idx[Ind[inst][idx]][0]]
                
                original_note = Ind_pitch[inst][idx]
                tuning_steps = Ind_tuning[inst][idx]
                sample_shifted = pitch_shift(sample, note, sr, original_note + tuning_steps)

                sample_chopped, offset = envelope(sample_shifted , velocity, duration, sr)
                unclipped = Unclipping(sample_chopped, offset, smoothing_level=0.2)
                
                if synth == 1:         
                    pitch = librosa.midi_to_hz(note)
                    synthed_sample = Synthesize(unclipped.copy(), msg_i, msg_f, sec, pitch)
                    
                    #unclip again because of synthesized harsh sounds
                    synthed_sample = Unclipping(synthed_sample, 0, smoothing_level=0.4)
                    
                    # prepending the silence that came before the current note (space between notes)
                    if silence>0:
                        synthed_sample = np.append(np.zeros(silence), synthed_sample)
                    
                    #normalize sample
                    synthed_sample /= max(synthed_sample)
                    
                    comp_synthed_sample = comp.arctan_compressor(synthed_sample, 4)
                    Inst_signal_synth[inst]=np.append(Inst_signal_synth[inst], comp_synthed_sample)
                
                # prepending the silence that came before the current note (space between notes)
                if silence>0:
                    unclipped = np.append(np.zeros(silence), unclipped)
                
                compressed_sample = comp.arctan_compressor(unclipped, 1)             
                Inst_signal[inst]=np.append(Inst_signal[inst], compressed_sample)
                
            else:
                print('\nCorrupted MIDI file\n')
                break
    
        compressed_inst = comp.arctan_compressor(Inst_signal[inst], factor=1)
        
        #BEG and END on "sampling time"
        BEG_s=round(int(BEG)*samples_per_tick)
        END_s=round(int(END)*samples_per_tick)
        
        if synth==1:
            comp_synth_inst = comp.arctan_compressor(Inst_signal_synth[inst])
            comp_synth_inst = comp_synth_inst[BEG_s:min(len(comp_synth_inst),END_s)]
        
        compressed_inst = compressed_inst[BEG_s:min(len(compressed_inst),END_s)]
        
        if ex==-1:
            sound_filenames.append(out_path+'/output_music/NO_SYNTH/best_samples/'+inst_name[inst]+id_name+'_best'+'_sound.wav')
            sf.write(sound_filenames[-1] ,compressed_inst,sr)
            if synth==1:
                sound_filenames_synth.append(out_path+'/output_music/SYNTH/best_samples_synth/'+inst_name[inst]+id_name+'_best_synth'+'_sound.wav')
                sf.write(sound_filenames_synth[-1] ,comp_synth_inst,sr)
        else:
            sound_filenames.append(out_path+'/examples/example_samples/'+inst_name[inst]+'_ex'+ str(ex) +'_sound.wav')
            sf.write(sound_filenames[-1],compressed_inst,sr)
            
    Processed_Inst = Pedalboard_func(sound_filenames, sr)
    max_len = max(len(Processed_Inst[0].flatten()),len(Processed_Inst[1].flatten()),len(Processed_Inst[2].flatten()))
    
    for i in range(len(midi_files)):
        Processed_Inst[i] = np.append(Processed_Inst[i].flatten(),np.zeros(max_len-len(Processed_Inst[i].flatten())))  
    
    Final_Signal = np.array(Processed_Inst[0]+Processed_Inst[1]+Processed_Inst[2])/3#TEM DE TER O MM TAMANHO!!!
    Final_Signal_compressed = comp.arctan_compressor(Final_Signal, factor=1)
    
    if ex==-1:        
        if synth==1:
            Processed_Inst_synth = Pedalboard_func(sound_filenames_synth, sr)
            max_len_synth = max(len(Processed_Inst_synth[0].flatten())
                          ,len(Processed_Inst_synth[1].flatten()),
                          len(Processed_Inst_synth[2].flatten()))
            
            for i in range(len(midi_files)):
                Processed_Inst_synth[i] = np.append(Processed_Inst_synth[i].flatten(),
                                              np.zeros(max_len_synth-len(Processed_Inst_synth[i].flatten())))  
            
            Final_Signal_synth = np.array(Processed_Inst_synth[0]+Processed_Inst_synth[1]+Processed_Inst_synth[2])/3
            Final_Signal_synth_comp = comp.arctan_compressor(Final_Signal_synth, factor=1)      
            
            filename='Best'+'_sound'+'_synth'
            file_path = out_path+'/output_music/'+'SYNTH/'
            sf.write(file_path+filename+id_name+'.wav' , Final_Signal_synth_comp ,sr)
            #for the mp3 player
            write_mp3(file_path+'best_sound_mp3_synth/'+filename+'.mp3', sr, Final_Signal_synth_comp, normalized=True)  

            return file_path+'best_sound_mp3_synth/'+filename+'.mp3'
        
        #if there is no synth
        else:
            filename='Best'+'_sound'+id_name
            file_path = out_path+'/output_music/'+ 'NO_SYNTH/'
            sf.write(file_path+filename+'.wav' , Final_Signal_compressed, sr)        
            #for the mp3 player
            write_mp3(file_path+'best_sound_mp3/'+filename+'.mp3', sr, Final_Signal_compressed, normalized=True)

    else:
        filename = 'Example'+ str(ex) +'_sound'
        file_path = out_path+'/examples/example_music/'
        sf.write(file_path + filename+'.wav' , Final_Signal_compressed, sr)
        #for the mp3 player
        write_mp3(file_path+'MP3/'+filename+'.mp3', sr, Final_Signal_compressed, normalized=True)
        
        df_ind=pd.DataFrame()
        idx=0
        
        for sections in Ind:
            key = 'inst_'+str(idx)
            df_ind[key] = list(sections)
            idx+=1
             
        ind_csv_label = 'Example'+ str(ex) +'_sound.csv'
        df_ind.to_csv(out_path+'/GA1_training/Individuals/'+ 
                      ind_csv_label, index=False)
    
    return file_path+'best_sound_mp3/'+filename+'.mp3'