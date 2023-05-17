# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:41:40 2022

@author: JP
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def MOD_RULES(previous, current, state):
    #0: init; 1: B; 2:A; 3:C
    
    if previous>current and state == 1:
        state = 2
    elif previous>current and state == 2:
        state = 1
    elif previous>current and state == 3:
        state = 1
    elif previous<current and state == 1:
        state = 2
    elif previous<current and state == 2:
        state = 3
    elif previous<current and state == 3:
        state = 3
    elif previous==current and state == 1:
        state=2
    elif previous==current and state == 2:
        state=3
    
    return state

def Apply_Rules(density):
    state_vec=np.zeros(len(density))
    
    #init state:
    if density[0] > np.mean(density):
        state_vec[0] = 2
    else:
        state_vec[0] = 1
    
    for i in range(len(density)-1):
        state_vec[i+1] = MOD_RULES(density[i], density[i+1], state_vec[i])
    return state_vec

def MOD_evolution(state_vec, signal_size, window_size, ceiling=0.8):
    #mod_evolution = np.zeros(signal_size)
    mod_evolution = np.zeros(len(state_vec)*window_size)
    
    for sec in range(len(state_vec)):
        if state_vec[sec]==1:
            func = ceiling/2 #ceiling is the maximum probability of mod
        elif state_vec[sec]==3:
            func = ceiling
        elif state_vec[sec]==2:
            func = np.linspace(0, ceiling, num = window_size)
            
        mod_evolution[window_size*(sec):window_size*(sec+1)] = func
        
# =============================================================================
#         plt.plot(mod_evolution)
#         plt.show()      
# =============================================================================
    return mod_evolution[:signal_size]

def Multiplication(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    gap = 6 #quantos ticks de gap entre notas (cada tick são 5.2ms e o limite de percepção audivel é 4.2ms)    
    #nearest_q = int(np.round(msg_i/quantization,decimals=0)*quantization) para  o caso de se precisar os timesteps certos (quantizados)
    
    vel_vec = np.array([velocity])
    
    tick_step = int(time_division*4*ppq) #(4*ppq means the tick size of a compass)
    
    if len(sig)-note_start > tick_step: #if the note is big enough to apply mod
        for i in range(note_start+tick_step, len(sig), tick_step): #saltar de time division em time division
            
            if sig[i]>0 and i!=note_start:  #if note will be created
                vel_vec=np.append(vel_vec, [velocity]) #append new velocity (NOT MODULATED YET)
                sig[i-gap-1:i-1] = np.zeros(gap) # fazer um corte ao sinal num tal gap (antes de i, claro)
    
    #print('num of vels:',np.shape(vel_vec))
    '''
    plt.plot(sig)
    plt.show()
    '''
    #print('MULT :D')
    #else: print('not big enough')    
    return sig, vel_vec 

def Exp_func_down(x):
    y = (1-(math.exp(1-x)))/(1-math.exp(1))
    return y

def Exp_func_up(x):
    y = ((math.exp(x))-1)/(math.exp(1)-1)
    return y

def Antecipation(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    vel_vec=np.array([velocity])
    
    start_step = int(time_division*4*ppq) #gives a small silence before the effect starts    
    
    if start_step < note_start:
        
        spaces_exp = list()
        
        for i in range(start_step+1, note_start+1):
            
            x = (i - start_step) /(note_start-start_step)
            spaces_exp.append(int(Exp_func_up(x)*start_step)+1)
            
        
        ant_note_len = int(time_division*ppq)
        
        #novo (retirar se voltar atras)
        sig[start_step:note_start] = np.ones(note_start-start_step)*sig[note_start+1]
        
        j = note_start
        
        while j > start_step:
            gap = spaces_exp[-(note_start - j)-1] #compute gap length
                             
            sig[ j-gap : j ] = np.zeros(gap) #apply gap
            
            j-= (gap + ant_note_len) #go back in signal
            
        
        n_notes=count_notes(sig)
        if n_notes>1:
            vel_vec = np.append(np.ones(n_notes-1)*velocity, vel_vec) #prepend velocity when note is created
            
        '''    
        plt.plot(sig)
        plt.show()
        '''
    
        #print('num of vels:',np.shape(vel_vec))
        '''
        plt.plot(sig)
        plt.show()
        '''
        #print('ANT :D')
    
    return sig, vel_vec

def Freeze(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    if note_start>0:
        mod_sig = np.zeros(note_start) #silence before the note
    else:
        mod_sig = np.array([])
   
    mod_sig = np.append(mod_sig, np.ones(len(sig)-note_start-1)*sig[note_start]) #append freezed note (-1 because of the next append)
    mod_sig = np.append(mod_sig, np.zeros(1)) #just so it has a separation from the next note
    
    vel_vec = np.array([velocity])
    
    return mod_sig, vel_vec


def Mute(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    
    mod_sig = np.zeros(len(sig))
    vel_vec=np.array([])
    
    return mod_sig, vel_vec


def Apply_mod(note, starting_note, ppq, quantization, intensity, velocity, next_note, note_end, rand_mod):
    if len(note)!=0:
        rand_mod = random.randint(0, 4) #which mod will ocurr
        
        '''
        plt.plot(note)
        plt.show()
        '''
        if rand_mod == 0:
            #print('mult')
            mod_note, new_vel = Multiplication(note, starting_note, ppq, 
                                1/8, quantization, intensity,
                                velocity) # apply mod to this interval
        elif rand_mod == 1:
            #print('mute')
            mod_note, new_vel = Mute(note, starting_note, ppq, 
                                1/4, quantization, intensity,
                                velocity) # apply mod to this interval
        elif rand_mod == 2:
            #print('freeze')
            mod_note, new_vel = Freeze(note, starting_note, ppq, 
                                 1/4, quantization, intensity,
                                 velocity) # apply mod to this interval    

            note_end = next_note-2 # ou -1??
            '''
            print('mute')
            mod_note, new_vel = Mute(note, starting_note, ppq, 
                                1/4, quantization, intensity,
                                velocity) # apply mod to this interval
            '''
            
        elif rand_mod == 3:
            #print('ant')
            mod_note, new_vel = Antecipation(note, starting_note, ppq, 
                                1/8, quantization, intensity,
                                velocity) # apply mod to this interval                
        elif rand_mod==4:
            #print('vel_mod!')
            mod_note = note
            new_vel = np.array([(intensity[starting_note]+0.3)*127]) #assign velocity depending of intensity evolution in the signal
       
        '''
        plt.plot(mod_note)
        plt.show()
        '''
    else:
        print('Error: signal with size 0')
    return mod_note, new_vel, note_end, rand_mod

def update_velocity(velocity, new_vel, note_count, rand_mod):
    velocity = np.delete(velocity, note_count)
    
    if (rand_mod == 0) or (rand_mod == 2) or (rand_mod == 3) or (rand_mod == 4): #mult or ant or vel
        velocity = np.insert(velocity, note_count, new_vel) #insert new velocities in the velocity vector, indexed by the note_count
        #print('len(new_vel) = ', len(new_vel))
        
        note_count += len(new_vel) -1 #-1 because there was a note already there
    else: note_count -=1 #in case of mute
    
    return velocity, note_count

 
def count_notes(signal):
    note_count=0
    
    for i in range(len(signal)-1):
        if i==0 and signal[i]>0:
            note_count +=1
        elif signal[i] != signal[i+1] and signal[i+1]>0:
            note_count +=1
    
    return note_count

def detect_on_note(signal): #detect the beginning of notes in the signal
    
    on_idxs=[]
    
    for i in range(1, len(signal)):
        if i==1 and signal[i-1]>0:
            on_idxs.append(i-1)
        elif signal[i] != signal[i-1] and signal[i] >0:
            on_idxs.append(i)
    
    if len(on_idxs)==0:
        print('no note in interval (detect_on_note function)')
            
    return on_idxs

def detect_off_note(signal): #detect the ending of notes in the signal
    
    off_idxs=[]
    
    for i in range(1, len(signal)):
        if signal[i] != signal[i-1] and signal[i-1] >0:
            off_idxs.append(i-1)
        elif i==len(signal)-1 and signal[i]>0:
            off_idxs.append(i)
            
    return off_idxs

def MIDI_MODIFY(signal, velocity, note_density, vel_density, section_info, window_size, ppq, user_info):
    
    state_prob = Apply_Rules(note_density) #get rule for each section
    state_intensity = Apply_Rules(vel_density)
    
    prob = MOD_evolution(state_prob, len(signal), window_size, user_info.mod) #create linspace with rules's evolution
    intensity = MOD_evolution(state_intensity, len(signal), window_size, 0.5)
    
    #ppq is the size of a quarter note (in ticks)!
    qua_step = user_info.quantization_step
    quantization = qua_step*ppq     #(não precisa de ser o qua_step...)
    
    last_note = 0 #step where the previous note ended
    next_note = 0 #step where next note starts
    note_end = 0
    
    note_count = int(0) #counting the pre existing notes to index velocities
    
    #signal = signal[-int(len(signal)/16)+5:] #just for testing
    '''
    plt.plot(signal)
    plt.show()
    '''
    signal_print = signal.copy()
    
    current_note = 0
    
    i=0
    
    on_idxs=detect_on_note(signal)
    off_idxs=detect_off_note(signal)
    
    '''
    plt.plot(signal)
    plt.scatter(on_idxs, np.ones(len(on_idxs)))
    plt.scatter(off_idxs, np.ones(len(off_idxs)))
    plt.show()
    '''
    on_idxs.append(len(signal)) #append idx of the last sample
    on_idxs.pop(0) #remove first note_beg
    off_idxs.insert(0,0) #prepend first sample
    off_idxs.pop() #remove last note_end
    
    mod = -1 # what type of midi mod had ocurred
    
    for i in range(0, len(on_idxs)):
        next_note = on_idxs[i]
        
        if mod==2: #o freeze mina o final da nota anterior
           last_note = note_end
        else:
            last_note = off_idxs[i]
        
        '''
        print('last_note: ', last_note)
        print('next_note: ', next_note)
        '''
        note = signal[last_note+1:next_note].copy()
        note_start_vec = detect_on_note(note)
        
        if len(note_start_vec)==0:
            print('no note in interval (detect_on_note function)')
        note_start=note_start_vec[0]
        
        #print('note_start: ', note_start)
        
        '''
        plt.plot(note)
        plt.show()
        '''
        note_beg = last_note+note_start
        
        #i>0 so that the first note doesnt suffer from a mod (it would mess with the beg_compass tracking)
        if random.uniform(0, 1) < prob[note_beg] and i>0:
            '''
            plt.figure()
            plt.title('before mod. last_note: ' + str(last_note))
            plt.plot(signal[last_note+1-10 : next_note+10])
            plt.axvline(x = 10, color = 'b')
            plt.axvline(x = 10 + next_note+1-last_note, color = 'b')
            plt.show()
            '''
            #print('last_note: ', last_note)
            #print('next_note: ', next_note)
            
            mod_note, mod_vel, note_end, mod = Apply_mod(note.copy(), note_start, ppq, quantization,
                                                intensity[last_note+1 : next_note],
                                                velocity[int(note_count)], next_note, note_end, mod)
            
            if len(mod_note)!=len(note):
                print('different sizes')
            

            #print('added notes: ', count_notes(mod_note))
            #print('added velocities: ', len(mod_vel))
            #print('\n')
            
            
            signal[last_note+1 : next_note] = mod_note.copy()
            
            '''
            plt.figure()
            plt.title('after mod. last_note: '+ str(last_note))
            plt.plot(signal[last_note+1-10 : next_note+10])
            plt.axvline(x = 10, color = 'b')
            plt.axvline(x = 10 + next_note+1- last_note, color = 'b')
            plt.show()
            '''
            
            if count_notes(mod_note)!=len(mod_vel):
                print('\ndebug mod_vel here\n')
            
            #INSERIR NOVAS VELOCIDADES
            velocity, note_count = update_velocity(velocity.copy(), mod_vel, note_count, mod)
            
            #print('\nnumber of notes: ', count_notes(signal))           
            #print('number of velocities: ', len(velocity), '\n')
            
            if count_notes(signal)!=len(velocity):
                
                plt.figure()
                plt.title('mod signal window. last_note: ' + str(last_note))
                plt.plot(signal[last_note+1-1000 : next_note+1000])
                plt.axvline(x = 1000, color = 'b')
                plt.axvline(x = 1000 + next_note+1-last_note, color = 'b')
                plt.show()
                
                plt.figure()
                plt.title('og signal window. last_note: ' + str(last_note))
                plt.plot(signal_print[last_note+1-1000 : next_note+1000])
                plt.axvline(x = 1000, color = 'b')
                plt.axvline(x = 1000 + next_note+1-last_note, color = 'b')
                plt.show()
                '''
                ##########################################################
                import winsound
                import time
                frequency = 1500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                time.sleep(1.1)
                winsound.Beep(frequency-500, duration)
                ##########################################################
                '''
                print('problem is in update_velocity() function!')
                
        else: mod = -1
        
        
        note_count+=1
        '''
        print('note_count: ', note_count)
        print('velocity count: ', len(velocity))
        print('\n')
        '''
        
    #print('hey')
    '''
    plt.plot(signal)
    plt.show()
    '''
    n_notes = count_notes(signal)
    if n_notes < len(velocity): #velocity vector always has one element more?? mas quando conto a olho parece que tou a contar bem...
        velocity = velocity[:n_notes]
        print('final different number of notes. Real:',n_notes,'; Counted:', len(velocity) )
    
    return signal, velocity
