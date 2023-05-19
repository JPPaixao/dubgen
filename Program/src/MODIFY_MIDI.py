"""
Created on Tue Jul 12 12:41:40 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: modidy MIDI. Applies MIDI effects modulated by note and velocity density 
"""
import numpy as np
import random
import math

#Sets Rules for MIDI Mod probability function
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


# apply rules determined by note/velocity density
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


# given a MOD rule, creates a MIDI Mod Probability window
def MOD_evolution(state_vec, signal_size, window_size, ceiling=0.8):
    mod_evolution = np.zeros(len(state_vec)*window_size)
    
    #ceiling is the maximum probability of mod
    for sec in range(len(state_vec)):
        if state_vec[sec]==1:
            func = ceiling/2 
        elif state_vec[sec]==3:
            func = ceiling
        elif state_vec[sec]==2:
            func = np.linspace(0, ceiling, num = window_size)
            
        mod_evolution[window_size*(sec):window_size*(sec+1)] = func

    return mod_evolution[:signal_size]

# Multiplication effect
def Multiplication(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    # tick gap size between notes (each tick is 5.2ms and the minimum audible perception time is around 4.2ms)
    gap = 6
    vel_vec = np.array([velocity])
    
    #(4*ppq means the tick size of a compass)
    tick_step = int(time_division*4*ppq)
    
    #if the note is big enough to apply mod
    if len(sig)-note_start > tick_step: 
        #jump from time division to time division
        for i in range(note_start+tick_step, len(sig), tick_step):
            #if note will be created
            if sig[i]>0 and i!=note_start:
                #append new velocity
                vel_vec=np.append(vel_vec, [velocity])
                #cut the signal given the respective gap
                sig[i-gap-1:i-1] = np.zeros(gap)
      
    return sig, vel_vec 

# exponentially decrescent function
def Exp_func_down(x):
    y = (1-(math.exp(1-x)))/(1-math.exp(1))
    return y

# exponentially crescent function
def Exp_func_up(x):
    y = ((math.exp(x))-1)/(math.exp(1)-1)
    return y

# Anticipation Effect
def Antecipation(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    vel_vec=np.array([velocity])
    
    #gives a small silence before the effect starts  
    start_step = int(time_division*4*ppq)  
    
    if start_step < note_start:
        spaces_exp = list() 
        
        for i in range(start_step+1, note_start+1):
            x = (i - start_step) /(note_start-start_step)
            spaces_exp.append(int(Exp_func_up(x)*start_step)+1)
        
        ant_note_len = int(time_division*ppq)
        sig[start_step:note_start] = np.ones(note_start-start_step)*sig[note_start+1]
        j = note_start
        
        while j > start_step:
            gap = spaces_exp[-(note_start - j)-1] #compute gap length
            sig[ j-gap : j ] = np.zeros(gap) #apply gap
            j-= (gap + ant_note_len) #go back in signal
            
        n_notes=count_notes(sig)
        if n_notes>1:
            #prepend velocity when note is created
            vel_vec = np.append(np.ones(n_notes-1)*velocity, vel_vec)
    
    return sig, vel_vec


# Freeze Effect
def Freeze(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    if note_start>0:
        mod_sig = np.zeros(note_start) #silence before the note
    else:
        mod_sig = np.array([])
   
    mod_sig = np.append(mod_sig, np.ones(len(sig)-note_start-1)*sig[note_start]) #append freezed note (-1 because of the next append)
    mod_sig = np.append(mod_sig, np.zeros(1)) #just so it has a separation from the next note
    
    vel_vec = np.array([velocity])
    
    return mod_sig, vel_vec

# Mute Effect
def Mute(sig, note_start, ppq, time_division, quantization, intensity, velocity):
    mod_sig = np.zeros(len(sig))
    vel_vec=np.array([])
    
    return mod_sig, vel_vec

# apply MIDI Mods to signal
def Apply_mod(note, starting_note, ppq, quantization, intensity, velocity, next_note, note_end, rand_mod):
    if len(note)!=0:
        #determine which mod will ocurr randomly
        rand_mod = random.randint(0, 4)

        if rand_mod == 0:
            #MULTIPLICATION
            mod_note, new_vel = Multiplication(note, starting_note, ppq, 
                                1/8, quantization, intensity,
                                velocity)
        elif rand_mod == 1:
            #MUTE
            mod_note, new_vel = Mute(note, starting_note, ppq, 
                                1/4, quantization, intensity,
                                velocity)
        elif rand_mod == 2:
            #FREEZE
            mod_note, new_vel = Freeze(note, starting_note, ppq, 
                                 1/4, quantization, intensity,
                                 velocity)   
            note_end = next_note-2
            
        elif rand_mod == 3:
            #ANTICIPATION
            mod_note, new_vel = Antecipation(note, starting_note, ppq, 
                                1/8, quantization, intensity,
                                velocity)               
        elif rand_mod==4:
            #VELOCITY MOD
            mod_note = note
            #assign velocity depending on the signal's intensity evolution
            new_vel = np.array([(intensity[starting_note]+0.3)*127])
       
    else:
        print('Error: signal with size 0')
    return mod_note, new_vel, note_end, rand_mod


#update velocity vector, because of MIDI Mod's added notes
def update_velocity(velocity, new_vel, note_count, rand_mod):
    velocity = np.delete(velocity, note_count)
    
    #mult or ant or vel
    if (rand_mod == 0) or (rand_mod == 2) or (rand_mod == 3) or (rand_mod == 4):
        #insert new velocities in the velocity vector, indexed by the note_count
        velocity = np.insert(velocity, note_count, new_vel)  
        note_count += len(new_vel) -1 #-1 because there was a note already there
    else: note_count -=1 #in case of mute
    
    return velocity, note_count


 #count notes on a signal
def count_notes(signal):
    note_count=0
    for i in range(len(signal)-1):
        if i==0 and signal[i]>0:
            note_count +=1
        elif signal[i] != signal[i+1] and signal[i+1]>0:
            note_count +=1
    return note_count

# detect the beginning of notes in the signal
def detect_on_note(signal):
    on_idxs=[]
    for i in range(1, len(signal)):
        if i==1 and signal[i-1]>0:
            on_idxs.append(i-1)
        elif signal[i] != signal[i-1] and signal[i] >0:
            on_idxs.append(i)
    
    if len(on_idxs)==0:
        print('no note in interval (detect_on_note function)')
            
    return on_idxs


# detect the ending of notes in the signal
def detect_off_note(signal): 
    off_idxs=[]
    for i in range(1, len(signal)):
        if signal[i] != signal[i-1] and signal[i-1] >0:
            off_idxs.append(i-1)
        elif i==len(signal)-1 and signal[i]>0:
            off_idxs.append(i)
            
    return off_idxs

# modify MIDI signal given its note and velocity density
def MIDI_MODIFY(signal, velocity, note_density, vel_density, section_info, window_size, ppq, user_info):
    
    state_prob = Apply_Rules(note_density) #get rule for each section
    state_intensity = Apply_Rules(vel_density)
    
    prob = MOD_evolution(state_prob, len(signal), window_size, user_info.mod) #create linspace with rules's evolution
    intensity = MOD_evolution(state_intensity, len(signal), window_size, 0.5)
    
    #ppq is the size of a quarter note (in ticks)!
    qua_step = user_info.quantization_step
    quantization = qua_step*ppq
    
    last_note = 0 #step where the previous note ended
    next_note = 0 #step where next note starts
    note_end = 0 #step where current note starts
    note_count = int(0) #counting the pre existing notes to index velocities   
    
    on_idxs=detect_on_note(signal)
    off_idxs=detect_off_note(signal)
    on_idxs.append(len(signal)) #append idx of the last sample
    on_idxs.pop(0) #remove first note_beg
    off_idxs.insert(0,0) #prepend first sample
    off_idxs.pop() #remove last note_end
    
    mod = -1 # what type of midi mod had ocurred
    i=0
    
    for i in range(0, len(on_idxs)):
        next_note = on_idxs[i]
        
        if mod==2:
           last_note = note_end
        else:
            last_note = off_idxs[i]
        
        note = signal[last_note+1:next_note].copy()
        note_start_vec = detect_on_note(note)
        
        if len(note_start_vec)==0:
            print('no note in interval (detect_on_note function)')
            
        note_start=note_start_vec[0]
        note_beg = last_note+note_start
        
        #i>0 so that the first note doesnt suffer from a mod (it would mess with the beg_compass tracking)
        if random.uniform(0, 1) < prob[note_beg] and i>0:

            mod_note, mod_vel, note_end, mod = Apply_mod(note.copy(), note_start, ppq, quantization,
                                                intensity[last_note+1 : next_note],
                                                velocity[int(note_count)], next_note, note_end, mod)
            
            if len(mod_note)!=len(note):
                print('MIDI MOD ERROR: different note sizes')
        
            signal[last_note+1 : next_note] = mod_note.copy()
            
            if count_notes(mod_note)!=len(mod_vel):
                print('MIDI MOD ERROR')
            
            #insert new velocities
            velocity, note_count = update_velocity(velocity.copy(), mod_vel, note_count, mod)
            
            if count_notes(signal)!=len(velocity):
                print('MIDI MOD ERROR: number of notes doesnt match number of velocities')
        else: mod = -1

        note_count+=1

    n_notes = count_notes(signal)
    if n_notes < len(velocity): 
        velocity = velocity[:n_notes]
        print('MIDI MOD ERROR: different number of notes. Real:',n_notes,'; Counted:', len(velocity) )
    
    return signal, velocity
