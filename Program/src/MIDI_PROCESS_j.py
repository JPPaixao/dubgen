"""
Created on Fri Apr 22 18:14:10 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: process MIDI: convert to array, get info (bpm, number of sections, mean note velocity...), and then apply MODS 
"""
import numpy as np
import os
import copy
import py_midicsv as m_csv

import MODIFY_MIDI as mm

#transform string into substrings
def first_substring(strings, substring):
    return next(i for i, string in enumerate(strings) if substring in string)

#gets arr element closest to desired value
def closest_value(arr, input_value):
  i = (np.abs(arr - input_value)).argmin()
  return arr[i]

# if value is different from 0, return 1
def Bit_difference(val):
    val[val!=0]=1
    return val

# quantize MIDI tick into grid
def quantize(tick, quantization, amount, start_q):
    #nearest quantization step (by remainder of division)
    nearest_q = int(np.round(tick/quantization,decimals=0)*quantization)
    
    #quantization condition
    if abs(tick-nearest_q )>((1-amount)*quantization/2) and start_q == 1:
        tick = nearest_q
    return tick


#correct MIDI csv string order
def correct_midi_csv(messages):
    note_on = 'Note_on_c'
    note_off = 'Note_off_c'
    
    for i in range(1,len(messages)):
        msg_list_prev = messages[i-1].split(', ')
        tick_prev = int(msg_list_prev[1])
        type_prev = msg_list_prev[2]
        msg_list = messages[i].split(', ')
        tick = int(msg_list[1])
        type_current = msg_list[2]
               
        if tick==tick_prev:
            #switch message order condition
            if type_prev == note_on and type_current == note_off:
                aux_msg = messages[i]
                messages[i] = messages[i-1]
                messages[i-1] = aux_msg
    return messages

#count midi notes
def count_notes(signal):
    note_count=0
    for i in range(len(signal)-1):
        if i==0 and signal[i]>0:
            note_count +=1
        elif signal[i] != signal[i+1] and signal[i+1]>0:
            note_count +=1
    return note_count


#convert MIDI csv to array
def MIDI_to_array(csv_strings, user_info, section_info, quantization_check):
    
    note_on = 'Note_on_c'
    note_off = 'Note_off_c'
    msg_beg = first_substring(csv_strings, note_on)
    msg_end = first_substring(csv_strings, 'End_track')-1 
    last_msg = csv_strings[msg_end].split(',')
    signal = np.zeros(int(last_msg[1])) #size=total number of ticks
    velocity = np.zeros(int(msg_end-msg_beg)) #vetor com as velocidades (uma por nota on)
    vel_idx=0
    last_note=-2
    last_note_off_tick=-2  
    msg_i = 0
    msg_f = 0
    header = csv_strings[0].split(',')
    ppq = int(header[-1]) #ticks per quarter note
    #quantization_check = 1 #1 for quantization to happen 
    qua_step = user_info.quantization_step
    quantization = qua_step*ppq #interval by which you quantize (0.5 qua_step--->quantize by eight notes)
    start_q = 1 #quantize both start and end
    amount = 0.7 #amount of quantization
    beg_compass = section_info.beginning
    end_song = section_info.end
    window_size = user_info.section_size*4*ppq
    sec = 0
    n_sections = int(np.ceil((end_song-beg_compass+1)/window_size))
    note_density = np.zeros(n_sections)
    vel_density = np.zeros(n_sections)
    
    stack_notes = [] #list with open notes
    stack_vel = []
    stack_ticks = [] #tick of each note start
    
    #put everything in correct order (if two consequent messages have the same tick, the order is off then on)
    csv_strings_notes = correct_midi_csv(csv_strings[msg_beg:msg_end+1])
    
    #process midi string messages
    for msg in csv_strings_notes:
        msg_list = msg.split(', ')
        tick = int(msg_list[1])
        note = int(msg_list[4])
        vel= int(msg_list[-1])

        if msg_list[2] == note_on:
            if len(stack_notes)>0 and stack_ticks[-1]<tick:
                if tick > (window_size*(sec+1)) + beg_compass :
                    sec+=1 #new section
                
                msg_i = stack_ticks[-1]
                msg_f = tick
                
                if quantization_check == 1:
                    msg_i = quantize(msg_i, quantization, amount, start_q)
                
                signal[msg_i:msg_f+1] = stack_notes[-1]             
                velocity[vel_idx] = stack_vel[-1]
                
                #just in case two equal notes have two consequent messages (same off and on tick)
                #(it is a thing that can happen in MIDI)
                if last_note == stack_notes[-1]:
                    signal[msg_i-1]=0
                    
                vel_idx += 1
                note_density[sec] +=1
                vel_density[sec] += stack_vel[-1]
                
            stack_notes.append(note)
            stack_vel.append(vel)
            stack_ticks.append(tick)
            
        elif msg_list[2] == note_off:
            if stack_notes[-1] == note:
                if tick > (window_size*(sec+1)) + beg_compass :
                    sec+=1 #new section
                    
                msg_i = stack_ticks[-1]
                msg_f = tick
                
                if quantization_check == 1:
                    msg_i = quantize(msg_i, quantization, amount, start_q)
                    
                signal[msg_i:msg_f+1] = note
                velocity[vel_idx] = stack_vel[-1]
                
                #just in case two equal notes have two consequent messages (same off and on tick)
                #(it is a thing that can happen in MIDI)               
                if (last_note==note and last_note_off_tick+1 >= msg_i)==True:
                    signal[msg_i-1]=0
                    
                vel_idx += 1
                note_density[sec] +=1
                vel_density[sec] += velocity[vel_idx-1]
                new_start = tick                      
                
                if count_notes(signal)!=vel_idx:
                    print('Error while Tracking Velocity Index')
                    vel_idx = count_notes(signal)
                
                last_note = note
                last_note_off_tick = tick
                
            else: new_start = stack_ticks[-1]          
            
            del stack_ticks[stack_notes.index(note)]
            del stack_vel[stack_notes.index(note)]
            stack_notes.remove(note)
            
            if len(stack_notes) > 0:
                for i in range(0,len(stack_ticks)):
                    stack_ticks[i] = new_start
                    
        elif msg_list[2] != 'Control_c':
            print('\nCorrupted file\n')
            break
        
        if count_notes(signal)!=vel_idx:
            print('Error while Tracking Velocity Index')
    
    for w in range(len(vel_density)):
        if vel_density[w]>0:
            vel_density[w] = vel_density[w]/note_density[w]

    vel_density = vel_density/127 #normalize by maximum velocity
    vel_density=np.ceil(vel_density*10)/10 #ceiling function in the first decimal
    
    note_density = note_density/(max(note_density))
    note_density=np.ceil(note_density*10)/10 #ceiling function in the first decimal
    
    velocity = velocity[:vel_idx]
    
    return signal, velocity, ppq, msg_beg, note_density, vel_density


#check if all elements of list are equal
def all_equal_ivo(lst):
    return not lst or lst.count(lst[0]) == len(lst)

#analyze sections of MIDI
def Section_detection(signal, ppq, window_size, section_info):
    beg_compass = section_info.beginning
    if len(signal)<section_info.end:
        signal = np.append(signal,np.zeros(section_info.end-len(signal)))
    
    #only analyse midi starting in that compass
    sig = signal[max(0,beg_compass-1):-1]
    #number of windows
    n_windows = int(np.floor(len(sig)/window_size))
    
    #if there is more notes past the last section (because it passed the integer division of parts...)
    if int(np.floor(len(sig)/window_size))<(len(sig)/window_size):
        n_windows+=1 #add window for the music ending
        #fill the signal with zeros at the end to fill that section
        sig = np.append(signal,np.zeros((n_windows*window_size)-len(sig)))

        
    asm1 = np.zeros((n_windows,n_windows))
    section_match = np.ones(n_windows, dtype='int')*[-1]
    
    
    #coefficent that estabilishes when a section is considered unique (too different from other windows)
    #the bigger monotomy, the harder it is to determine a unique section
    monotomy=0.2
    
    #computing ASM and section matches
    if n_windows>1:
        for i in range(0,n_windows):
            window = sig[i*window_size:(i+1)*window_size]
            for j in range(0,n_windows):
                asm1[i][j] =  np.sum(Bit_difference(sig[j*window_size:(j+1)*window_size] - window))/window_size
            
            asm1[i][i] = float('inf')
            
            if min(asm1[i])<monotomy:
                section_match[i] = np.argmin(asm1[i])

    pairs = np.argwhere(asm1 < 0.5)
    # Expand the pairs to include transitive pairs
    for i, j in pairs:
        for k in np.argwhere(pairs[:, 0] == j):
            if k[0] != i:
                pairs = np.concatenate((pairs, [[i, pairs[k[0]][1]]]), axis=0)
    # Remove duplicate pairs and self-pairs
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    
    # Create a dictionary to hold the class assignments
    class_dict = {}
    
    # Assign each element to a class
    class_num = 0
    for i in range(asm1.shape[0]):
        if i not in class_dict:
            class_num += 1
            class_dict[i] = class_num
            for j in np.argwhere(pairs[:, 0] == i):
                class_dict[pairs[j[0]][1]] = class_num
    
    # Create a vector of labels
    ordered_matches = np.zeros(asm1.shape[0], dtype=int)
    for i, j in class_dict.items():
        ordered_matches[i] = j - 1

    return ordered_matches, sig, n_windows


#convert MIDI back to csv file
def array_to_CSV(signal, velocity, csv_strings, msg_beg):
    note_count=0
    #create new csv midi file with same header
    csv_out = csv_strings[0:msg_beg]
    off_aux=0
    
    for i in range(len(signal)-1):
        if i==0 and signal[0]>0:
            if note_count > len(velocity)-1:
                print('Error while Tracking Note Index')
                
            note_on_string = "1, %d, Note_on_c, 0, %d, %d " %(i+1, signal[i+1], velocity[note_count])
            csv_out.append(note_on_string)
            note_count+=1
            off_aux=signal[i]
            
        elif signal[i] != signal[i+1] and signal[i+1]>0 :
            #note_on msg
            if signal[i]>0:
                note_off_string = "1, %d, Note_off_c, 0, %d, 0 " %(i, signal[i])
                csv_out.append(note_off_string)
                off_aux=0  
            if note_count > len(velocity)-1:
                print('Error while Tracking Note Index')
                
            note_on_string = "1, %d, Note_on_c, 0, %d, %d " %(i+1, signal[i+1], velocity[note_count])
            csv_out.append(note_on_string)
            note_count+=1
            off_aux= signal[i+1]
            
        elif signal[i] != signal[i+1] and signal[i+1]==0 :
            #note_off msg
            note_off_string = "1, %d, Note_off_c, 0, %d, 0 " %(i, signal[i])
            csv_out.append(note_off_string)
            off_aux=0
    
    if off_aux!=0:
        note_off_string = "1, %d, Note_off_c, 0, %d, 0 " %(len(signal), off_aux)
        csv_out.append(note_off_string)
        
    #append end of file
    end_track_string = "1, %d, End_track " %(len(signal)) 
    csv_out.append(end_track_string)
    end_of_file="0, 0, End_of_file"
    csv_out.append(end_of_file)

    return csv_out


# adjust number of sections to desired length
def adjust_section_sequence(sequence, desired_length):
    length = len(sequence)
    
    if length >= desired_length:
        return sequence[:desired_length]
    else:
        sequence=list(sequence)
        repetitions = desired_length // length
        remainder = desired_length % length
        adjusted_sequence = sequence * repetitions + sequence[:remainder]
        return np.array(adjusted_sequence)

    return sequence


#class with MIDI info
class info_midi:
    def __new__(cls, *args, **kwargs):
        #print("1. Create a new instance of info_midi.")
        return super().__new__(cls)
    
    def __init__(self, n_sections, section_type, ppq, midi_signal, velocity, csv_out, beg_compass):
        self.n_sections = n_sections
        self.section_type = section_type
        self.ppq = ppq
        self.midi_signal = midi_signal
        self.velocity = velocity
        self.csv_strings = csv_out
        self.beg_compass = beg_compass

#process MIDI and apply MIDI MODs
def PROCESS_MIDI(user_info, midi_in, section_info, instrument, folder):

    os.chdir(user_info.midi_folder+folder)
    
    csv_in = m_csv.midi_to_csv(midi_in)
    
    #convert MIDI strings to array
    signal_in, velocity, ppq, msg_beg, note_density, vel_density = MIDI_to_array(csv_in, user_info, 
                                                   section_info, 0)
    
    window_size = user_info.section_size*4*ppq 
    
    #Section Detection
    section_type, sig, n_sections = Section_detection(signal_in, ppq, window_size, section_info)    
    
    #MIDI MODIFICATION
    mod_midi_signal, mod_velocity = mm.MIDI_MODIFY(signal_in.copy(), velocity,
                                                   note_density, vel_density, section_info, window_size, ppq, user_info)
    
    #repeat modification (n times respecting the fb (feedback) parameter)
    for i in range(int(user_info.fb)):
        mod_midi_signal, mod_velocity = mm.MIDI_MODIFY(mod_midi_signal, mod_velocity, note_density,
                                                       vel_density, section_info, window_size, ppq, user_info)
    
    #convert back to csv 
    csv_out = array_to_CSV(mod_midi_signal, mod_velocity, csv_in, msg_beg)
    # Parse the CSV output of the previous command back into a MIDI file
    midi_object = m_csv.csv_to_midi(csv_out)
       
    # Save the parsed MIDI file to disk
    filename = os.path.splitext(midi_in)[0]
    filename += '_OUT.mid'  
    inst_ref = ['BASS_', 'HARMONY_', 'MELODY_']
    filepath = os.path.dirname(user_info.main_path)+'/output/output_MIDI/'+inst_ref[instrument]+ filename
    
    with open(filepath, "wb") as output_file:
        midi_writer = m_csv.FileWriter(output_file)
        midi_writer.write(midi_object)
     
    max_sec = 5   
    section_type_out = adjust_section_sequence(copy.deepcopy(section_type), max_sec)
    
    #with info about the midi signal with MOD  
    midi_info_out = info_midi.__new__(info_midi)    
    midi_info_out.__init__(max_sec, section_type_out, ppq, mod_midi_signal, mod_velocity, csv_out, section_info.beginning)
    
    #with info about the midi signal without MOD
    midi_info_in = info_midi.__new__(info_midi)
    midi_info_in.__init__(n_sections, section_type, ppq, mod_midi_signal, mod_velocity, csv_out, section_info.beginning)    
    
    return midi_info_out, midi_info_in