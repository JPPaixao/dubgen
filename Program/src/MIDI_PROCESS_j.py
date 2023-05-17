# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:14:10 2022

@author: JP
"""
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import py_midicsv as m_csv

import MODIFY_MIDI as mm
import copy

def first_substring(strings, substring):
    return next(i for i, string in enumerate(strings) if substring in string)

def plot_midi_signal(signal): #melhorar plot :)
    plt.figure()
    plt.plot(signal, marker='o', linestyle='')
# =============================================================================
#     plt.ylim((0,max(signal)+1))
#     plt.xlim((0,len(signal)))
# =============================================================================
    return

def plot_asm(matrix):
    [m,n] = np.shape(matrix)

    #Colour Map using Matrix
    plt.figure()
    plt.imshow(matrix, alpha=0.8, cmap=plt.cm.Blues)
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(m))
    plt.xlabel('Window 2')
    plt.ylabel('Window 1')
    plt.title('Similarity of Windows')
    plt.show()
    return

def closest_value(arr, input_value):
 
  i = (np.abs(arr - input_value)).argmin()
 
  return arr[i]

def Bit_difference(val):
    val[val!=0]=1
    return val

def quantize(tick, quantization, amount, start_q):
    nearest_q = int(np.round(tick/quantization,decimals=0)*quantization) #nearest quantization step (by resto of division)
    #if msg_i > nearest_q+((1-amount)*quantization/2) or msg_i < nearest_q+((1-amount)*quantization/2):
    if abs(tick-nearest_q )>((1-amount)*quantization/2) and start_q == 1: #quantization condition
        tick = nearest_q
    return tick

#put everything in correct order (if two consequent messages have the same tick, the order is off then on)
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
            #condiçaõ para trocar mensagens de ordem!
            if type_prev == note_on and type_current == note_off:
                #checar se nao ha stress the shallow/deep copy!!!!
                aux_msg = messages[i]
                messages[i] = messages[i-1]
                messages[i-1] = aux_msg
    
    return messages

def count_notes(signal):
    note_count=0
    
    for i in range(len(signal)-1):
        if i==0 and signal[i]>0:
            note_count +=1
        elif signal[i] != signal[i+1] and signal[i+1]>0:
            note_count +=1
    
    return note_count

def MIDI_to_array(csv_strings, user_info, section_info, quantization_check):
    
    note_on = 'Note_on_c'
    note_off = 'Note_off_c'
    msg_beg = first_substring(csv_strings, note_on)
    msg_end = first_substring(csv_strings, 'End_track')-1 
    
    # init vector for asm comparison
    last_msg = csv_strings[msg_end].split(',')
    signal = np.zeros(int(last_msg[1])) #size=total number of ticks
    velocity = np.zeros(int(msg_end-msg_beg)) #vetor com as velocidades (uma por nota on)
    vel_idx=0
    
    msg_i = 0
    msg_f = 0
    
    header = csv_strings[0].split(',')
    ppq = int(header[-1]) #ticks per quarter note
    
    ##############################################################
    #quantization_check = 1 #1 for quantization to happen 
    ##############################################################
    
    qua_step = user_info.quantization_step #quantization step (referencia é quarter note (1)) ...eight note=0.5...
    quantization = qua_step*ppq #interval by which you quantize (0.5 qua_step--->quantize by eight notes)
    start_q = 1 #quantize both start and end
    end_q = 1
    amount = 0.7
    
    beg_compass = section_info.beginning
    end_song = section_info.end
    window_size = user_info.section_size*4*ppq
    sec = 0
    n_sections = int(np.ceil((end_song-beg_compass+1)/window_size))
    note_density = np.zeros(n_sections)
    vel_density = np.zeros(n_sections)
    
    prev_msg = 0 #previous msg 0:note off 1:note on    

    last_note=-2
    last_note_off_tick=-2 #tem de ser menor que -1!

    stack_notes = [] #list with open notes
    stack_vel = []
    stack_ticks = [] #tick of each note start
    
    #FAZER DEBUG DA FUNÇAO SEGUINTE!!
    
    #put everything in correct order (if two consequent messages have the same tick, the order is off then on)
    csv_strings_notes = correct_midi_csv(csv_strings[msg_beg:msg_end+1])
    
    #isto pode dar stresses se houver notas tocadas ao mm tempo (dá prioridade à ultima nota tocada)
    for msg in csv_strings_notes:
        
        msg_list = msg.split(', ')
        tick = int(msg_list[1])
        note = int(msg_list[4])
        vel= int(msg_list[-1])
        
        #if tick == 4436:
        #    print('debug midi to array aqui')
        
        if msg_list[2] == note_on:
            
            #print('note on: ', note, ', tick: ', tick)
            
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
            #print('note off: ', note, ', tick: ', tick)
            
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
                
                new_start = tick                        #rever isto
                
                if count_notes(signal)!=vel_idx:
                    print('Error while Tracking Velocity Index')
                    vel_idx = count_notes(signal)
                
                last_note = note
                last_note_off_tick = tick
                
            else: new_start = stack_ticks[-1]           #rever isto
            
            del stack_ticks[stack_notes.index(note)]
            del stack_vel[stack_notes.index(note)]
            stack_notes.remove(note)
            
            #print('stack: ', stack_notes)
            
            if len(stack_notes) > 0:
                for i in range(0,len(stack_ticks)):
                    stack_ticks[i] = new_start          #rever isto. Tem de estar de acordo com o inicio da ultima nota (seforem duas notas seguidas)
                    
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



def all_equal_ivo(lst): #check if all elements of list are equal
    return not lst or lst.count(lst[0]) == len(lst)

def Section_detection(signal, ppq, window_size, section_info):
    ############################################################################
        #SECTION DETECTION (assumindo bpm e time signature)
    ############################################################################
    
    #TODO : IMPLEMENTAR DETETAR ANACRUSIS!!!!!!!!!
    
    beg_compass = section_info.beginning #TESTING THE UNIVERSAL BEGINnING (SAME FOR EVERY INSTRUMENT)
    
    if len(signal)<section_info.end: #TESTING THE UNIVERSAL BEGINnING (SAME FOR EVERY INSTRUMENT)
        signal = np.append(signal,np.zeros(section_info.end-len(signal)))
    
    #only analyse midi starting in that compass
    sig = signal[max(0,beg_compass-1):-1]
    
    
    #section_bars = 1 #vai ter de haver um bar size para cada midi? para cada midi??
    
    #window_size é o tamanho generico de uma secção de uma musica: 4 a 16 bars
    n_windows = int(np.floor(len(sig)/window_size)) #number of windows
    
    if int(np.floor(len(sig)/window_size))<(len(sig)/window_size): #if there is more music past the last section (because it passed the integer division of parts...)
        n_windows+=1 #add window for the music ending
        sig = np.append(signal,np.zeros((n_windows*window_size)-len(sig))) #fill the signal with zeros at the end to fill that section

        
    asm1 = np.zeros((n_windows,n_windows))
    section_type = np.ones(n_windows, dtype='int')*[-2] #init vector with sections type
    #sec=0# section class
    section_match = np.ones(n_windows, dtype='int')*[-1]
    
    monotomy=0.2 #coefficent determined by the user (probably a slider) that estabilishes when a section is considered unique (too different from other windows)
    #the bigger monotomy, the harder it is to determine a unique section
    
    if n_windows>1: #computing ASM and section matches
        for i in range(0,n_windows):
            window = sig[i*window_size:(i+1)*window_size]
            for j in range(0,n_windows):
                asm1[i][j] =  np.sum(Bit_difference(sig[j*window_size:(j+1)*window_size] - window))/window_size
            
            asm1[i][i] = float('inf')
            
            if min(asm1[i])<monotomy:
                section_match[i] = np.argmin(asm1[i])
            #if i==0: plot_midi_signal(window)
        #plot_asm(asm1)
    '''
    #turning the "different parts (-1)" into a normal part (basically the section corresponds to the idx)
    section_match[np.where(section_match == -1)[0]] = np.where(section_match == -1)[0]
    
    sec_list = list([np.array([0,section_match[0]], dtype='int')]) #list with bag of sections #init with first element
    aux=0
    sum_s=2 #sum of sections (starts with two) (used this to stop the cycle when all sections are analyzed)
    
    every_s = list() #group all visited section
    
    for j in range(len(section_match)):
        if (((j in every_s) and (section_match[j] in every_s)) ==False) or j==0:
            if j!=0:
                if j==section_match[j]:
                    sec_list.append(np.array([j], dtype='int'))
                    sum_s+=1
                    aux+=1
                else:
                    sec_list.append(np.array([j,section_match[j]], dtype='int'))
                    sum_s+=2
                    aux+=1      
            for w in range(j+1,len(section_match)):
                if w in sec_list[aux]:
                    if (section_match[w] in sec_list[aux])==False:
                        sec_list[aux]=np.append(sec_list[aux],section_match[w])
                        sum_s+=1
                if section_match[w] in sec_list[aux]:
                    if (w in sec_list[aux])==False:
                        sec_list[aux] = np.append(sec_list[aux],w)
                        sum_s+=1
        every_s += list(sec_list[aux])
        if sum_s >=len(section_match): break
        
        
    for s in range(len(sec_list)):
        section_type[sec_list[s]] = s
    
# =============================================================================
#     section_type[0]=sec
#     for s in range(0, len(section_type)): #SECTION attribution (detect pairings)
#         if section_match[s]==-1 and s>0: #init section that wasn´t labeled yet
#             sec+=1
#             section_type[s] = sec #Create new section
#         else:
#             if section_type[s]== -1: #if section is unique
#                 sec+=1
#                 section_type[s]= sec
#             section_type[np.where(section_match==sec)[0]] = sec
# =============================================================================

# =============================================================================        
#    ordered_matches = np.ones(len(section_match), dtype='int')*(-2)
#    section_aux=0
#     for s in range(0, len(ordered_matches)): #order number of section (indexes range from 0 to...)
#         if section_match[s]>section_aux:    
#             ordered_matches[np.where(section_match==section_match[s])] = section_aux
#             section_aux += 1    
# =============================================================================

    
    ordered_matches = section_type

    if all_equal_ivo(list(ordered_matches))==True: #if all elements are equal
        ordered_matches=np.zeros(len(ordered_matches), dtype='int') #correct exception where all sections are not of zero type
    '''
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

def array_to_CSV(signal, velocity, csv_strings, msg_beg):
    #CONVERSÃO PARA MIDI TEM DE TER A VELOCIDADE!! ATENÇÃO A ISSO
    #POR ISSO, CONVERSÃO PARA "SINAL" NÃO PODE PERDER NOTAS!! (PARA HAVER MATCH DE VELOCIDADES)
    
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
    #convert csv to midi
    return csv_out

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

def PROCESS_MIDI(user_info, midi_in, section_info, instrument, folder): #ISTO TALVEZ ASSUME QUE TODOS OS MIDIS VAO SER DO MESMO TAMANHO!!
    # Load the MIDI file and parse it into CSV format
    #import sys
    
    #sys.path.append(midi_folder)
    os.chdir(user_info.midi_folder+folder)
    
    csv_in = m_csv.midi_to_csv(midi_in)
    
    #BLINDAR PARA VARIOS TIPOS DE MIDI FILES!!! IGNORAR MAIS QUE UM CANAL ETC...
    
    
    #converter notas e velocity em representação de array
    signal_in, velocity, ppq, msg_beg, note_density, vel_density = MIDI_to_array(csv_in, user_info, 
                                                   section_info, 0) #NÃO TÁ A QUANTIZAR!
    #plot_midi_signal(signal[0:1000])
    
    window_size = user_info.section_size*4*ppq 
    
    #Section Detection
    section_type, sig, n_sections = Section_detection(signal_in, ppq, window_size, section_info)    

    ############################################################################
        #MIDI MODIFICATION 
    ############################################################################
    # MIDI CLOCK RATE: 24 PER QUARTER NOTE! ASSUMINDO (POR EX) 96 TICKS PER QUARTER NOTE (PPQ),
    # TEMOS O TAMANHO MINIMO DE UMA NOTA DE 4 TICKS!!
    
    #convert back to csv (the input midi but just in monophonic)
    csv_mono = array_to_CSV(signal_in.copy(), velocity, csv_in, msg_beg)
    
    #MIDI MODIFICATION
    mod_midi_signal, mod_velocity = mm.MIDI_MODIFY(signal_in.copy(), velocity, note_density, vel_density, section_info, window_size, ppq, user_info)
    
    #repeat modification (n times respecting the fb (feedback) parameter)
    for i in range(int(user_info.fb)):
        mod_midi_signal, mod_velocity = mm.MIDI_MODIFY(mod_midi_signal, mod_velocity, note_density, vel_density, section_info, window_size, ppq, user_info)
    
    #convert back to csv 
    csv_out = array_to_CSV(mod_midi_signal, mod_velocity, csv_in, msg_beg)
    
      
    # Parse the CSV output of the previous command back into a MIDI file
    midi_object = m_csv.csv_to_midi(csv_out)
       
    
    # Save the parsed MIDI file to disk
    filename = os.path.splitext(midi_in)[0]
    filename += '_OUT.mid'
    
    inst_ref = ['BASS_', 'HARMONY_', 'MELODY_']
    
    #FILENAME ESTÁ MAL!! ESTÁ UM CAMINHO INTEIRO!!
    filepath = os.path.dirname(user_info.main_path)+'/output/output_MIDI/'+inst_ref[instrument]+ filename
    
    with open(filepath, "wb") as output_file:
        midi_writer = m_csv.FileWriter(output_file)
        midi_writer.write(midi_object)
     
    max_sec = 5
    #ADJUST SECTION_TYPE SO IT IS ALWAYS 5
    
    section_type_out = adjust_section_sequence(copy.deepcopy(section_type), max_sec)
     
    #print('section_type:', section_type)
    #print('section_type_out:', section_type_out)
    #sys.exit()
    
    #with info about the midi signal with MOD  
    midi_info_out = info_midi.__new__(info_midi)    
    midi_info_out.__init__(max_sec, section_type_out, ppq, mod_midi_signal, mod_velocity, csv_out, section_info.beginning)
    
    
    #with info about the midi signal without MOD
    midi_info_in = info_midi.__new__(info_midi)
    midi_info_in.__init__(n_sections, section_type, ppq, mod_midi_signal, mod_velocity, csv_out, section_info.beginning)    
    
    #return midi_info_in, midi_info_in
    return midi_info_out, midi_info_in