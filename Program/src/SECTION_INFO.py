"""
Created on Mon Jun 13 21:58:27 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Extract Info from MIDI: beggining and ending of the music (in ticks)
"""
import os
import sys
import numpy as np
import py_midicsv as m_csv

from MIDI_PROCESS_j import first_substring


# class with information about MIDI
class info_section:
    def __init__(self, beginning, end):
        self.beginning = beginning
        self.end = end
        
# extract marker for beggining and ending of the music (in ticks)
def prepare_section_analysis(user_info):
    note_on = 'Note_on_c'
    
    os.chdir(user_info.midi_folder)
      
    music_start = float('inf')
    music_end = 0
    ppq=0
    count_midi=0
    
    FOLDERS = ['/INSERT_BASS_MIDI_HERE/', '/INSERT_HARMONY_MIDI_HERE/', '/INSERT_MELODY_MIDI_HERE/' ]
    
    for folder in FOLDERS:
        path = user_info.midi_folder+folder
        file_count=0 
        for filename in os.listdir(path):
            if filename.endswith(".mid") or filename.endswith(".midi"):
                count_midi+=1
                csv_strings = m_csv.midi_to_csv(path+filename)
                
                header = csv_strings[0].split(',')
                if (int(header[-1])!=ppq) and (music_end != 0):
                    print('\nMIDI ERROR, FILES HAVE DIFFERENT PPQ')
                                               
                ppq = int(header[-1]) #ticks per quarter note
                msg_beg = first_substring(csv_strings, note_on)
                msg_end= first_substring(csv_strings, 'End_track')-1
                msg_list = csv_strings[msg_beg].split(', ') 
                
                if int(msg_list[1])< music_start: music_start = int(msg_list[1])
                
                msg_list = csv_strings[msg_end].split(', ')    
                
                if int(msg_list[1])> music_end: music_end = int(msg_list[1])
                    
            file_count+=1
            
    if count_midi<user_info.n_inst:
        print('\nOne or more MIDI files are missing\n')
        sys.exit()
                
    quantization= user_info.quantization_step*ppq
    
    music_start = int(np.round(music_start/quantization,decimals=0)*quantization)
    music_end = int(np.round(music_end/quantization,decimals=0)*quantization)
    
    section_info = info_section(int(music_start), music_end)
    return section_info
    
