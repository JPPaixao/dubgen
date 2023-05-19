"""
Created on Mon Jun 27 13:37:12 2022

@author: JP

@brief: Applies audio effects to input signal (Chorus, Delay, Reverb, etc.)
"""
from pedalboard import Pedalboard, Chorus, Reverb, Gain, Limiter, Delay
from pedalboard.io import AudioFile

# applies audio effects to input signal
def Pedalboard_func(sound_files, sr):
    sr=float(sr)
    audio = []
    effected_audio = []
    
    for filename in sound_files:
        # Read in a whole audio file:
        with AudioFile(filename,'r') as f:
          audio.append(f.read(f.frames))
          sr = f.samplerate
    
    # Create Pedalboard objects, containing multiple plugins:
    board_bass =  Pedalboard([Chorus(), Gain(gain_db=2), Limiter()])
    board_harm =  Pedalboard([Chorus(), Limiter(), Delay(delay_seconds=0.125, mix=0.1), Reverb(room_size=0.4)])
    board_melody = Pedalboard([Chorus(), Limiter(), Reverb(room_size=0.4)])

    bass_sound = board_bass(audio[0], sr)
    harmony_sound = board_harm(audio[1], sr)
    melody_sound = board_melody(audio[2], sr)
    
    # Runs the audio through this pedalboard
    effected_audio.append(bass_sound)
    effected_audio.append(harmony_sound)
    effected_audio.append(melody_sound)
    
    return effected_audio