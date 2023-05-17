"""
Created on Wed Aug  3 13:36:40 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Main script with fucntions (modules) for GUI usage
"""

import os
import sys
import pickle
import time

import warnings
warnings.filterwarnings("ignore")

class info:
    def __init__(self, n_inst, bpm, section_size, sound_folder, midi_folder,
                 quantization_step, quantization_amount, main_path, test_path, mod, fb):
        self.n_inst = n_inst
        self.bpm = bpm
        self.section_size = section_size
        self.sound_folder = sound_folder
        self.midi_folder = midi_folder
        self.quantization_step = quantization_step
        self.quantization_amount = quantization_amount
        self.main_path = main_path
        self.test_path = test_path
        self.mod =mod
        self.fb = fb
        
def Load_Object(path, Pickle_name="stored_object.pickle", 
                folder = '/Data/User_Data/'):
    
    file_to_read = open(path + folder
                        + Pickle_name, "rb")
    loaded_object = pickle.load(file_to_read)
    
    file_to_read.close()

    return loaded_object

def Save_Object(Object, path, Pickle_name="stored_object.pickle", 
                folder='/Data/User_Data/'):
    
    file_to_store = open(path+ folder
                         + Pickle_name, "wb")
    pickle.dump(Object, file_to_store)
    
    file_to_store.close()

    return

def Create_user(user_data):
    main_path=user_data.main_path
    bpm=user_data.bpm
    
    mod=user_data.mod
    fb=user_data.fb

    sound_folder = os.path.dirname(main_path)+'/user/train_sounds'
    midi_folder = os.path.dirname(main_path)+'/user/midi'

    test_path = os.path.dirname(main_path)+'/user/test_sounds'
    
    section_size = user_data.section_size
    #section_size = 8

    user_info = info(3, bpm, section_size, sound_folder, midi_folder, 0.5, 0.7, main_path, test_path, mod, fb) #8 no tamanho da section???

    return user_info

#remove existing files of output directories (where the program will write)
def Remove_Files(user_info):
    end_path = [ '/output/output_samples',
                 '/output/output_music/NO_SYNTH/',
                '/output/output_music/NO_SYNTH/best_samples',
                '/output/output_music/NO_SYNTH/best_sound_mp3',
                 '/output/output_music/SYNTH/',
                '/output/output_music/SYNTH/best_samples_synth',
                '/output/output_music/SYNTH/best_sound_mp3_synth',
                 '/output/output_MIDI','/output/output_MIDI']
    extension = [ ".wav", ".wav", '.wav','mp3', '.wav','wav',
                 'mp3', "_OUT.mid", "_OUT.midi"]
    
    path=user_info.main_path
    
    for path_end, ext in zip(end_path, extension):
        for filename in os.listdir(os.path.dirname(path) + path_end ):
            if filename.endswith(ext) ==True:
                os.remove(os.path.dirname(path)+path_end+'/'+filename)
            
def adjust_sequences(sequences, midi_info_list):
    
    desired_length=0
    
    for info_midi in midi_info_list:
        desired_length = max(desired_length, info_midi.n_sections)
    
    adjusted_sequences = []

    for sequence in sequences:
        if len(sequence) >= desired_length:
            adjusted_sequence = sequence[:desired_length]
        else:
            repetitions = desired_length // len(sequence)
            remainder = desired_length % len(sequence)
            adjusted_sequence = sequence * repetitions + sequence[:remainder]
        adjusted_sequences.append(adjusted_sequence)

    return adjusted_sequences

def Sampling_and_Mod(user_data, event, train_classification, tkinter):
    
    sys.path.append(user_data.main_path+'/src')
    
    import SAMPLE_PROCESS_j as SP
    import MIDI_PROCESS_j as MP
    
    user_info = Create_user(user_data) 
    
    #import time
    #time.sleep(5)
    
    if event.is_set():
        print('Thread Stopped!')
        return
    
    sys.path.append(user_info.sound_folder)
    sys.path.append(user_info.test_path)
    sys.path.append(user_info.midi_folder)
    
    #remove existing files of output directories (where the program will write)
    Remove_Files(user_info)
    
    # remove miscellaneous sounds or not
    leave_misc = 0
    
    #TRAINING
    #Get dataframe with classified samples and respective features
    if train_classification==1:
        if event.is_set():
            print('Thread Stopped!')
            return
        
        #print('\nTraining the Classification Model:')
        sample_train_info, model_train_info, pca_info_misc, pca_info_inst, use_dubgen_model = SP.SAMPLE_PROCESSING(
                                                                            user_info,
                                                                            train = train_classification,
                                                                            leave_misc= leave_misc
                                                                            )
        
        if event.is_set():
            print('Thread Stopped!')
            return
        
        if use_dubgen_model:
            model_train_info = Load_Object(user_info.main_path, "stored_svm_model.pickle",
                                           '/Data/Model_Data/program_model/SVM_model/')
            
            pca_info_misc = Load_Object(user_info.main_path, "stored_pca_info_misc.pickle",
                                           '/Data/Model_Data/program_model/PCA_model/')
            pca_info_inst = Load_Object(user_info.main_path, "stored_pca_info_inst.pickle",
                                           '/Data/Model_Data/program_model/PCA_model/')
        else:
            #save sample_info object and the SVM model
            Save_Object(sample_train_info, user_info.main_path, 
                        Pickle_name= "stored_sample_info.pickle", 
                            folder='/Data/Model_Data/user_model/SVM_model/')
            Save_Object(model_train_info, user_info.main_path, 
                        Pickle_name= "stored_svm_model.pickle", 
                            folder='/Data/Model_Data/user_model/SVM_model/')
            
            #save pca model for misc and inst classification
            Save_Object(pca_info_misc, user_info.main_path, 
                        Pickle_name= "stored_pca_info_misc.pickle", 
                            folder='/Data/Model_Data/user_model/PCA_model/')
            Save_Object(pca_info_inst, user_info.main_path, 
                        Pickle_name= "stored_pca_info_inst.pickle", 
                            folder='/Data/Model_Data/user_model/PCA_model/')
    else:
        #load saved objects
    
        model_train_info = Load_Object(user_info.main_path, "stored_svm_model.pickle",
                                       '/Data/Model_Data/program_model/SVM_model/')
        
        pca_info_misc = Load_Object(user_info.main_path, "stored_pca_info_misc.pickle",
                                       '/Data/Model_Data/program_model/PCA_model/')
        pca_info_inst = Load_Object(user_info.main_path, "stored_pca_info_inst.pickle",
                                       '/Data/Model_Data/program_model/PCA_model/')
    

    if event.is_set():
        print('Thread Stopped!')
        return

    tkinter.config(text='oi...') #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    #print('\nClassifying Samples:')
    sample_info, model_info, pca_info_misc, pca_info_inst,_ = SP.SAMPLE_PROCESSING(user_info, train = 0,
                                                                    leave_misc = leave_misc,
                                                                    model_info = model_train_info,
                                                                    pca_info_misc = pca_info_misc ,
                                                                    pca_info_inst = pca_info_inst)
    
    if event.is_set():
        print('Thread Stopped!')
        return
    
    midi_info_out = list()
    midi_info_in = list()
    
    import SECTION_INFO as SI
    
    section_info = SI.prepare_section_analysis(user_info)
    Save_Object(section_info, user_info.main_path, 'stored_section_info.pickle' )
    
    #print('\n')
    count_midi=0
    instrument=0
    
    FOLDERS = ['/INSERT_BASS_MIDI_HERE/', '/INSERT_HARMONY_MIDI_HERE/', '/INSERT_MELODY_MIDI_HERE/' ]
    
    for folder in FOLDERS:
        path = user_info.midi_folder+folder
        file_count=0
        
        for filename in os.listdir(path):
            if file_count==0:
                if filename.endswith(".mid") or filename.endswith(".midi"):
                        count_midi+=1
                        if event.is_set():
                            print('Thread Stopped!')
                            break
                        
                        print("\r\033[2KModifying MIDI File {}/{}".format(count_midi, 3),
                              end="", flush=True)
                        aux_out, aux_in = MP.PROCESS_MIDI(user_info, filename, section_info,
                                                          instrument, folder)
                        midi_info_out.append(aux_out)
                        midi_info_in.append(aux_in)
            file_count+=1
        instrument+=1
    
    Save_Object(midi_info_out,  user_info.main_path, Pickle_name="stored_midi_info_out.pickle")
    
                    
    if event.is_set():
        print('Thread Stopped!')
        return
    
    if count_midi<user_info.n_inst:
        print('\nOne or more MIDI files are missing\n')
        exit()
        
    if train_classification!=1:
        sample_train_info = sample_info
    
    #save info objects in pickle file
    Save_Object(user_info, user_info.main_path, 'stored_user_info.pickle' )
    Save_Object(sample_train_info, user_info.main_path, 'stored_sample_info.pickle' )
    Save_Object(section_info, user_info.main_path, 'stored_section_info.pickle' )

    
    return user_info, midi_info_out, sample_info, sample_train_info, section_info, midi_info_in

def main_GA_PREPOP(user_info, midi_info_in, sample_info):
    import GA_j as gen
    #FLAT SAMP VAI ENTRAR NO PLAYBACK!! LOGO TEM DE ENTRAR NO GA
    info_GA, class_idx, Pop, df = gen.GA_PREPOP(user_info, midi_info_in, sample_info)
    
    return info_GA, class_idx, Pop, df

def main_GA(user_info, sample_info, class_idx, midi_info_out, Pop, df, info_GA, midi_info_real):
    import GA_j as gen
    music_path, best_score, sample_Ind = gen.GA(user_info, sample_info, class_idx, midi_info_out,
                                     Pop, df, info_GA, midi_info_real)
    
    return music_path, best_score, sample_Ind

def main_synth_GA(user_info, midi_info_out, info_GA, sample_info, Ind_sample, Ind_synth, synths, midi_info_real):

    import GA_SYNTH as ga_sy
    import PLAY_MIDI_AND_SAMPLES_j as pb
    
    if Ind_synth == list():
        Ind_synth, Parameters_type, synths = ga_sy.GA(user_info, midi_info_out, info_GA,
                                                      sample_info, Ind_sample)
    else:
        #PARAMETERS==PARAMETERS_TYPE
        AutoFilter = {'cutoff_floor':float(), 'cutoff_ceiling':float(), 'lfo_floor':float(),
                      'lfo_ceiling': float(), 'lfo_shape': str(), 'lfo_evo': str(), 'high_pass': bool() }
        Granular = {'grain_size': float(), 'grain_space': float(),'smoothness': bool(),
                    'order': bool(), 'sync': bool()}
        Interpolator = {'shape': str(), 'evo': str()}
        Parameters_type = [AutoFilter, Granular, Interpolator]
        
    
    midi_files=list() #these midi files are output (WITH MOD)
    for filename in os.listdir(os.path.dirname(user_info.main_path)+'/output/output_MIDI'):
        if (filename.endswith("_OUT.mid") or filename.endswith("_OUT.midi"))==True:
            if filename.startswith('BASS'): midi_files.append(filename)
            elif filename.startswith('HARMONY'): midi_files.append(filename)
            elif filename.startswith('MELODY'): midi_files.append(filename)
            
    Ind_synth = adjust_sequences(Ind_synth, midi_info_real)
    
    music_path = pb.PLAYBACK_midi_samples(Ind_sample, user_info, midi_info_real, 
                              sample_info, midi_files, sr=22050, BPM=user_info.bpm, ex=-1,
                              id_name= '_mod_and_synth_mus' , first_sec = 0,
                              synth = 1, synths_param=Ind_synth, synths=synths,
                              Parameters_types = Parameters_type)
    
    return music_path


def GUI_TEST():
    done=0

    for i in range(10):
        print(i,' second(s)')
        time.sleep(1)
    done=1
    return done

