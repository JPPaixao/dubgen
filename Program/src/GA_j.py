"""
Created on Wed May 11 16:28:46 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Genetic Algorithm that picks samples for MIDI Arrangement
"""
import numpy as np
import itertools
import random
import os
import pickle
import copy

#disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from sys import setrecursionlimit

import warnings
warnings.filterwarnings("ignore")

import SAMPLE_PROCESS_j as SP
import LSTM_j as LSTM
import PLAY_MIDI_AND_SAMPLES_j as pb

# permutations with replacement
# by Sriv https://codereview.stackexchange.com/questions/236693/permutations-with-replacement-in-python
def permutations_with_replacement(n: int, m: int, cur=None):

    setrecursionlimit(10 ** 4)    

    if cur is None:
        cur = []

    if n == 0:
        yield cur
        return

    for i in range(1, m + 1):
        yield from permutations_with_replacement(n - 1, m, cur + [i])

# get permutations with replacement
def get_permutations_with_replacement(n, class_idx):

    setrecursionlimit(10 ** 4)
    
    m = len(class_idx)
    permutations = list()
    
    for i in permutations_with_replacement(n, m):
        perm=[]
        for s in range(len(i)):
            perm.append(class_idx[i[s]-1])
            
            
        permutations.append(tuple(perm))
    
    return permutations
    
# create Population (list of arrangements)
def Create_Population(n_sections, class_idx, midi_info_list, pop_size, seed=None):
    Pop = []    
    Pop_comb = []   
    random.seed(seed)
    #maximum number of combinations for each instruments (so the program doesn't freeze)
    max_comb = 5_000_000

    for inst in range(len(class_idx)):
        unique, counts = np.unique(midi_info_list[inst].section_type, return_counts=True)
        if len(class_idx[inst])> len(unique):
            if len(unique)<4:
                comb = set(itertools.permutations(class_idx[inst], len(unique)))
            else:
                comb = set(itertools.islice(itertools.permutations(class_idx[inst], len(unique)), max_comb))
        else: comb = get_permutations_with_replacement(len(unique), class_idx[inst])
        
        if len(comb)!=len(class_idx[inst]):
            #remove tuples where all samples are the same
            comb = {t for t in comb if len(set(t)) != 1}
        if len(comb)>=pop_size:
            Pop_comb.append(random.sample(comb, pop_size))
        else:
            w=0
            pop_aux=[]*pop_size
            while w<np.floor(pop_size/len(comb)):
                pop_aux[len(comb)*w:len(comb)*(w+1)] = random.sample(comb, len(comb))
                w+=1
            pop_aux[len(comb)*w:pop_size] = random.sample(comb, pop_size-(len(comb)*w))
            Pop_comb.append(pop_aux)
            
    transposed_tuples = list(zip(*Pop_comb))
    transposed = [list(sublist) for sublist in transposed_tuples]
    
    Pop_combinations = list(np.array(transposed))
    
    for ind in Pop_combinations:
        for c in range(len(ind)):
            aux = np.ones(n_sections)*(-1)
            for j in range(len(ind[c])):
                aux[np.where(midi_info_list[c].section_type==j)] = ind[c][j].astype(int) 
            if c==0: individual=aux
            else:
                individual = np.column_stack((individual.T, aux.T)).T
        Pop.append(individual.astype('int'))
        
    for ind in Pop_combinations:
        #for every instrument
        for c in range(len(ind)):
            unique = len(ind[c])
            aux = np.ones(n_sections)*(-1)
            
            #for every section_type
            for j in range(unique):
                aux[np.where(midi_info_list[c].section_type==j)] = ind[c][j].astype(int) 
                
            if c==0: individual=aux
            else:
                individual = np.column_stack((individual.T, aux.T)).T
        Pop.append(individual.astype('int'))

    return Pop

# compute fitness of Arrangement (Individual)
def Fitness(Individual, model, features, scaler_norm):
    Scores_user=[[0]]
    Pop = [Individual]
    Pop = repeat_arrays(Pop, 5)
            
    X_ind, _ , _ , _ = LSTM.Training_data_for_GA_CNN(Pop, features, Scores_user, 1, scaler_norm) #Scores entry (y_ind) doesn´t matter
    
    X_ind = np.array(X_ind)
    score = model.predict(X_ind, verbose=0)
    return score[0][0]
    

# selects Individuals with best fitness (best half)
def Select(Pop, Scores):
    Scores_array = np.array(Scores) 
    idx = np.argsort(Scores_array.T)
    
    Pop_array = np.array(Pop)
    return list(Pop_array[idx[int(len(idx.T)/2):]])

# creates new Individuals by switching instruments between two parent Individuals
def Crossover(Pop, user_info):
    #find mating partner
    new_Pop = copy.deepcopy(Pop)
    
    count=0
    while equal_order_lists(Pop, new_Pop) and count < 50:
        random.shuffle(new_Pop) #shuffled population
        count+=1
    
    #switch combination of one instrument for each pair
    for i in range(len(Pop)):
        inst = random.randint(0, (user_info.n_inst -1)) #random instrument
        new_Pop[i][inst] = Pop[i][inst].copy()
    
    return Pop + new_Pop


#Compare two lists and check if any element in List A is equal to the element in List B with the same index.
def equal_order_lists(list_a, list_b):
    for i in range(len(list_a)):
        if np.array_equal(list_a[i], list_b[i]):
            return True
    return False


#mutates Individuals given a probability
def Mutation(Pop, class_idx, user_info, midi_info_list, prob=0.3): 
    for i in range(len(Pop)):        
        if random.uniform(0, 1)<=prob:
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument
            #gather different samples
            unique, counts = np.unique(Pop[i][inst], return_counts=True)
            
            if len(unique)>1:
                switch= random.sample(range(len(unique)),2) #pick which samples to switch
                #switching place of samples
                aux=copy.deepcopy(Pop[i][inst]) # only used to keep original order to find indexes
                Pop[i][inst][np.where(Pop[i][inst]==unique[switch[0]])[0]] = unique[switch[1]]
                Pop[i][inst][np.where(aux==unique[switch[1]])[0]] = unique[switch[0]]   
                
        if random.uniform(0, 1)<=prob:
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument
            if len(class_idx[inst]) > 1:
                unique, counts = np.unique(Pop[i][inst], return_counts=True) #gather different samples
                
                #this condition protects from this case: class idx=[s1, s2], inst=[s1, s2, s1, s2], switch s2--->ind=[s1, s1, s1, s1]
                if len(unique) < len(class_idx[inst]):
                    mutate_samp = random.sample(set(unique),1)        
                    rand_samp=mutate_samp
                    
                    while rand_samp==mutate_samp: #to avoid mutating to the same sample
                        rand_samp=random.sample(set(class_idx[inst]), 1)[0]
                        if (rand_samp in unique): #to avoid picking a sample that already is in the individual
                            rand_samp==mutate_samp #just so that the while cycle continues
                            
                    Pop[i][inst][np.where(Pop[i][inst]==mutate_samp)[0]] = rand_samp
    return Pop

#class with GA info
class GA_info:
    def __init__(self, POP_SIZE, N_GEN):
        self.POP_SIZE = POP_SIZE
        self.N_GEN = N_GEN


#creates class_idx vector (samples' instrument classes) and generates a population
def create_pop_info(midi_info_list, sample_info, n_pop, seed=None):
    df = sample_info.df_samples
    classes = df['Class'].to_numpy()

    bass_idx = np.where(classes == 0)[0]
    harm_idx = np.where(classes == 1)[0]
    melo_idx = np.where(classes == 2)[0]
    
    class_idx = [bass_idx, harm_idx, melo_idx]
    
    #just so that there isnt instruments with less sections
    n_sections = max(midi_info_list[0].n_sections,
                     midi_info_list[1].n_sections,
                     midi_info_list[2].n_sections)

    for j in range(len(midi_info_list)):  
        if midi_info_list[j].n_sections<n_sections:
            section_type = midi_info_list[j].section_type
            
            #add sections to section type (repeating the last section as to
            #minimize affecting the learning process)
            while len(section_type)<n_sections:
                section_type = np.append(section_type, section_type[-1])
            
            midi_info_list[j].section_type = section_type
            midi_info_list[j].n_sections=n_sections

    Pop = Create_Population(n_sections, class_idx, midi_info_list, n_pop, seed)
    
    return class_idx, df, Pop


#class with population info
class info_pop:
    def __init__(self, Pop_train, class_idx_train, seed, first_section):
        self.Pop_train = Pop_train
        self.class_idx_train = class_idx_train
        #self.idx_musician = idx_musician
        self.seed = seed
        self.first_section = first_section
      
        
#load pickle object
def Load_Object(path, Pickle_name="stored_object.pickle", 
                folder = '/Data/User_Data/'):
    file_to_read = open(path + folder
                        + Pickle_name, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()

    return loaded_object


# save pickle object
def Save_Object(Object, path, Pickle_name="stored_object.pickle", 
                folder= '/Data/User_Data/'):
    file_to_store = open(path+ folder
                         + Pickle_name, "wb")
    pickle.dump(Object, file_to_store)
    file_to_store.close()
    return


#find duplicates in Population
def find_duplicates(arrays_list):
    count = 0
    seen = {}
    duplicates = []
    for i, array in enumerate(arrays_list):
        array_hash = hash(array.tobytes())
        if array_hash in seen:
            count += 1
            duplicates.append((seen[array_hash], i))
        else:
            seen[array_hash] = i
    return duplicates, count


# before the GA, create individuals and sound files for user scoring
def GA_PREPOP(user_info, midi_info_list, sample_info, tkinter):
    POP_SIZE = 30
    N_GEN=30
    info_GA = GA_info(POP_SIZE, N_GEN)
    
    text = 'Starting Genetic Algorithm... ' 
    tkinter.config(text=text)
    
    class_idx, df_test, Pop_test = create_pop_info(midi_info_list, sample_info,
                                                      n_pop = POP_SIZE) 
    
    return info_GA, class_idx, Pop_test, df_test


#repeat arrays in list until we get the desired length
def repeat_arrays(arr_list, desired_length):
    max_len = max(max([arr.shape[1] for arr in arr_list]), desired_length)
    repeated_arr_list = []
    for arr in arr_list:
        repeated_arr = repeat_sections(arr, max_len)
        repeated_arr_list.append(repeated_arr)
    return repeated_arr_list


#repeat sections in list until we get the desired length
def repeat_sections(arr, max_len):
    current_len = arr.shape[1]
    if current_len < max_len:
        repeated_sections = np.repeat(arr, np.ceil(max_len / current_len), axis=1)
        start_idx = np.random.randint(0, repeated_sections.shape[1] - max_len + 1)
        repeated_sections = repeated_sections[:, start_idx:start_idx+max_len]
        return repeated_sections
    else:
        return arr


#remove duplicates in array list
def remove_duplicates(arrays):
    unique_arrays = []
    seen_arrays = []
    for i, array in enumerate(arrays):
        flattened_array = array.flatten()
        if flattened_array.tolist() not in seen_arrays:
            seen_arrays.append(flattened_array.tolist())
            unique_arrays.append(array)
    return unique_arrays


#create new individuals in case there is duplicates, so that the population size is maintained
def add_new_pop(Pop_unique, class_idx, user_info, midi_info_list, n_gen):
    Pop_all=Pop_unique.copy()
    
    while len(Pop_all) < n_gen:
        Pop_new = Mutation(Pop_all.copy(), class_idx, user_info, midi_info_list, prob=1)
        Pop_all = Pop_all + Pop_new
        Pop_all = remove_duplicates(Pop_all)
        
    return Pop_all[:n_gen]
    

#Genetic Algorithm, picks samples for a MIDI Arrangement
def GA(user_info, sample_info, class_idx, midi_info_list,
        Pop, df_test, info_GA, midi_info_real, tkinter):
    
    midi_files=list()
    for filename in os.listdir(os.path.dirname(user_info.main_path)+'/output/output_MIDI'):
        if (filename.endswith("_OUT.mid") or filename.endswith("_OUT.midi"))==True:
            if filename.startswith('BASS'): midi_files.append(filename)
            elif filename.startswith('HARMONY'): midi_files.append(filename)
            elif filename.startswith('MELODY'): midi_files.append(filename)
    
    POP_SIZE = info_GA.POP_SIZE
    N_GEN = info_GA.N_GEN
    
    new_feat=[]
    _, _, _, new_feat, _ = SP.Create_FeatureSpace( sample_info.flat_samp , window = 2048, SR = 22050)

    model_path = '/Data/Model_Data/program_model/NN_model/'
    model = load_model( user_info.main_path +model_path+"stored_cnn_model.h5")
    scaler= Load_Object(  user_info.main_path, Pickle_name="stored_cnn_model_scaler.pickle", 
                    folder=model_path)
    
    #Genetic algorithm
    gen=0
    #record best Individual
    goat_score = 0
    #early stop init
    es_counter=0
    early_stop=False

    while gen< N_GEN and early_stop==False:
        Pop_unique = remove_duplicates(Pop)
        
        if len(Pop_unique)!=len(Pop):
            Pop = add_new_pop(Pop, class_idx, user_info, midi_info_list, POP_SIZE)   
        
        Scores = []
        for Ind in Pop:
            Scores.append(Fitness(Ind, model, new_feat, scaler))
        
        #early stop check
        if max(Scores)>goat_score:
            goat = Pop[np.argmax(Scores)]
            goat_score = max(Scores)
            es_counter =-1
        elif es_counter>4:
            early_stop = True 
        
        best_score = round(max(Scores),3)

        text = 'Generation ' + str(gen+1) + '/' + str(N_GEN)+ ' (Best Score: ' + str(best_score)+ '/' + str(1) + ')'
        tkinter.config(text=text)
        
        new_Pop = Select(Pop, Scores)
        new_Pop = Crossover(new_Pop, user_info)
        Pop = Mutation(new_Pop, class_idx, user_info, midi_info_list, 0.3)
        
        gen+=1
        es_counter +=1
    
    best_Ind = goat
    
    text = 'Generating Best Arrangement... '
    tkinter.config(text=text)
    
    music_path = pb.PLAYBACK_midi_samples(best_Ind, user_info, midi_info_real,
                                          sample_info, midi_files, sr = 22050, BPM = user_info.bpm,
                                          ex = -1, id_name='_with_mod_no_synth', tkinter=tkinter)

    return music_path, max(Scores), best_Ind