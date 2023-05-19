"""
Created on Tue Oct 25 11:54:12 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Genetic Algorithm that picks synth parameters for MIDI Arrangement
"""
import numpy as np
import itertools
import random
import os
import copy
from sklearn.metrics import mean_squared_error
from sys import setrecursionlimit

import SYNTH as sy
import SAMPLE_PROCESS_j as SP
import PLAY_MIDI_AND_SAMPLES_j as pb

import warnings
warnings.filterwarnings("ignore")

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


# get parameters' possible string values
def get_var_strings():
    lfo_shapes = ['sine', 'square', 'triangle', 'sawl', 'sawr']
    ip_shapes = ['sine', 'square', 'triangle', 'sawl', 'sawr']
    evos = ['constant', 'linear_up', 'linear_down', 'exp_up', 'exp_down', 'random']
    
    #autofilter
    af_string_var = {'lfo_shape': lfo_shapes, 'lfo_evo': evos}
    #interpolator
    ip_string_var = {'shape': ip_shapes, 'evo': evos}
    
    string_vars = [af_string_var, dict() ,  ip_string_var]  
    return string_vars


#just check if params are correct (specificaly in the freq range param)
def correct_param(inst_param, synth):
    if synth == 0: # Autofilter
        if inst_param[0]>inst_param[1]:
            aux = inst_param[0]
            inst_param[0] = inst_param[1]
            inst_param[1] = aux
        elif inst_param[0]==inst_param[1]:
            inst_param[1] += 0.1
            
        if inst_param[2]>inst_param[3]:
            aux = inst_param[2]
            inst_param[2] = inst_param[3]
            inst_param[3] = aux
        elif inst_param[2]==inst_param[3]:
            inst_param[3] += 0.1
        
    return inst_param


#orders the unique sections regarding section_type order
def order_params_in_section(unique_sec_params, section_type): 
    inst_param = list()
    for sec in range(len(section_type)):
        inst_param.append(unique_sec_params[section_type[sec]])
    return inst_param

#Compare two lists and check if any element in List A is equal to the element in List B with the same index.
def equal_order_lists(list_a, list_b):
    for i in range(len(list_a)):
        if list_a[i] == list_b[i]:
            return True
    return False

# create Population (list of Individuals). Each individual is a sequence of synths' parameters
def Create_Population(n_inst, synths_list, parameters, midi_info_list, pop_size, seed=None):
    Pop = []
    
    # string_vars: list with dictionary.
    # each dictionary key has a list with possible strings
    # for a specific parameter.
    # ex:
    # for wavetable-> wt_strs = string_vars[2] ---> possible string results for lfo_shape->
    # -> wt_strs['lfo_shape'] = ['sine', 'square', 'triangle', 'sawl', 'sawr']
    string_vars = get_var_strings()
    
    for i in range(pop_size):
        Ind  = list()
        for inst in range(n_inst):
            section_type = midi_info_list[inst].section_type

            unique, unique_counts = np.unique(section_type, return_counts=True)
            count_sec = len(unique) # number of different sections
            
            synth = synths_list[inst] #which synth is currently used
            param_type = parameters[synth] #dictionary with type of variables
            
            unique_sec_params=list() #list of parameters for every unique section
            for sec in range(count_sec): #for every unique section
                sec_param = list()
                for key, value in param_type.items():
                    if type(value) == type(float()):
                        sec_param.append(round(random.uniform(0, 1), 2))
                    elif type(value) == type(bool()):
                        sec_param.append(bool(random.getrandbits(1)))
                    else:
                        # dictionary with possible string results for that specific synth
                        str_params = string_vars[synth]
                        #key is the name of the variable
                        # possible params has a list of possible results for that parameter
                        possible_params = str_params[key]
                        sec_param += list(random.sample(possible_params, 1))
            
                # just check if everything is in order
                sec_param = correct_param(sec_param, synth)
                
                unique_sec_params.append(sec_param)
                
            inst_param = order_params_in_section(unique_sec_params, section_type)
            Ind.append(inst_param)
            
        Pop.append(Ind)

    return Pop


#get upper triangle matrix without the diagonal
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]


# Calculate mse similarity between every pair of vectors in matrix
def get_similarity_mse_1D(matrix):
    combinations = list(itertools.combinations(range(len(matrix)), 2))
    
    mse_sum=0
    for tuple_inst in combinations:
        mse_sum += mean_squared_error(matrix[tuple_inst[0]], matrix[tuple_inst[1]])
    similarity_value = mse_sum/len(matrix)
    
    return similarity_value


#normalize features
def normalize(feat):
    norm_feat=copy.deepcopy(feat)
    
    #mfccs (a bias is added so that the values are always from 0 to 1 (taking care of the negative mfccs))
    norm_feat[:24]=list((np.array(norm_feat[:24])+30)/60)
    #frequency range based features (floor)
    norm_feat[28:30]=list(np.array(norm_feat[28:30])/10_000)
    #frequency range based features
    norm_feat[24:28]=list(np.array(norm_feat[24:28])/20_000)
    
    #compress features that surpass the range [0, 0.95]
    norm_feat = list(compressor(norm_feat, 0.95))
    
    return norm_feat


#compress values above threshold
def compressor(x, thr):
    """
    Compresses the list values in x such that any value above the given threshold
    is scaled to be within the range [threshold, 1] using a x/(x+1) function
    
    It also compress values below a "floor threshold" of 1- threshold, so it maps [-inf, thr] to [0, thr]
    """
    y=np.zeros_like(x)
    bias = thr*(1- (1/thr))
    for i, s in enumerate(x):
        if s > thr:
            #apply compression function
            new_s = (s/(s+1)) + bias
            new_s *= (1-thr)
            
            y[i] = thr + new_s
            
        # values below floor threshold  
        elif s< 1-thr:
            #distance to floor thr
            dist_thr = abs((1-thr-s))
            #convert to positive value so we can apply the same formula as for the positive case 
            new_s = thr+dist_thr
            new_s = (new_s/(new_s+1)) + bias
            new_s *= (1-thr)
            
            #convert back to floor threshold range
            y[i] = 1 - thr - new_s
        else: y[i] = s
    return y


#normalize the score to range from 0 to 1
def norm_score(score, objectives):
    #get the vector with minimum scores possible ("negative" objectives would score 1 (-1 at the end) and positive objectives would score 0)
    min_vec = [0 if x == 1 else 1 for x in objectives]
    #worst score possible
    b=sum(min_vec)
    #adding that bias to the score so that i couldn t be less than zero
    score +=b
    #adjust to the number of sections (they were all summed up earlier in the for loop of the scoring function)
    score/=len(objectives)
    
    return score


# class with GA scoring info
class Score_class:
    def __init__(self, synths, n_sections, user_info, midi_info_list, flat_samp, parameters_range,
                 parameters_types, window_size, beg_silence, Ind_samples):
        
        #features function for features that will be compared in the fitness function
        self.get_feat = SP.Create_FeatureSpace
        self.synths=synths
        
        #define objective for each section (if it is to diverge or converge in similarity)
        self.Objectives=list()
        
        self.midi_info_list = midi_info_list
        self.section_type_bass = np.array(midi_info_list[0].section_type)
        
        unique, counts = np.unique(self.section_type_bass, return_counts=True)
        obj_arr=np.ones(len(self.section_type_bass))*-1
        for sec_type in unique:
            obj = random.choice([-1, 1])
            sec_idxs = np.where(self.section_type_bass==sec_type)[0]
            obj_arr[sec_idxs] = obj
            
        self.Objectives = list(obj_arr)
        self.user_info = user_info
        self.midi_info_list = midi_info_list
        self.parameters_range = parameters_range
        self.parameters_types = parameters_types
        self.flat_samp = flat_samp
        self.window_size = window_size
        self.beg_silence = beg_silence
        self.Ind_samples = Ind_samples
        

    # Scoring Function  
    def Fitness(self, Ind_synth, Ind_samples):
        score=0        
        Ind_feat = self.get_Ind_feat(self.get_feat, Ind_samples, Ind_synth)
        
        for section, objective in zip(np.transpose(Ind_feat, (1, 0, 2)), self.Objectives):
            score += objective*get_similarity_mse_1D(section)
            
        if score > 2:
            ind_erro=np.transpose(Ind_feat, (1, 0, 2))
            filename = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/IND_LIXADO.txt'
            with open(filename, "w") as f:
                for section in ind_erro:
                    for inst in section:
                        arr_str = ' '.join(map(str, inst))
                        f.write(arr_str+"\n")
        score = norm_score(score, self.Objectives)

        return score
    
    
    # get features for input Individual
    def get_Ind_feat(self, get_feat, Ind_samples, Ind_synth):
        Ind_feat=list()
        pitch_inst=[65.4, 261.6, 523.25]
        for inst_idx, inst, inst_param, pitch in zip(range(len(Ind_samples)), Ind_samples, Ind_synth, pitch_inst):
            section_feat=list()            
            section_beg = self.beg_silence
            
            inst_synth = sy.SYNTH(self.synths[inst_idx], 22050, self.user_info, self.parameters_types[self.synths[inst_idx]],
                               self.parameters_range, inst_param, 
                               self.window_size,
                               self.midi_info_list[inst_idx], self.beg_silence, inst_idx)     
            
            for section_sample, n_section in zip(inst, range(len(inst))):
                #function that applies the synthesis
                Synthesize  = inst_synth.synthetize
                
                sample = self.flat_samp[section_sample]
                
                automation_beginning = int(section_beg)
                automation_ending = automation_beginning + len(sample)
                
                #synthesize new sample
                new_sample = Synthesize(copy.deepcopy(sample), automation_beginning,
                               automation_ending, n_section, pitch)
                
                _, _, _, mfcc_feat, new_feat  = get_feat([new_sample])
                
                features = normalize(new_feat[0])
                section_feat.append(features)
                
                section_beg += self.window_size
                
            Ind_feat.append(np.array(section_feat))
        
        return np.array(Ind_feat)


# select best Individuals
def Select(Pop, Scores):
    Scores_array = np.array(Scores)
    idx = np.argsort(Scores_array.T)
    chosen_Pop=list()
    for chosen_ind in idx[int(len(idx)/2):]:
        chosen_Pop.append(copy.deepcopy(Pop[chosen_ind]))
    return chosen_Pop


# creates new Individuals by switching instruments between two parent Individuals
def Crossover(Pop, user_info):
    #find mating partner
    new_Pop = copy.deepcopy(Pop)
    count=0
    while equal_order_lists(Pop, new_Pop) and count < 50:
        #print("shuffle Pop in crossover")
        random.shuffle(new_Pop) #shuffled population
        count+=1
       
    #switch combination of one instrument for each pair
    for i in range(len(Pop)):
        inst = random.randint(0, (user_info.n_inst -1)) #random instrument (3)
        new_Pop[i][inst] = copy.deepcopy(Pop[i][inst])
        
    Pop_all = copy.deepcopy(Pop + new_Pop)
    
    return Pop_all


#mutates Individuals given a probability
def Mutation(Pop, user_info, midi_info_list, synths, Parameters, prob=0.3):
    #ex: ind[inst]=[a, b, a, b, c]--->ind[inst]=[b, a, b, a, c]
    for i in range(len(Pop)):        
        if random.uniform(0, 1)<=prob: #switch parameters between two sections
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument 
            section_type = midi_info_list[inst].section_type
            
            unique, counts = np.unique(section_type, return_counts=True) #gather different samples
            
            if len(unique)>1:
                section1 = random.choice(unique) #random section
                other_sections = list(unique)
                other_sections.remove(section1)
                section2 = random.choice(other_sections)
                
                idx_section1 = np.where(section_type==section1)[0] #where one of the section occurs
                idx_section2 = np.where(section_type==section2)[0]
                
                aux1 = Pop[i][inst][idx_section1[0]] #save parameters for section 1
                aux2 = Pop[i][inst][idx_section2[0]] #save parameters for section 2
                
                #switch parameters between sections
                for idx1 in idx_section1:
                    Pop[i][inst][idx1] = copy.deepcopy(aux2)
                for idx2 in idx_section2:
                    Pop[i][inst][idx2] = copy.deepcopy(aux1)
        
        if random.uniform(0, 1)<=prob: #generate new parameters for that section
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument
            section_type = midi_info_list[inst].section_type
            
            unique, counts = np.unique(section_type, return_counts=True) #gather different samples
            section1 = random.choice(unique) #random section
            idx_section1 = np.where(section_type==section1)[0] #where one of the section occurs
            current_parameters = Pop[i][inst][idx_section1[0]]
            
            new_parameters = Generate_New_Parameters(inst, synths, Parameters, current_parameters)
            
            for idx1 in idx_section1:
                Pop[i][inst][idx1] = new_parameters

    return Pop


# generate new parameters for an Individual's synth
def Generate_New_Parameters(inst, synths_list, parameters_type, current_parameters):
    synth = synths_list[inst] #which synth is currently used
    param_type = parameters_type[synth] #dictionary with type of variables
    sec_param = copy.deepcopy(current_parameters)
    string_vars = get_var_strings()
    
    while current_parameters == sec_param: #to guarantee it doesn generate equal parameters
        sec_param=[]
        for key, value in param_type.items():
            if type(value) == type(float()):
                sec_param.append(round(random.uniform(0, 1), 2))
            elif type(value) == type(bool()):
                sec_param.append(bool(random.getrandbits(1)))
            else:
                # dictionary with possible string results for that specific synth
                str_params = string_vars[synth]
                # key is the name of the variable
                # possible params has a list of possible results for that parameter
                possible_params = str_params[key]
                sec_param += list(random.sample(possible_params, 1))
               
        # just check if everything is in order
        sec_param = correct_param(sec_param, synth)
    
    return sec_param


#class with synth GA info
class GA_info:
    def __init__(self, POP_SIZE, N_GEN, n_ex):
        self.POP_SIZE = POP_SIZE
        self.N_GEN = N_GEN


#Attribute Synths to instruments randomly
def Choose_Synths(n_inst):
    # Synth order:
    # Auto Filter: 0
    # Granular: 1
    # Interpolator: 2
    Synth_List = random.sample(list(range(n_inst)), n_inst)
    return Synth_List


#remove duplicates in Population
def remove_duplicates(inds):
    unique_arrays = []
    seen_arrays = []
    for ind in inds:
        array=np.array(ind)
        flattened_array = array.flatten()
        if flattened_array.tolist() not in seen_arrays:
            seen_arrays.append(flattened_array.tolist())
            unique_arrays.append(array.tolist())
    return unique_arrays


# get score of already scored Individuals
def get_Visited_Score(Visited_Inds, Visited_Scores, Ind):
    flattened_ind = np.array(Ind).flatten().tolist()
    
    for indv, scorev in zip(Visited_Inds, Visited_Scores):
        array=np.array(indv)
        flattened_array = array.flatten()
        
        if flattened_array.tolist()==flattened_ind:
            return scorev
    return None


# create new individuals in case there is duplicates, so that the population size is maintained
def add_new_pop(Pop_unique, user_info, midi_info_list, synths, Parameters_type, n_pop):
    Pop_all=copy.deepcopy(Pop_unique)
    
    while len(Pop_all) < n_pop:
        Pop_new = Mutation(copy.deepcopy(Pop_all), user_info, midi_info_list, synths, Parameters_type, prob=1)
        Pop_all = copy.deepcopy(Pop_all + Pop_new)
        Pop_all = remove_duplicates(Pop_all)
        
    return Pop_all[:n_pop]


# SYNTH GA: picks synth parameters for MIDI arrangement
def GA(user_info, midi_info_list, info_GA, sample_info, Ind_samples, tkinter):
    
    midi_files=list()
    for filename in os.listdir(os.path.dirname(user_info.main_path)+'/output/output_MIDI'):
        if (filename.endswith("_OUT.mid") or filename.endswith("_OUT.midi"))==True:
            if filename.startswith('BASS'): midi_files.append(filename)
            elif filename.startswith('HARMONY'): midi_files.append(filename)
            elif filename.startswith('MELODY'): midi_files.append(filename)
    
    POP_SIZE = 6
    N_GEN = 2

    #SET PARAMETERS VARIABLE TYPE
    AutoFilter = {'cutoff_floor':float(), 'cutoff_ceiling':float(), 'lfo_floor':float(),
                  'lfo_ceiling': float(), 'lfo_shape': str(), 'lfo_evo': str(), 'high_pass': bool() }
    Granular = {'grain_size': float(), 'grain_space': float(),'smoothness': bool(),
                'order': bool(), 'sync': bool()}
    Interpolator = {'shape': str(), 'evo': str()}
    Parameters_type = [AutoFilter, Granular, Interpolator]
    
    synths=[0,1,2]
    random.shuffle(synths)
    
    text = 'Starting Genetic Algorithm...'
    tkinter.config(text= text)
    
    Pop = Create_Population(user_info.n_inst, synths, Parameters_type, midi_info_list, pop_size = POP_SIZE)    
    
    ppq = midi_info_list[0].ppq
    delta_time = 60/(user_info.bpm*ppq) #time of each tick
    sr=22050
    samples_per_tick = delta_time*sr
    
    #compute_window_size
    window_size = user_info.section_size*4*ppq
    window_size *= samples_per_tick
    
    #compute_beg_silence
    beg_silence = midi_info_list[0].beg_compass
    beg_silence *= samples_per_tick
    
    #init scoring functions. Order by instruments
    scoring = Score_class(synths, len(Pop[0]), user_info, midi_info_list, sample_info.flat_samp, 
                          pb.get_range_parameters(), Parameters_type, window_size, beg_silence, Ind_samples)
    
    gens_all=list()
    gen_idx_all=list()
    #record best Individual
    goat_score = 0
    #early stop init
    es_counter=0
    early_stop=False
    
    Visited_Inds = list()
    Visited_Scores = list()
    
    #Genetic algorithm
    gen=0
    while gen< N_GEN and early_stop==False:
        Pop_unique = remove_duplicates(Pop)
        if len(Pop_unique)<len(Pop):
            print(len(Pop)-len(Pop_unique), 'duplicates!')
            Pop = add_new_pop(Pop_unique, user_info, midi_info_list, synths, Parameters_type, POP_SIZE)   
        
        Scores = []
        n_ind=1
        for Ind in Pop:
            score = get_Visited_Score(Visited_Inds, Visited_Scores, Ind)
            
            text = 'Generation ' + str(gen+1) + '/' + str(N_GEN) + ' (Scoring Process: ' + str(n_ind)+ '/' + str(len(Pop)) + ' )'
            tkinter.config(text=text)
            n_ind+=1
            
            if score == None:
                score = scoring.Fitness(Ind, Ind_samples)         
                Visited_Inds.append(Ind)
                Visited_Scores.append(score)
                
            Scores.append(score)
        
        #early stop check
        if max(Scores)>goat_score:
            goat = Pop[np.argmax(Scores)]
            goat_score = max(Scores)
            es_counter =-1
        elif es_counter>6:
            early_stop = True
            
        gens_all.append(np.array(Scores))
        gen_idx_all.append(gen+1)  
        ########################################################################
        new_Pop = Select(Pop, Scores)
        new_Pop = Crossover(new_Pop, user_info)
        Pop = Mutation(new_Pop, user_info, midi_info_list, synths, Parameters_type, prob=0.2)
        
        gen+=1
        es_counter +=1
    
    best_Ind = goat
    
    return best_Ind, Parameters_type, synths

