# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:28:46 2022

@author: JP
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
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


import warnings
warnings.filterwarnings("ignore")


import SAMPLE_PROCESS_j as SP
import LSTM_j as LSTM
import PLAY_MIDI_AND_SAMPLES_j as pb

def permutations_with_replacement(n: int, m: int, cur=None): # by Sriv https://codereview.stackexchange.com/questions/236693/permutations-with-replacement-in-python

    setrecursionlimit(10 ** 4)    

    if cur is None:
        cur = []

    if n == 0:
        yield cur
        return

    for i in range(1, m + 1):
        yield from permutations_with_replacement(n - 1, m, cur + [i])

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
    

def Create_Population(n_sections, class_idx, midi_info_list, pop_size, seed=None): #TEM DE RESPEITAR REGRAS DAS SECTIONS!
    Pop = []    
    Pop_comb = []   
    
    random.seed(seed) #ATENÇÃO!!! RETIRAR NO FUTURO!!
    
    #maximum number of combinations for each instruments
    max_comb = 5_000_000

    
    for inst in range(len(class_idx)):
        unique, counts = np.unique(midi_info_list[inst].section_type, return_counts=True)
        
        if len(class_idx[inst])> len(unique):
            '''
            if len(class_idx[inst])>30:
                rand_idx = random.sample(set(class_idx[inst]), 30) # so that we dont get a huge number of permutations (m!/(m-n)!)
                comb = set(itertools.permutations(rand_idx, len(unique)))
            else:
                 comb = set(itertools.permutations(class_idx[inst], len(unique)))
                '''
            if len(unique)<4:
                comb = set(itertools.permutations(class_idx[inst], len(unique)))
            else:
                comb = set(itertools.islice(itertools.permutations(class_idx[inst], len(unique)), max_comb))
        #else: comb = list(itertools.combinations_with_replacement(class_idx[inst], n_sections))
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
                #aux = Insert_section_in_slots(aux, ind[c])
                #isto tá mal pq os elements c em ind podem ter tamanhos diferentes
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
    
    
    return Pop #list(array()) to remove tuple format :)


def Fitness(Individual, model, features, scaler_norm):    #*TODO*
    #score = [round(random.uniform(0, 1),3), round(random.uniform(0, 1),3)]
    Scores_user=[[0]]
    Pop = [Individual]
    
    Pop = repeat_arrays(Pop, 5)
            
    X_ind, _ , _ , _ = LSTM.Training_data_for_GA_CNN(Pop, features, Scores_user, 1, scaler_norm) #Scores entry (y_ind) doesn´t matter
    
    
    X_ind = np.array(X_ind)
    score = model.predict(X_ind, verbose=0)

    return score[0][0]
    

def Select(Pop, Scores):
    Scores_array = np.array(Scores)
    #Scores_array = np.sum(Scores_array, axis=1) 
    
    idx = np.argsort(Scores_array.T)
    
    Pop_array = np.array(Pop)
    return list(Pop_array[idx[int(len(idx.T)/2):]])

def Crossover(Pop, user_info):#just switches one instrument
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
        new_Pop[i][inst] = Pop[i][inst].copy()
    
    return Pop + new_Pop

def equal_order_lists(list_a, list_b):
    """
    Compare two lists and check if any element in List A is equal 
    to the element in List B with the same index.
    """
    for i in range(len(list_a)):
        if np.array_equal(list_a[i], list_b[i]):
            return True
    return False

def Mutation_random(Pop, class_idx, midi_info_list):
    n_sections = len(Pop[0][0])
    
    for i in range(len(Pop)):
        inst = random.randint(0, 2) #random instrument (3)
        
        if random.uniform(0, 1)<=0.5:
            mutated_samp=random.randint(0, n_sections-1)
            rand_samp=mutated_samp
            while rand_samp==mutated_samp: #to avoid disrupting the section sequence
                rand_samp=random.sample(set(class_idx[inst]), 1)[0]
            Pop[i][inst][mutated_samp] = rand_samp
    return Pop

def Mutation(Pop, class_idx, user_info, midi_info_list, prob=0.3): 
    
    for i in range(len(Pop)):        
        if random.uniform(0, 1)<=prob:
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument (3)
            unique, counts = np.unique(Pop[i][inst], return_counts=True) #gather different samples
            
            if len(unique)>1:
                switch= random.sample(range(len(unique)),2) #pick which samples to switch
                #switching place of samples
                aux=copy.deepcopy(Pop[i][inst]) # only used to keep original order to find indexes
                Pop[i][inst][np.where(Pop[i][inst]==unique[switch[0]])[0]] = unique[switch[1]]
                Pop[i][inst][np.where(aux==unique[switch[1]])[0]] = unique[switch[0]]   
                
        if random.uniform(0, 1)<=prob:
            inst = random.randint(0, (user_info.n_inst-1)) #random instrument (3)
            
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


def Create_Sound_for_training(Pop, user_info, midi_info_list, sample_info, midi_file):
    for i in range(len(Pop)):
        print('\n Building example ',i ,'...')
        sr=22050
        BPM=user_info.bpm
        
        pb.PLAYBACK_midi_samples(Pop[i], user_info, midi_info_list,
                    sample_info, midi_file, sr, BPM, ex=i)

    return


def Fit_Model_CNN(Pop, df, Scores_user, features):
    
    
    #for the case of training with 50k (fake) and testing with 3k (real)
    ##################################################
    Pop_real = Pop[:3000].copy()
    Scores_user_real = Scores_user[:3000].copy()
    Pop = Pop[3000:]
    Scores_user = Scores_user[3000:]

    # TESTAR TIRAR OS ZEROS
    zero_idx = np.where(np.array(Scores_user_real)==0)[0]
    scores=np.array(Scores_user_real)
    scores[zero_idx]=-1
    Scores_user_real = list(scores)
    
    #Pop_real, Scores_user_real, _ = LSTM.Pad_zeros(Pop_real, Scores_user_real, features)
    Pop_real = repeat_arrays(Pop_real, 5)
    ##################################################
    
    # TESTAR TIRAR OS ZEROS
    zero_idx = np.where(np.array(Scores_user)==0)[0]   
    scores=np.array(Scores_user)
    scores[zero_idx]=-1
    Scores_user = list(scores)
    
    
    Pop_ex = repeat_arrays(Pop, 5)
    
    #Pop_pad, Scores_pad, features = LSTM.Pad_zeros(Pop, Scores_user, features)
    
    #Pop_ex += Pop_pad
   # Scores_user += Scores_pad
    
    #choose a fifth of Pop_ex for training(for the final model)
    test_idx = random.sample(list(range(len(Pop_ex))), int(len(Pop_ex)/5))
    
    Pop_test = [Pop_ex[idx] for idx in test_idx]
    scores_test = [Scores_user[idx] for idx in test_idx]
    #remove those individuals from Pop_ex
    Pop_train = [item for idx, item in enumerate(Pop_ex) if idx not in test_idx]
    scores_train = [item for idx, item in enumerate(Scores_user) if idx not in test_idx]
    
    X_train, y_train, scaler, pca = LSTM.Training_data_for_GA_CNN(Pop_train, features, scores_train, 0, 0, 0)
    
    Save_Object(scaler,  'C:/Users/joaop/Desktop/TESE/dubgen/MAIN', Pickle_name="stored_cnn_model_scaler.pickle", 
                    folder='/GA1_training/Model_Data/')

    X_train, y_train = shuffle(X_train, y_train)
 
    metric= tf.keras.metrics.MeanAbsoluteError()
    
    opt=list()
    opt.append(tf.keras.optimizers.Adam(learning_rate=0.001))
    opt.append(tf.keras.optimizers.Adam(learning_rate=0.01))
    #opt.append(tf.keras.optimizers.Adam(learning_rate=0.1))
    opt.append(tf.keras.optimizers.SGD(learning_rate=0.01))
    #opt.append(tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))
    
    
    es = tf.keras.callbacks.EarlyStopping( monitor="val_loss", min_delta=0, patience=50, 
                    verbose=0, mode="auto", baseline=None, restore_best_weights=True)
    cv = KFold(n_splits=3, random_state=33, shuffle=True)
    
    #AINDA NÃO SE TESTOU REGULARIZAÇÃO!!!!!!!!!!! L1, L2...
    
    param = dict(epochs=[150], n_sections = [0],
                      n_inst = [3], n_feat = [np.shape(X_train[0])[1]],
                      drop = [0.2, 0.4],
                      opt = opt, fc_act=['relu', 'sigmoid'])
    
    
    a_fifth = int(len(X_train)/5)
    X_v = X_train[:a_fifth]
    y_v =y_train[:a_fifth]
    
    X_t = X_train[a_fifth:]
    y_t=y_train[a_fifth:]
    
    #treinar lstm com input de varying length temporal
    #https://ai.stackexchange.com/questions/22763/how-to-train-an-lstm-with-varying-length-input
    print('X shape:', np.shape(np.array(X_t)))
    '''
    kgs = KerasGridSearch(LSTM.get_model_CNN2, param, monitor='val_loss', greater_is_better=False)
    kgs.search(np.array(X_t), np.array(y_t), validation_data=(np.array(X_v), np.array(y_v)), callbacks=[es], batch_size=128)
    
    param_best = kgs.best_params #get best parameters    
    print('best params:', param_best)
    '''
    #n_feat = np.shape(X_train[0])[1]
    
    param_best = dict(epochs=150, n_sections = 5,
                      n_inst = 3, n_feat = 12,
                      drop = 0.2,
                      opt = opt[0], fc_act='sigmoid', shape = (12, 12, 5, 1))
    
    #X_train_list, y_t_list = batch_by_length(X_train, y_t)
    #X_v_list, y_v_list = batch_by_length(X_v, y_v)
    X_t_list = [X_t]
    X_v_list = [X_v]
    y_t_list =  [y_t]
    y_v_list = [y_v]
    
    
    #model = LSTM.get_model_LSTM1(param_best)
    #model = LSTM.get_model_CNN2_light(param_best)
    model = LSTM.get_model_CNN3(param_best)
    model.summary()
    
    for X_train, y_t, X_v, y_v in zip(X_t_list, y_t_list, X_v_list, y_v_list):
        n_epochs = 1000
        history = model.fit(np.array(X_t), np.array(y_t), validation_data=(np.array(X_v), np.array(y_v)),
                            batch_size = 512, epochs = n_epochs,
                             verbose=0,  callbacks=[es])
        LSTM.Analyse_Model(history, 'mean_absolute_error', n_epochs)

    X_test, y_test, _, _ = LSTM.Training_data_for_GA_CNN(Pop_test, features, scores_test, 1, scaler, pca)

    #X_test_list, y_test_list = batch_by_length(X_test, y_test)
    X_test_list = [X_test]
    y_test_list= [y_test]
    

    for X_test, y_test in zip(X_test_list, y_test_list):
        print('\n')
        print('batch test:')
        X_test=np.array(X_test)
        y_test= np.array(y_test)
        
        y_hat=model.predict(X_test)
        
        mse = (np.square(y_test - y_hat)).mean(axis=None)
        print('test mse: ', round(mse, 3))
        
        y_class = y_hat.copy()
        y_class[y_class>0.5]=1
        y_class[y_class<0.5]=0
        y_class = y_class.flatten()
        accuracy = accuracy_score(y_test, y_class)
        precision = precision_score(y_test, y_class)
        precision = precision_score(y_test, y_class)
        recall = recall_score(y_test, y_class)
        f1 = f1_score(y_test, y_class)
        print('test accuracy: ', round(accuracy, 3))
        print('test precision: ', round(precision, 3))
        print('test recall: ', round(recall, 3))
        print('test f1-score: ', round(f1, 3))
        
        r1 = np.mean(y_test)
        print("\ny_test mean: ", round(r1, 3))
        r1 = np.mean(y_hat)
        print("\ny_hat mean: ", round(r1, 3))
        
        r2 = np.std(y_hat)
        print("\ny_hat std: ", round(r2, 3))
    
    #print(y_class)
    
    filename1 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_test.txt'
    filename2 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_hat.txt'
    filename3 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_class.txt'
    
    with open(filename1, "w") as f1, open(filename2, "w") as f2, open(filename3, "w") as f3:
        for i in range(len(y_test)):
            f1.write("%s\n" % y_test[i])
            f2.write("%s\n" % y_hat[i])
            f3.write("%s\n" % y_class[i])
            
           
    #################################################################################################################
    print('\nTESTING WITH REAL DATA')
    X_test, y_test, _, _ = LSTM.Training_data_for_GA_CNN(Pop_real, features,
                                                         Scores_user_real, 1, scaler, pca)

    #X_test_list, y_test_list = batch_by_length(X_test, y_test)
    X_test_list = [X_test]
    y_test_list= [y_test]
    

    for X_test, y_test in zip(X_test_list, y_test_list):
        print('\n')
        print('batch test:')
        X_test=np.array(X_test)
        y_test= np.array(y_test)
        
        y_hat = model.predict(X_test)
        
        mse = (np.square(y_test - y_hat)).mean(axis=None)
        print('test mse: ', round(mse, 3))
        
        y_class = y_hat.copy()
        y_class[y_class>0.5]=1
        y_class[y_class<0.5]=0
        y_class = y_class.flatten()

        accuracy = accuracy_score(y_test, y_class)
        precision = precision_score(y_test, y_class)
        precision = precision_score(y_test, y_class)
        recall = recall_score(y_test, y_class)
        f1 = f1_score(y_test, y_class)
        print('test accuracy: ', round(accuracy, 3))
        print('test precision: ', round(precision, 3))
        print('test recall: ', round(recall, 3))
        print('test f1-score: ', round(f1, 3))
        
        r1 = np.mean(y_test)
        print("\ny_test mean: ", round(r1, 3))
        r1 = np.mean(y_hat)
        print("\ny_hat mean: ", round(r1, 3))
        
        r2 = np.std(y_hat)
        print("\ny_hat std: ", round(r2, 3))
    
    #print(y_class)
    
    filename1 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_test_3K.txt'
    filename2 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_hat_3K.txt'
    filename3 = 'C:/Users/joaop/Desktop/TESE/dubgen/MAIN/y_class_3K.txt'
    
    with open(filename1, "w") as f1, open(filename2, "w") as f2, open(filename3, "w") as f3:
        for i in range(len(y_test)):
            f1.write("%s\n" % y_test[i])
            f2.write("%s\n" % y_hat[i])
            f3.write("%s\n" % y_class[i])
    ##################################################################################################################
    
    return model, scaler

class GA_info:
    def __init__(self, POP_SIZE, N_GEN):
        self.POP_SIZE = POP_SIZE
        self.N_GEN = N_GEN

def batch_by_length(X,y):
    top_len=5
    #sorted_X = list(sorted(X, key = len))
    X_sorted = list()
    y_sorted = list()
    
    for length in range(top_len):
        X_sorted.append(list())
        y_sorted.append(list())
        for i in range(len(X)):
            if len(X[i]) == length+1:
                X_sorted[length].append(X[i])
                y_sorted[length].append(y[i])
                
    
    return X_sorted, np.array(y_sorted)

#creates class_idx vector and generates a population
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
    
    print("\r\033[2KStarting Genetic Algorithm...", end="", flush=True)

    Pop = Create_Population(n_sections, class_idx, midi_info_list, n_pop, seed)
    
    return class_idx, df, Pop

class info_pop:
    def __init__(self, Pop_train, class_idx_train, seed, first_section):
        self.Pop_train = Pop_train
        self.class_idx_train = class_idx_train
        #self.idx_musician = idx_musician
        self.seed = seed
        self.first_section = first_section
        
def Load_Object(path, Pickle_name="stored_object.pickle", 
                folder = '/Data/User_Data/'):
    
    file_to_read = open(path + folder
                        + Pickle_name, "rb")
    loaded_object = pickle.load(file_to_read)
    
    file_to_read.close()

    return loaded_object

def Save_Object(Object, path, Pickle_name="stored_object.pickle", 
                folder= '/Data/User_Data/'):
    
    file_to_store = open(path+ folder
                         + Pickle_name, "wb")
    pickle.dump(Object, file_to_store)
    
    file_to_store.close()

    return

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
def GA_PREPOP(user_info, midi_info_list, sample_info):
    POP_SIZE = 30
    N_GEN=30
    
    info_GA = GA_info(POP_SIZE, N_GEN)
    
    #for testing (actual GA)
    class_idx, df_test, Pop_test = create_pop_info(midi_info_list, sample_info,
                                                      n_pop = POP_SIZE) 
    
    #print('Pop is a',type(Pop_test), ' with shape:', np.shape(np.array(Pop_test)) )
    
    return info_GA, class_idx, Pop_test, df_test


#cut individuals so that they have less sections (useful for training timeseries)
def cut_sections(Pop):
    n_sections = np.shape(Pop[0])[1]
    
    #first_section records what was the first section of that Ind, so we can
    #do the playback starting in the correct section
    first_section = list(np.zeros(len(Pop)).astype('int'))
    
    #indexes where we will cut sections
    cut_idxs = random.sample(range(len(Pop)), int(2*len(Pop)/3))
    
    for i in range(len(cut_idxs)):
        idx = cut_idxs[i]
        sec_end = random.randint(1,n_sections)
        sec_beg = random.randint(0, min(sec_end-1, n_sections-2))
        Pop[idx] = Pop[idx][:,sec_beg:sec_end]
        first_section[idx]=sec_beg
    
    return Pop, first_section

def repeat_arrays(arr_list, desired_length):
    max_len = max(max([arr.shape[1] for arr in arr_list]), desired_length)
    #print('max_len:',max_len)
    repeated_arr_list = []
    for arr in arr_list:
        repeated_arr = repeat_sections(arr, max_len)
        repeated_arr_list.append(repeated_arr)
    return repeated_arr_list


def repeat_sections(arr, max_len):
    current_len = arr.shape[1]
    if current_len < max_len:
        repeated_sections = np.repeat(arr, np.ceil(max_len / current_len), axis=1)
        start_idx = np.random.randint(0, repeated_sections.shape[1] - max_len + 1)
        repeated_sections = repeated_sections[:, start_idx:start_idx+max_len]
        return repeated_sections
    else:
        return arr


def remove_duplicates(arrays):#, scores):
    unique_arrays = []
    #unique_scores = []
    seen_arrays = []
    for i, array in enumerate(arrays):
        flattened_array = array.flatten()
        if flattened_array.tolist() not in seen_arrays:
            seen_arrays.append(flattened_array.tolist())
            unique_arrays.append(array)
            #unique_scores.append(scores[i])
    return unique_arrays#, unique_scores

def add_new_pop(Pop_unique, class_idx, user_info, midi_info_list, n_gen):
    
    Pop_all=Pop_unique.copy()
    
    while len(Pop_all) < n_gen:
        Pop_new = Mutation(Pop_all.copy(), class_idx, user_info, midi_info_list, prob=1)
        
        Pop_all = Pop_all + Pop_new
        
        Pop_all = remove_duplicates(Pop_all)
        
    return Pop_all[:n_gen]
    

def GA(user_info, sample_info, class_idx, midi_info_list,
        Pop, df_test, info_GA, midi_info_real):
    
    midi_files=list() #these midi files are output (WITH MOD)
    for filename in os.listdir(os.path.dirname(user_info.main_path)+'/output/output_MIDI'):
        if (filename.endswith("_OUT.mid") or filename.endswith("_OUT.midi"))==True:
            if filename.startswith('BASS'): midi_files.append(filename)
            elif filename.startswith('HARMONY'): midi_files.append(filename)
            elif filename.startswith('MELODY'): midi_files.append(filename)
    
    POP_SIZE = info_GA.POP_SIZE
    N_GEN = info_GA.N_GEN
    
    new_feat=[]#feature vector with more features   
    
    _, _, _, new_feat, _ = SP.Create_FeatureSpace( sample_info.flat_samp , window = 2048, SR = 22050)
    
    #print('Preparing Model')
    
    model_path = '/Data/Model_Data/program_model/NN_model/'
    model = load_model( user_info.main_path +model_path+"stored_cnn_model.h5")
    scaler= Load_Object(  user_info.main_path, Pickle_name="stored_cnn_model_scaler.pickle", 
                    folder=model_path)
    
    #Genetic algorithm
    gen=0
    
    #record best Individual
    goat_score = 0
    goat_Ind = np.array([])
    
    #early stop init
    es_counter=0
    early_stop=False
    
    #print('Pop is a',type(Pop), ' with shape:', np.shape(np.array(Pop)) )

    while gen< N_GEN and early_stop==False:
        
        ##########################################################################
        Pop_unique = remove_duplicates(Pop)
        
        if len(Pop_unique)!=len(Pop):
            Pop = add_new_pop(Pop, class_idx, user_info, midi_info_list, POP_SIZE)   
        ##########################################################################
        
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
        print("\r\033[2KGeneration {}/{}".format(gen+1, N_GEN),
              end=" (Best Score: {})".format(best_score), flush=True)
        #print('worst fitness:', round(min(Scores),3))
        
        new_Pop = Select(Pop, Scores)
        new_Pop = Crossover(new_Pop, user_info)
        Pop = Mutation(new_Pop, class_idx, user_info, midi_info_list, 0.3)
        
        gen+=1
        es_counter +=1
    
    best_Ind = goat
    
    print("\r\033[2KGenerating Best Arrangement...",
              end="", flush=True)
    music_path=str()
    
    #Save_Object(best_Ind,  user_info.main_path, Pickle_name="stored_best_Ind.pickle", folder='/GA1_training/Model_Data/')
    
    music_path = pb.PLAYBACK_midi_samples(best_Ind, user_info, midi_info_real,
                                          sample_info, midi_files, sr = 22050, BPM = user_info.bpm,
                                          ex = -1, id_name='_with_mod_no_synth')

    
    return music_path, max(Scores), best_Ind