"""
Created on Tue Apr 5 09:49:25 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: script with functions for Sampling Processing: converting to arrays,\\
    cutting into samples, extracting features and classifying sounds
"""

import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import os

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

from PLAY_MIDI_AND_SAMPLES_j import Unclipping
from LSTM_j import Norm_and_Scale

import warnings
warnings.filterwarnings("ignore")

# RMS Normalization
def RMS_Norm(sig,rms_level):
    #rms level is in dBs
    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    coef = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )

    # normalize
    norm_sig = (sig * coef)
    return norm_sig, coef

#normalize audio to range [0,1]
def normalize_audio(sig):
    max_amp = max(abs(sig))
    return sig/max_amp, (1/max_amp)

#find input audio onset markers
def Find_Onset(sig):
    window = 2048
    onset_stamps = [0]*(round(len(sig)/window)+1)
    idx = 0
    e=0.0001
    
    for w in range(1,round(len(sig)/window)):
        if sum(abs(sig[window*(w-1):window*w]))/(sum(abs(sig[w*window:(w+1)*window]))+e)<0.7:
            onset_stamps[idx] = window*(w-1)
            idx+=1
    onset_stamps = onset_stamps[0:idx]
    
    if (not onset_stamps) or onset_stamps[0] != 0:
        onset_stamps.insert(0,0)
        
    onset_stamps.insert(len(onset_stamps), len(sig)-1)
    
    return np.array(onset_stamps)

#remove silence from audio
def Remove_Silence(sample):

    yt, idx = librosa.effects.trim(sample, top_db=8, frame_length=1024, hop_length=512)

    e_frame=10000 #almost 0.5 seconds
    e_hop=int(e_frame/2)
    i=1
    s=0
    E_max_window=0
    
    while s<len(yt):
        if sum(abs(yt[(i-1)*e_frame:i*e_frame])**2)>E_max_window:
            E_max_window = sum(abs(yt[(i-1)*e_frame:i*e_frame])**2)
            s = len(yt)

        s = ((i+1)*e_frame)- e_hop
        i += 1
    
    energy_avg = E_max_window
    
    frame=1024
    hop=512
    i=1
    s=0
    
    silence_threshold=0.03
    
    while s<len(yt):
        if sum(abs(yt[(i-1)*frame:i*frame])**2)/sum(abs(
                yt[(i*frame)-hop:((i+1)*frame)-hop])**2)<(silence_threshold*energy_avg):
            yt = yt[i*frame:]
            s = len(yt)
            idx[0]= i*frame
        else:
            s = ((i+1)*frame)-hop
            i += 1

    return yt, idx


#takes onset stamps and creates individual samples
def Create_Samples(sig, onset_stamps, window, norm_coef, sig_stereo):
    samples_for_inst = [[0]]*(len(onset_stamps)) #downsampled and mono sample
    aux=0
    
    volume_avg = sum(abs(sig))/len(sig)
    
    for i in range(len(onset_stamps)-1):
        delta = onset_stamps[i+1]-onset_stamps[i]
        
        #avoid silence and short samples
        if sum(abs(sig[onset_stamps[i]:onset_stamps[i+1]]))/delta>0.3*volume_avg and delta > 4*window: 
            
            #in this next function, sampling is a half
            yt, idx = Remove_Silence(sig[onset_stamps[i]:onset_stamps[i+1]]/norm_coef)
            
            samples_for_inst[aux] = yt*norm_coef

            aux +=1
    
    return samples_for_inst[0:aux]


# determines SPECTRAL ADSR markers of a sound
def Envelope_ADSR(feat_vec, Y, Spec, feat_aux):
   
    size_box = int(np.ceil(len(feat_vec[2])/20))
    box = np.ones(size_box)/size_box
    poly = feat_vec[2]
    poly/=max(poly)
    poly1 = np.convolve(poly, box, mode='same')# smoothing 1
    poly2 = np.convolve(poly1, box, mode='same')# smoothing 2
    poly3 = np.convolve(poly2, box, mode='same')# smoothing 3
    
    poly_der = np.diff(poly3)
    
    #in case we dont detect a correct adsr
    len_poly=len(feat_vec[2])
    fake_adsr=[int(len_poly/4), int(len_poly/2), int(3*len_poly/4)]
    
    zero_crossings = np.where(np.diff(np.signbit(poly_der)))[0]
    
    if len(zero_crossings) ==0:
        return fake_adsr
    
    #Attack: first derivative zero after prev_A
    A = max(1,zero_crossings[0])
    
    #only consider zeros after A
    zero_cross = zero_crossings[zero_crossings>A]
    
    if len(zero_cross) > 1:
        
        D = zero_cross[1] #DECAY 2nd zero of derivative
        S= zero_cross[-1]
        
        if (A<D<S) and ((S-A)>(size_box*4)):
            return [A, D, S]
        else:
            return fake_adsr
            
    else:
        return fake_adsr


# Create a "light" Feature vector
def Redux_Features(feature_vec):
    mfcc = feature_vec[3]
    n_mfcc = len(mfcc)
    n_features=5 + n_mfcc -1
    features_classify = [[]]*n_features 
    
    A = feature_vec[10][0] #Attack index
    S = feature_vec[10][2] #Sustain index
    
    #mean MFCC coef for the A-S interval
    features_classify[0:n_mfcc] = (np.sum(mfcc[:,A:S], axis = 1)/(S-A)) 
    #mean centroid #WITH LOG NORMALIZATION
    features_classify[n_mfcc] = float(np.sum(feature_vec[4][:,A:S], axis = 1)/(S-A))
    #mean roll of (top and bottom)
    features_classify[n_mfcc + 1] = float(np.sum(feature_vec[8][:,A:S], axis = 1)/(S-A))
    features_classify[n_mfcc + 2] = float(np.sum(feature_vec[9][:,A:S], axis = 1)/(S-A))
    #mean spectral flatness
    features_classify[n_mfcc + 3] = float(np.sum(feature_vec[1][:,A:S], axis = 1)/(S-A))
    
    return features_classify

# Create a bigger Feature vector
def Redux_Features_bigger(feature_vec):
    mfcc = feature_vec[3]
    features_classify = list()
    
    A = feature_vec[10][0] #Attack index
    D = feature_vec[10][1] #Decay index
    S = feature_vec[10][2] #Sustain index
    
    #mean mfcc inside A-S interval    
    features_classify.extend(list(np.sum(mfcc[:,A:D], axis = 1)/(D-A)))
    features_classify.extend(list(np.sum(mfcc[:,D:S], axis = 1)/(S-D)))
    #mean centroid WITH LOG NORMALIZATION
    features_classify.extend(list(np.sum(feature_vec[4][:,A:D], axis = 1)/(D-A)))
    features_classify.extend(list(np.sum(feature_vec[4][:,D:S], axis = 1)/(S-D)))
    #mean roll of (top and bottom)
    features_classify.extend(list(np.sum(feature_vec[8][:,A:D], axis = 1)/(D-A)))
    features_classify.extend(list(np.sum(feature_vec[8][:,D:S], axis = 1)/(S-D)))
    #mean roll of (top and bottom)
    features_classify.extend(list(np.sum(feature_vec[9][:,A:D], axis = 1)/(D-A)))
    features_classify.extend(list(np.sum(feature_vec[9][:,D:S], axis = 1)/(S-D)))
    #mean spectral flatness
    features_classify.extend(list(np.sum(feature_vec[1][:,A:D], axis = 1)/(D-A)))
    features_classify.extend(list(np.sum(feature_vec[1][:,D:S], axis = 1)/(S-D)))
    
    return features_classify


# features for MISC Classification
def MISC_Features(feature_vec):
    spec_contrast = feature_vec[0]
    n_contrast = len(spec_contrast)  
    n_features= n_contrast + 4
    features_classify = [[]]*n_features
    
    A = feature_vec[10][0] #Attack index
    S = feature_vec[10][2] #Sustain index
    
    #spec_contrast
    features_classify[:n_contrast] = (np.sum(spec_contrast[:,A:S], axis = 1)/(S-A))
    #zcr
    features_classify[n_contrast] = float(np.sum(feature_vec[2][:,A:S], axis = 1)/(S-A))
    #spectral flatness
    features_classify[n_contrast +1] = float(np.sum(feature_vec[1][:,A:S], axis = 1)/(S-A))
    #bandwidth
    features_classify[n_contrast +2] = float(np.sum(feature_vec[7][:,A:S], axis = 1)/(S-A))
    #spectral centroid
    features_classify[n_contrast +3] = float(np.sum(feature_vec[4][:,A:S], axis = 1)/(S-A))
    
    return features_classify

# Create Feature vectors for all samples
def Create_FeatureSpace(samples, window=2048, SR=22050):
    n_features = 11
    feature_vec = [[]]*len(samples)
    feature_classify = [[]]*len(samples)
    feature_misc = [[]]*len(samples)
    feature_bigger = [[]]*len(samples)
    features_cnn = [[]]*len(samples)
    N_FFT = 512
    
    
    for x in range(0,len(samples)):
        feat_aux=[[]]*n_features
        Y = np.array(samples[x]).T
        Y = Unclipping(Y,0)
        
        Spec = np.abs(librosa.stft(y = Y , n_fft = N_FFT))
        #Noise
        feat_aux[0] = librosa.feature.spectral_contrast(y=Y, sr=SR, S = Spec, n_bands = 3, n_fft = N_FFT)
        feat_aux[1] = librosa.feature.spectral_flatness(y=Y, n_fft = N_FFT, S = Spec)
        feat_aux[2] = librosa.feature.zero_crossing_rate(y=Y)
        #Timbre
        feat_aux[3] = librosa.feature.mfcc(y= Y , sr=SR, hop_length=window, n_mfcc=12,n_fft = N_FFT , S = Spec) 
        feat_aux[4] = librosa.feature.spectral_centroid(y=Y, sr=SR, hop_length=window, n_fft = N_FFT , S = Spec)
        feat_aux[5] = librosa.feature.poly_features(y=Y, sr=SR, n_fft = N_FFT , S = Spec, order=2) #define order! talvez um pouco redundante relativamente a um espectrograma
        #Register (+Dissonance+ Polyphony)
        feat_aux[6] = librosa.feature.chroma_stft(y= Y, sr=SR, n_fft = N_FFT , S = Spec)
        feat_aux[7] = librosa.feature.spectral_bandwidth(y=Y, sr=SR, n_fft = N_FFT, S = Spec)
        feat_aux[8] = librosa.feature.spectral_rolloff(y=Y, sr=SR, n_fft = N_FFT, S = Spec, roll_percent=0.99)
        feat_aux[9] = librosa.feature.spectral_rolloff(y=Y, sr=SR, n_fft = N_FFT, S = Spec, roll_percent=0.01) #rolloff min
        
        feat_aux[10] = Envelope_ADSR(feat_aux[5], Y, Spec, feat_aux)
        
        feature_vec[x] = feat_aux
        
        feature_classify[x] = Redux_Features(feature_vec[x])
        feature_misc[x] = MISC_Features(feature_vec[x])
        feature_bigger[x]= Redux_Features_bigger(feature_vec[x])
        features_cnn[x] = [get_Features_CNN(feat_aux[3].copy(), time_stamps = feat_aux[10])]
    
    return feature_vec, feature_classify, feature_misc, features_cnn, feature_bigger


#get features for CNN model (ADSR frame with mfccs)
def get_Features_CNN(mfccs, time_stamps=list()):
    mfccs = mfccs.T
    
    time_steps=4
    time_bins = mfccs.shape[0]
    bin_size = time_bins // time_steps
    
    reduced_mfccs = np.zeros((time_steps, mfccs.shape[1]))
    
    #if no ADSR is delivered, create equidistant markers
    if time_stamps==list():
        for i in range(time_steps):
            start_bin = i * bin_size
            end_bin = start_bin + bin_size
            reduced_mfccs[i, :] = np.mean(mfccs[start_bin:end_bin, :], axis=0)
    else:
        last_stamp=0
        i=0
        time_stamps.append(time_bins)
        
        for time_stamp in time_stamps:
            
            reduced_mfccs[i, :] = np.mean(mfccs[last_stamp:time_stamp, :], axis=0)
            last_stamp=time_stamp
            i+=1
        
    return reduced_mfccs.T
    
# create SVC model and also return predictions
def get_SVC(Features, n_instruments, true_classes, classifier, misc=1, train=1):  
    x = Features
    y=true_classes
    x, y = shuffle(x, y)
    
    #miscellaneous classification
    if misc==1:
        y = list(map(lambda i: 0 if  i==1 or i==2 else i, y))
        y = list(map(lambda i: 1 if i==3 else i, y))
    
    # if we are creating SVC model, else, we just do predictions
    if train==1:
        parameters = {'kernel':('poly', 'rbf', 'sigmoid'), 'C':[1, 10], 'gamma': ('scale', 'auto'), 
                      'class_weight':('None', 'balanced'), 'decision_function_shape': ('ovo', 'ovr')}

        clf = GridSearchCV(classifier, parameters)        
        clf.fit(x,y)
        
        return clf.predict(Features), clf, true_classes
    
    return classifier.predict(x), classifier, y


#if a class has no samples, attribute n samples that are closest to the specific decision boundary
def attribute_closest_samples(Samples_List, Class_MISC, clf, class_id):
    #number of samples to be displaced
    n_samples=3
    
    #distance to each decision boundary
    boundary_dist = clf.decision_function(Samples_List)
    w_norm = np.linalg.norm(clf.coef_)
    boundary_dist = boundary_dist / w_norm
    
    #see what samples are closer to boundary
    dist_class = boundary_dist[:,class_id]
    idx_closest = np.argsort(dist_class)[:n_samples]
    
    Class_MISC[idx_closest] = class_id 
    
    return Class_MISC

#remove detected miscellaneous samples
def Remove_MISC(Samples_List, prev_samp_list, class_MISC, true_classes, Sample_Label,
                keep_classes, clf, leave_misc = 0):    
    idx_keep=np.array([])
    for class_id in keep_classes:
        idx_cl =np.where(np.array(class_MISC)==class_id)[0]
        
        if len(idx_cl)<1:
            class_MISC = attribute_closest_samples(prev_samp_list, class_MISC, clf, class_id)
            idx_cl =np.where(np.array(class_MISC)==class_id)[0]
        
        idx_keep = np.append(idx_keep, idx_cl)
    
    if leave_misc==0:
        idx_keep = np.sort(idx_keep).astype('int')
    else:
        idx_keep = np.range(len(class_MISC))
    
    array_samples = np.array(Samples_List)
    array_classes = np.array(true_classes)
    
    Samples_List_new = list(array_samples[idx_keep])
    true_classes = list(array_classes[idx_keep])

    aux=0
    sample_idx = []
    n_samp = -1
    count_sound = -1
    
    for sound in Sample_Label:
        aux=0
        count_sound += 1
        for samp in range(sound[1]):
            n_samp += 1 
            if n_samp in idx_keep:
                sample_idx.append([count_sound, aux])
                aux+=1
   
    return Samples_List_new, true_classes, sample_idx, idx_keep


#create wav file
def Create_wav(Samples, classes, sample_idx, sr, path):
    filetype='.wav'
    
    for s in range(len(sample_idx)):
        if classes[s]==0: inst='BASS '
        elif classes[s]==1: inst='HARMONY '
        else: inst='MELODY '
    
        sf.write(path+ inst + 'sample_' + str(sample_idx[s][0]) + '_' + str(sample_idx[s][1])
                 + filetype, Samples[sample_idx[s][0]][sample_idx[s][1]], samplerate=sr)
    return
    

#class with sample info
class info_sample:
    def __init__(self, df_samples, flat_samp, sample_idx, norm_coef,
                 filenames, visited_samples, visited_info):
        self.df_samples = df_samples
        self.flat_samp = flat_samp
        self.sample_idx = sample_idx
        self.norm_coef = norm_coef
        self.filenames = filenames
        self.visited_samples = visited_samples
        self.visited_info = visited_info

#class with model info
class info_model:
    def __init__(self, model_misc, model_inst, scaler_misc, scaler_inst):
        self.model_misc = model_misc
        self.model_inst = model_inst
        self.scaler_misc = scaler_misc
        self.scaler_inst= scaler_inst

#class with pca info
class info_pca:
    def __new__(cls, *args, **kwargs):
        #print("1. Create a new instance of info_pca.")
        return super().__new__(cls)
    
    def __init__(self, pca, pca_bigger, clf_pca, clf_pca4):
        self.pca = pca
        self.pca_bigger = pca_bigger
        self.clf_pca = clf_pca
        self.clf_pca4 = clf_pca4

# processes sounds and cuts them into samples. also gets features and classifies them
def SAMPLE_PROCESSING(user_info, train=1, leave_misc=0 ,
                      model_info=None, pca_info_misc=None, pca_info_inst=None, tkinter=None):
    
    Feature_vec = {}
    Samples_down = {}
    Features_Classify = {}
    Features_Misc = {}
    
    if train==1:
        folder = user_info.sound_folder
    else: folder = user_info.test_path
    
    s = 0
    n_inst = user_info.n_inst
    window = 2048
    down_sampling = 22050
    
    Sample_label = []
    true_classes = []
    norm_coef = []
    filenames = []
    use_dubgen_model=False
    
    if train==1:
        inst_folders = ['/BASS/', '/HARMONY/', '/MELODY/', '/MISCELLANEOUS/' ]
    else:
        inst_folders =[str()]

    #count the files to appear on print
    total_sounds=0
    for inst_folder in inst_folders:
        for filename in os.listdir(folder+inst_folder):
            if filename.endswith(".wav"):
                total_sounds+=1
                
    #Process each sound on the folders
    for inst_folder in inst_folders:
        count_wavs=0
        for filename in os.listdir(folder+inst_folder):
            if filename.endswith(".wav"):
                count_wavs+=1
        
                filepath = folder +inst_folder + '/' + filename
                data, samplerate= sf.read(filepath) 
                
                if data.ndim == 2:
                    data_mono = data.sum(axis=1)/2 #convert to mono
                else: data_mono = data
                
                if train==1:
                    if inst_folder == "/BASS/":
                        Sample_label.append([0,0]) 
                    elif inst_folder == "/HARMONY/":
                        Sample_label.append([1,0])
                    elif inst_folder == "/MELODY/":
                        Sample_label.append([2,0])
                    elif inst_folder == "/MISCELLANEOUS/":
                        Sample_label.append([3,0])       
                else:
                    Sample_label.append([0,0]) 
                
                #("\r\033[2KProcessing Sound {}/{}".format(s, total_sounds), end="", flush=True)
                text = 'Processing Sound ' + str(s) + '/' + str(total_sounds)+ ' ' 
                tkinter.config(text=text)
                
                #print('Processing Sound ', s, '...')
                #print(filename)
                filenames.append(filename)
                
                data_down = librosa.resample(data_mono, orig_sr=samplerate, target_sr= down_sampling)

                data_down, norm = normalize_audio(data_down)
                
                norm_coef.append(norm)
                
                onset_stamps = Find_Onset(data_down)
                
                #DownSampling
                Samples_down[s] = Create_Samples(data_down, onset_stamps, window, norm_coef[s], data)
                
                #save number of samples per sound
                Sample_label[s][1]=len(Samples_down[s])
                
                if Sample_label[s][1]==0:
                    #print('\nERROR IN SAMPLING (COULD NOT EXTRACT SAMPLE):',filename)
                    Sample_label.pop()
                    filenames.pop()
                    del Samples_down[s]
                    norm_coef.pop()

                else:
                    #Extracting Features
                    Feature_vec[s], Features_Classify[s], Features_Misc[s], _, _  = Create_FeatureSpace(
                                                                                Samples_down[s],
                                                                                window = 2048,
                                                                                SR = down_sampling)
                
                    s+=1
        if count_wavs<5:
              print('You need more WAV files! Please add sounds to the sound folders.')
              use_dubgen_model=True
        
    #flatten label list
    w=0
    for j in range(0,len(Sample_label)):
                true_classes[w:(w+Sample_label[j][1])]= [Sample_label[j][0]]*Sample_label[j][1]
                w += Sample_label[j][1]

    Feat_List = list(Features_Misc.values())
    flat_list = sum(Feat_List, [])
    
    Feat_List_c = list(Features_Classify.values())
    flat_list_c = sum(Feat_List_c, [])
    
    if train==1:
        text = 'Training Instrument Classification... '
        tkinter.config(text=text)
        
        #Transform the data (for MISC classification)  
        flat_list, scaler1 = Norm_and_Scale(np.array(flat_list))
        #Transform the data (for inst classification)
        flat_list_c, scaler2 = Norm_and_Scale(np.array(flat_list_c))
        
        #Classify Miscellaneous:
        class_MISC, model_misc, y = get_SVC(flat_list, n_inst, true_classes, SVC(), misc=1, train=1)
        
        #Remove classified as Miscellaneous
        flat_list_c, true_classes, sample_idx, idx_keep = Remove_MISC(flat_list_c, 
                                                                    flat_list, class_MISC, 
                                                                    true_classes, Sample_label, 
                                                                    [0], model_misc)
        
        #Remove misclassified Miscellaneous (for training)
        flat_list_c, true_classes, sample_idx, idx_keep = Remove_MISC(flat_list_c, 
                                                                    flat_list, true_classes, 
                                                                    true_classes, Sample_label, 
                                                                    [0, 1, 2],
                                                                    model_misc)
        
        #Classify Instruments      
        class_inst, model_inst, y = get_SVC(flat_list_c, n_inst, true_classes, SVC(), misc=0, train=1)
        
        model_info = info_model(model_misc, model_inst, scaler1, scaler2)
        
    else:
        text = 'Classifying Sounds... '
        tkinter.config(text=text)
        clf_misc = model_info.model_misc
        clf_inst = model_info.model_inst
        scl_misc = model_info.scaler_misc
        scl_inst = model_info.scaler_inst
                
        flat_list = scl_misc.transform(np.array(flat_list)) #scale misc
        flat_list_c = scl_inst.transform(np.array(flat_list_c)) #scale inst
        
        #Miscellaneous Sounds Detection      
        class_MISC, model_misc, y = get_SVC(flat_list, n_inst, true_classes, clf_misc,
                                            misc=1, train=0)
        
        #Remove Miscellaneous
        flat_list_c, true_classes, sample_idx, idx_keep = Remove_MISC(flat_list_c,
                            flat_list, class_MISC, true_classes, Sample_label,[0],
                            model_misc, leave_misc)
        
        #Instrument Classification
        class_inst, model_inst, y = get_SVC(flat_list_c, n_inst, true_classes, clf_inst, misc=0,
                                            train=0)
        
        #attribute samples in case some class is empty
        for class_id in [0,1,2]:
            if len(np.where(class_inst==class_id)[0])<1:
                class_inst = attribute_closest_samples(flat_list_c, class_inst, model_inst, class_id)
        
        text = 'Creating .wav files for samples... '
        tkinter.config(text=text)
        Create_wav(Samples_down, class_inst, sample_idx, sr=22050,
                   path = os.path.dirname(user_info.main_path)+'/output/output_samples/')
        
    # FOR THE GENETIC ALGORITHM: 
    # CREATE DATAFRAMES
    Samples_flat = list(Samples_down.values())
    flat_samp_aux = sum(Samples_flat, [])
    
    array_samples = np.array(flat_samp_aux)
    flat_samp = list(array_samples[idx_keep])
    
    df_samples=pd.DataFrame(columns =['Class'])
    df_samples['Class'] = class_inst

    column_names=['mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5',
                          'mfcc_6','mfcc_7','mfcc_8','mfcc_9',
                          'mfcc_10','mfcc_11','mfcc_12','sp_ctr',
                          'r_off_t','r_off_b','sp_flat']
    
    df_aux = pd.DataFrame(flat_list_c, columns=column_names)
    df_samples = pd.concat([df_samples,df_aux],axis=1)
    
    sample_info = info_sample(df_samples, flat_samp, sample_idx, norm_coef, filenames, list(), list())
    
    return sample_info, model_info, pca_info_misc, pca_info_inst, use_dubgen_model



