# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:45:05 2022

@author: JP
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow import stack
from keras import backend as K
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, MaxPooling2D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.activations import sigmoid
from tensorflow.keras import initializers
import itertools as it
import random 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def Training_data_for_GA(Pop, df, Scores, test, scaler_norm):
    
    #preparing dataframe to be used on nn
    features_df = df.copy()
    features_df = features_df.drop(['Class'], axis=1)
    feat_data = features_df.values.tolist()
    
    if test==1:
        norm_data = scaler_norm.transform(np.array(feat_data))
    else:        
        norm_data, scaler_norm = Norm_and_Scale(feat_data)
    
    X = list()
    for j in range(len(Pop[0])): #init X_train (list of 3 inst, each with shape: (n_ind, n_sections, n_features))
        X.append(np.zeros((len(Pop),np.shape(Pop[0])[1],len(norm_data[0]))))
    
    for i in range(len(Pop)):
        for inst in range(len(Pop[i])):
            for s in range(0, np.shape(Pop[i])[1]):
                sample = Pop[i][inst][s]
                features = norm_data[sample]
                X[inst][i][s]=features 
                
    #y_train = transform_score(np.array(Scores)) #force score to range [0,1]
    y = Scores
    
    count_pos = len(np.where(np.array(y)==1)[0])
    count_neg = len(np.where(np.array(y)==-1)[0])
    
    '''
    if test!=1:
        X, y = SMOTE_oversample(X, y)
    #n is the desired number of datapoints
    n=500
    #X_train, y_train = MIXUP_oversample(X_train, y_train, n)
    
    if test == 0 :
        if count_pos/count_neg<0.4 or count_neg/count_pos<0.4:
            X, y = SMOTE_oversample(X, y)
           
            #X_train, y_train = brute_oversample(X_train, y_train)
            
            #X_train, y_train = MIXUP_oversample(X_train, y_train)
    '''
    y = transform_score(np.array(y))
    
    if test !=1:
        X, y = brute_oversample_old(X, y)
        X=list(X)
    
    return X, np.array(y), scaler_norm

def Training_data_for_GA_var(Pop, df, Scores, test, scaler_norm, pca):
    
    #preparing dataframe to be used on nn
    features_df = df.copy()
    features_df = features_df.drop(['Class'], axis=1)
    feat_data = features_df.values.tolist()
    
    if test==1:
        norm_data = scaler_norm.transform(feat_data)
        
        fd_array=np.array(feat_data)
        mfccs_a = fd_array[:, :12]
        mfccs_b = fd_array[:, 12:24]
    else:        
        norm_data, scaler_norm = Norm_and_Scale(feat_data)
        
        fd_array=np.array(feat_data)
        mfccs_a = fd_array[:, :12]
        mfccs_b = fd_array[:, 12:24]
    
        pca = PCA(n_components=2)
        
        pca.fit(mfccs_a) #A TRANSFORMAÇÃO FOI FEITA COM MFCC_A COMO REFERENCIA!!
    
    
    reduced_mfccs_a = pca.transform(mfccs_a)
    reduced_mfccs_b = pca.transform(mfccs_b)
    
    norm_data = np.concatenate((reduced_mfccs_a,reduced_mfccs_b), axis=1)
        
    
    X = list()
    for j in range(len(Pop)): #init X_train (list of 3 inst, each with shape: (n_ind, n_sections, n_features))
        X.append(np.zeros((len(Pop[j]),np.shape(Pop[j])[1],len(norm_data[0]))))
    
    for i in range(len(Pop)):
        for inst in range(len(Pop[i])):
            for s in range(np.shape(Pop[i])[1]):
                sample = Pop[i][inst][s]
                features = norm_data[sample]
                X[i][inst][s]=features 
                
                
        #y_train = transform_score(np.array(Scores)) #force score to range [0,1]
    y = Scores
    y = transform_score(np.array(y))            
                
    X, y = brute_oversample(X, y)
    X=list(X)
                
    for r in range(len(X)):
        aux = X[r]
        n_inst, n_sec, n_feat = np.shape(aux) 
        
        aux = aux.reshape(n_sec, n_inst, n_feat)
        aux = aux.reshape(n_sec*n_inst*n_feat)
        
        X[r] = aux
                
    
    return X, np.array(y), scaler_norm, pca



def Training_data_for_GA_CNN(Pop, features, Scores, test, scalers, pca=0):
    
    feat_data = features
    
    array_data = np.array(feat_data)
    array_data = np.squeeze(array_data)
    #array_data = np.transpose(array_data,(1,2,0))
    
    if test==1:
        
        array_data=np.transpose(array_data,(1,2,0))
        
        for i in range(array_data.shape[0]):
            # Extract the current frequency band
            frequency_band = array_data[i].T
        
            # Apply the scaler for this frequency band to the new data
            scaler = scalers[i]
            frequency_band_scaled = scaler.transform(frequency_band)
        
            # Store the transformed frequency band back into the new data sample
            array_data[i] = frequency_band_scaled.T
        
        array_data=np.transpose(array_data,(2,0,1))
    else:        
        array_data, scalers = Norm_and_Scale_CNN_flat(array_data)

    #feat_data = np.transpose(array_data, (2,0,1))
    
    ############################# X será a concatenação dos mfccs dos 3 inst--->y=12 mfccs, x = 4+4+4 (4 timesteps de cada instrumento)
    X = list()
    #for j in range(len(Pop)): #init X_train (list of 3 inst, each with shape: (n_ind, n_sections, n_features))
       # X.append(np.zeros((len(Pop[j]),np.shape(Pop[j])[1],len(norm_data[0]))))
    
    for i in range(len(Pop)):
        mfcc_seq = list()  #(5 sections, 12 time steps, 12 mfccs)
        for s in range(np.shape(Pop[i])[1]):
            mfcc_panel = np.array([]) #(12,12) ate the end, which is the panels for each instrument concatenated (3inst*4timesteps, 12 coefs)
            for inst in range(len(Pop[i])):
                sample = Pop[i][inst][s]
                mfccs = array_data[sample]
                
                #mfccs comes as a list of an array, so we convert to array and remove the first dimension (1, 12, 4) to (12, 4)
                mfccs = np.squeeze(np.array(mfccs))
                if len(mfcc_panel)!=0:
                    mfcc_panel = np.concatenate((mfcc_panel, mfccs ), axis=1)
                else:
                    mfcc_panel = mfccs
            mfcc_seq.append(mfcc_panel)
            mfcc_array=np.array(mfcc_seq)
        
        #NA ARQUITETURA NORMAL (12,12,5) É MFCC_ARRAY.T.
        X.append(mfcc_array)
    ######################################################              
                
    #y_train = transform_score(np.array(Scores)) #force score to range [0,1]
    y = Scores
    y = transform_score(np.array(y))            
                
    if test!=1:
        X, y = brute_oversample(X, y)
        X=list(X)
    
    return X, np.array(y), scalers, pca




def custom_sigmoid(x):
    return ((2/(1 + K.exp(-x))) -1)
    #return ((1/(1 + K.exp(-x))))

def tanh(x):
    return (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))

def transform_score(x): #force score to range [0,1]
   return (0.5*x+0.5)

def transform_TO_score(x): #force score to range [-1,1]
   return (2*x-1)


def custom_mse(y_true, y_pred):
 
    # calculating squared difference between target and predicted values 
    loss = K.square(y_pred - float(y_true))  # (batch_size, 2)
                
    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)        # (batch_size,)
    
    print(loss)
    return loss

def get_model_LSTM3(param):    
    #explicação das dimensões: 
        #https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    
    #custom activation function (sigmoid between -1 and 1)
        
    n_inst = param['n_inst']
    n_feat = param['n_feat']
    
    inputA = layers.Input(shape=(None,n_feat)) 
    inputB = layers.Input(shape=(None,n_feat))
    inputC = layers.Input(shape=(None,n_feat))
    
    
    lstm_A = layers.LSTM(n_feat, dropout=param['drop'])(inputA)
                         #recurrent_dropout=param['recdrop'])(inputA)
    lstm_B = layers.LSTM(n_feat, dropout=param['drop'])(inputB)
                         #recurrent_dropout=param['recdrop'])(inputB)
    lstm_C = layers.LSTM(n_feat, dropout=param['drop'])(inputC)
                         #recurrent_dropout=param['recdrop'])(inputC)
    
    out_A = keras.Model(inputs=inputA, outputs=lstm_A)
    out_B = keras.Model(inputs=inputB, outputs=lstm_B)
    out_C = keras.Model(inputs=inputC, outputs=lstm_C)
    
   
    #t_stack = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([out_A.output, out_B.output, out_C.output])
    t_concat = tf.keras.layers.concatenate([out_A.output, out_B.output, out_C.output], axis=-1)
    #combined = stack([out_A.output, out_B.output, out_C.output],axis=1)
    #combined = tf.keras.layers.Reshape((n_feat, n_inst))(t_stack)
    
    #init_1 = tf.keras.initializers.HeNormal()

    if param['fc_act']== 'sigmoid':
        init_1 = tf.keras.initializers.GlorotNormal()
    else:
        init_1 = tf.keras.initializers.HeNormal()
        
        
    
    x1 = layers.Dense(12, activation=param['fc_act'],
                         kernel_initializer=init_1,
                         bias_initializer=initializers.Zeros())(t_concat)
    #x2 = tf.keras.layers.Reshape((1, n_feat))(x1)
    
    
    init_out = tf.keras.initializers.GlorotNormal()
    #model_output = layers.Dense(1, activation= custom_activation, kernel_initializer= init_out)(x1)
    
    fc_out = layers.Dense(3, activation= param['fc_act'], kernel_initializer= init_out)(x1)
 
    model_output = layers.Dense(1, activation= 'sigmoid', kernel_initializer= init_out)(fc_out)
    
    model = keras.Model(inputs=[out_A.input, out_B.input, out_C.input], outputs=model_output)
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    #print('hey')
    return model

def get_model_LSTM3_simple(param):    
    #explicação das dimensões: 
        #https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    
    #custom activation function (sigmoid between -1 and 1)
        
    n_inst = param['n_inst']
    n_feat = param['n_feat']
    
    inputA = layers.Input(shape=(None,n_feat)) 
    inputB = layers.Input(shape=(None,n_feat))
    inputC = layers.Input(shape=(None,n_feat))
    
    
    lstm_A = layers.GRU(1, dropout=param['drop'])(inputA)
                         #recurrent_dropout=param['recdrop'])(inputA)
    lstm_B = layers.GRU(1, dropout=param['drop'])(inputB)
                         #recurrent_dropout=param['recdrop'])(inputB)
    lstm_C = layers.GRU(1, dropout=param['drop'])(inputC)
                         #recurrent_dropout=param['recdrop'])(inputC)
    
    out_A = keras.Model(inputs=inputA, outputs=lstm_A)
    out_B = keras.Model(inputs=inputB, outputs=lstm_B)
    out_C = keras.Model(inputs=inputC, outputs=lstm_C)
    
   
    #t_stack = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([out_A.output, out_B.output, out_C.output])
    t_concat = tf.keras.layers.concatenate([out_A.output, out_B.output, out_C.output], axis=-1)
    #combined = stack([out_A.output, out_B.output, out_C.output],axis=1)
    #combined = tf.keras.layers.Reshape((n_feat, n_inst))(t_stack)
    
    #init_1 = tf.keras.initializers.HeNormal()
    init_1 = tf.keras.initializers.GlorotNormal()
    
    x1 = layers.Dense(3, activation=param['fc_act'],
                         kernel_initializer=init_1,
                         bias_initializer=initializers.Zeros())(t_concat)
    #x2 = tf.keras.layers.Reshape((1, n_feat))(x1)
    
    
    init_out = tf.keras.initializers.GlorotNormal()
    #model_output = layers.Dense(1, activation= custom_activation, kernel_initializer= init_out)(x1)
    
    fc_out = layers.Dense(1, activation= param['fc_act'], kernel_initializer= init_out)(x1)
 
    model_output = layers.Dense(1, activation= 'sigmoid', kernel_initializer= init_out)(fc_out)
    
    model = keras.Model(inputs=[out_A.input, out_B.input, out_C.input], outputs=model_output)
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    #print('hey')
    return model

def get_model_LSTM1(param):    
    #explicação das dimensões: 
        #https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
    
    #custom activation function (sigmoid between -1 and 1)
        
    n_inst = param['n_inst']
    n_feat = param['n_feat']
    
    inputA = layers.Input(shape=(None,n_feat)) 
    
    lstm_A = layers.GRU(n_feat, dropout=param['drop'])(inputA)
    
    
    init_out = tf.keras.initializers.GlorotNormal()
    if param['fc_act']== 'sigmoid':
        init_1 = tf.keras.initializers.GlorotNormal()
    else:
        init_1 = tf.keras.initializers.HeNormal()
    
    x1 = layers.Dense(int(n_feat/3), activation=param['fc_act'],
                         kernel_initializer=init_1,
                         bias_initializer=initializers.Zeros())(lstm_A)
    
    #model_output = layers.Dense(1, activation= custom_activation, kernel_initializer= init_out)(x1)
    
    #fc_out1 = layers.Dense(12, activation= param['fc_act'], kernel_initializer= init_1)(x1)
    
    fc_out2 = layers.Dense(3, activation= param['fc_act'], kernel_initializer= init_1)(x1)
 
    model_output = layers.Dense(1, activation= 'sigmoid', kernel_initializer= init_out)(fc_out2)
    
    model = keras.Model(inputs=inputA, outputs=model_output)
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile( loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    model.summary()
    #print('hey')
    return model


def get_model_CNN_old(param):
    model = Sequential()
    
    # Add a 3D Convolution layer with 32 filters and kernel size (3,3,3)
    model.add(Conv3D(8, kernel_size=3, input_shape=(12, 12, 5, 1)))#param['shape']))
    
    
    model.add(Conv3D(4, kernel_size=2))
    
    # Add a 3D Max Pooling layer with pool size (2,2,2)
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    
    # Flatten the output to pass it to a dense layer
    model.add(Flatten())
    
    init_out = tf.keras.initializers.GlorotNormal()
    model.add(Dense(8, activation='sigmoid', kernel_initializer= init_out))
    
    model.add(Dense(6, activation='sigmoid', kernel_initializer= init_out))
    
    # Add a dense layer with 1 neuron and a sigmoid activation
    model.add(Dense(1, activation='sigmoid', kernel_initializer= init_out))
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = 'mse', optimizer=param['opt'], metrics=[metric])
    
    return model

def get_model_CNN2_light(param):
    model = Sequential()
    
    # Add a 3D Convolution layer with 32 filters and kernel size (3,3,3)
    model.add(Conv3D(6, kernel_size=3, input_shape=(12, 12, 5, 1)))#param['shape']))
    
    # Add a 3D Max Pooling layer with pool size (2,2,2)
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    
    model.add(Dropout(param['drop']))
    # Flatten the output to pass it to a dense layer
    model.add(Flatten())
    
    init_dense = tf.keras.initializers.HeNormal()
    dense_activation = param['fc_act']
    
    model.add(Dense(8, activation=dense_activation, kernel_initializer= init_dense))
    
    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))
    
    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))
    
    init_out = tf.keras.initializers.GlorotNormal()
    # Add a dense layer with 1 neuron and a sigmoid activation
    model.add(Dense(1, activation='sigmoid', kernel_initializer= init_out))
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    
    return model

def get_model_CNN2(param):
    model = Sequential()
    
    # Add a 3D Convolution layer with 32 filters and kernel size (3,3,3)
    model.add(Conv3D(6, kernel_size=3, input_shape=(12, 12, 5, 1)))#param['shape']))
    model.add(Conv3D(3, kernel_size=3))
    
    # Add a 3D Max Pooling layer with pool size (2,2,2)
    model.add(MaxPooling3D(pool_size=(3, 3, 1)))
    
    model.add(Dropout(param['drop']))
    
    # Flatten the output to pass it to a dense layer
    model.add(Flatten())
    
    
    init_dense = tf.keras.initializers.HeNormal()
    dense_activation = 'sigmoid'
    
    model.add(Dense(12, activation=dense_activation, kernel_initializer= init_dense))
    
    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))
    
    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))
    
    init_out = tf.keras.initializers.GlorotNormal()
    # Add a dense layer with 1 neuron and a sigmoid activation
    model.add(Dense(1, activation='sigmoid', kernel_initializer= init_out))
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    
    return model

def get_model_CNN3(param):
    model = Sequential()

    # Add a 3D Convolution layer with 32 filters and kernel size (3,3,3)
    model.add(Conv2D(6, kernel_size=3, input_shape=(5, 12, 12)))#param['shape']))

    # Add a 3D Max Pooling layer with pool size (2,2,2)
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.2))

    # Flatten the output to pass it to a dense layer
    model.add(Flatten())

    init_dense = tf.keras.initializers.HeNormal()
    dense_activation = 'sigmoid'

    model.add(Dense(8, activation=dense_activation, kernel_initializer= init_dense))

    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))

    model.add(Dense(6, activation=dense_activation, kernel_initializer= init_dense))

    init_out = tf.keras.initializers.GlorotNormal()
    # Add a dense layer with 1 neuron and a sigmoid activation
    model.add(Dense(1, activation='sigmoid', kernel_initializer= init_out))

    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = "binary_crossentropy", optimizer=param['opt'], metrics=[metric])
    return model

def get_model_fc(param):
    #custom activation function (sigmoid between -1 and 1)
        
    n_sections = param['n_sections']
    n_inst = param['n_inst']
    n_feat = param['n_feat']
    
    input_size = n_inst*n_sections*n_feat
    
    init_1 = tf.keras.initializers.HeNormal()
    #init_1 = tf.keras.initializers.GlorotNormal()
    init_out = tf.keras.initializers.GlorotNormal()
    
    inputA = layers.Input(shape=(input_size)) 
    
    x1 = layers.Dense(input_size, activation=param['fc_act'], kernel_initializer=init_1)(inputA)
    x_drop = layers.Dropout(param['drop'])(x1)
    x2 = layers.Dense(input_size, activation=param['fc_act'], kernel_initializer=init_1)(x_drop)   
    x3 = layers.Dense(input_size, activation=param['fc_act'], kernel_initializer=init_1)(x2) 
    model_output = layers.Dense(1, activation= 'sigmoid', kernel_initializer= init_out)(x3)
    
    model = keras.Model(inputs=inputA, outputs=model_output)
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = 'mse', optimizer=param['opt'], metrics=[metric])
    
    return model

def get_model_fc_seq(param):
    #custom activation function (sigmoid between -1 and 1)
        
    n_sections = param['n_sections']
    n_inst = param['n_inst']
    n_feat = param['n_feat']
    
    input_size = n_inst*n_sections*n_feat
    
    init_1 = tf.keras.initializers.HeNormal()
    #init_1 = tf.keras.initializers.GlorotNormal()
    init_out = tf.keras.initializers.GlorotNormal()
    
    model = keras.Sequential(
    [
        layers.Dense(input_size, activation="relu", kernel_initializer= init_1, name="layer1"),
        layers.Dropout(param['drop'], name='layerdrop'),
        layers.Dense(input_size, activation="relu", kernel_initializer= init_1, name="layer2"),
        layers.Dense(input_size, activation="relu", kernel_initializer= init_1, name="layer3"),
        layers.Dense(1, activation= 'sigmoid', kernel_initializer= init_out, name="layerout"),
    ]
    )
    
    metric= tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss = 'mse', optimizer=param['opt'], metrics=[metric])
    
    return model

#when to use standartization vs normalization
#standartization assumes your data has a gaussian distribution, and normalization doesnt
#so, in theory, normalization would be more suitable for this case
#https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff

def Norm_and_Scale(data):
    # create scaler
    scaler_norm = MinMaxScaler()
    
    # fit scaler on data
    scaler_norm.fit(data)
    
    # apply transform
    normalized = scaler_norm.transform(data)
    
    #STANDARTIZE!!! way better
    sc = StandardScaler()
    sc.fit(data)
    std = sc.transform(data)
# =============================================================================
#     # create scaler
#     scaler = StandardScaler()
#     
#     # fit scaler on data
#     scaler.fit(normalized)
#     # apply transform
#     standardized = scaler.transform(normalized)
#     # inverse transform
#     inverse = scaler.inverse_transform(standardized)
# =============================================================================
    
    #return normalized, scaler_norm
    return std, sc

def Norm_and_Scale(data):
    data=np.array(data)
    
    '''
    from sklearn.preprocessing import MinMaxScaler
    # create scaler
    scaler_norm = MinMaxScaler()
    
    # fit scaler on data
    scaler_norm.fit(data)
    
    # apply transform
    normalized = scaler_norm.transform(data)
    '''
    #STANDARTIZE!!! way better
    sc = StandardScaler()
    sc.fit(data)
    std = sc.transform(data)
# =============================================================================
#     # create scaler
#     scaler = StandardScaler()
#     
#     # fit scaler on data
#     scaler.fit(normalized)
#     # apply transform
#     standardized = scaler.transform(normalized)
#     # inverse transform
#     inverse = scaler.inverse_transform(standardized)
# =============================================================================
    
    #return normalized, scaler_norm
    return std, sc

def Norm_and_Scale_CNN1(data):
    
    scalers=[]
    
    for i in range(data.shape[0]):
        # Extract the current frequency band
        frequency_band = data[i, :, :]
    
        # Initialize the StandardScaler object for the current frequency band
        scaler = StandardScaler()
    
        # Fit the scaler to the frequency band
        scaler.fit(frequency_band)
    
        # Transform all images in the frequency band using the same scaler
        frequency_band_scaled = scaler.transform(frequency_band)
    
        # Store the transformed frequency band back into the video data
        data[i, :, :] = frequency_band_scaled
    
        # Store the scaler for this frequency band
        scalers.append(scaler)
    return data, scalers

def Norm_and_Scale_CNN(data):
    
    scalers=[]
    
    for i in range(data.shape[1]):
        # Extract the current frequency band
        frequency_band = data[:, i, :].T
    
        # Initialize the StandardScaler object for the current frequency band
        scaler = StandardScaler()
    
        # Fit the scaler to the frequency band
        scaler.fit(frequency_band)
    
        # Transform all images in the frequency band using the same scaler
        frequency_band_scaled = scaler.transform(frequency_band)
    
        # Store the transformed frequency band back into the video data
        data[:, i, :] = frequency_band_scaled.T
    
        # Store the scaler for this frequency band
        scalers.append(scaler)
    return data, scalers

def Norm_and_Scale_CNN_flat(data):
    
    n_examples, mfccs, timesteps = np.shape(data)
    data=np.transpose(data,(1,2,0))
    #data=np.reshape(data, (mfccs, timesteps*n_examples))
    
    scalers=[]
    
    for i in range(data.shape[0]):
        # Extract the current frequency band
        frequency_band = data[i].T
    
        # Initialize the StandardScaler object for the current frequency band
        scaler = StandardScaler()
    
        # Fit the scaler to the frequency band
        scaler.fit(frequency_band)
    
        # Transform all images in the frequency band using the same scaler
        frequency_band_scaled = scaler.transform(frequency_band)
    
        # Store the transformed frequency band back into the video data
        data[i] = frequency_band_scaled.T
    
        # Store the scaler for this frequency band
        scalers.append(scaler)
        
    data =np.transpose(data,(2,0,1)) #transpose back to original shape
    return data, scalers
    
def Analyse_Model(history, metric, n_epochs):

    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history[metric][:n_epochs])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('train '+metric )
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'][:n_epochs])
    plt.plot(history.history['val_loss'][:n_epochs])
    #plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
    return

# smote doesnt create more datapoints than needed to balance data.
# to create more points we should duplicate points in one class to force 
# smote to work, then take those duplicates out. the number of duplicates should be :
# d: desired number of data points
# n: number of times we duplicate data from a class
# x: number of points for that class
#
# d= 3*n*x (for 3 classes) ---> n=d/(3*x)
#
def SMOTE_oversample(X_train, y_train):
    #imbalanced data
    oversample = SMOTE(sampling_strategy='all', k_neighbors=3)
    
    X=[]
    
    for inst in X_train:
        sections = np.reshape(inst, (np.shape(X_train[0])[1], len(y_train), np.shape(X_train[0])[2]))
        SEC_reshape = []
        #y_smote = []
        
        for sec in sections:

            sec, y = oversample.fit_resample(sec, y_train)
            SEC_reshape.append(sec)
            #y_smote.append(y) 
            
        sections = np.reshape(np.array(SEC_reshape), (len(y), np.shape(X_train[0])[1], np.shape(X_train[0])[2]))
        X.append(sections)

    return X, y

def apply_mix_up(x, y, n):
    #n is the number of desired datapoints
    #n_per_class is the number of datapoints per class
    n_per_class = int(n/3)
    
    x_new = x.copy()
    y_new = y.copy()
    
    it=1
    
    alpha=0.2
    #lambda (for every sample):
    l_list = sample_beta_distribution(n, alpha, alpha).numpy()
    
    #change points that are 1 (it would replicate one of the samples) 
    where_1 = np.where(l_list==1)
    l_list[where_1] = 0.99  
    l_list=list(l_list)
    
    for sample, label in zip(x, y):
        next_sample = x[it]
        next_label = y[it]
        
        #NEXT STEP: ESCOLHER IT RANDOMLY (SEM REPETIR PONTOS)
        it+=1
        
        new_sample, new_label = mix_up(sample, next_sample, label, next_label, l_list.pop(1))
        
        x_new = np.append(x_new, new_sample)
        y_new = np.append(y_new, new_label)
        
        if len(y_new) >= n:
            return x_new, y_new
    
    return x_new, y_new

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(sample, next_sample, l):
    
    new_sample = l*sample + (1-l)*next_sample
    
    return new_sample

def MIXUP_oversample(X_train, y_train ,n):
    
    X=[]
    
    #number of datapoints to be created
    n_new = max(0,n-len(y_train))
       
    #idx of datapoints
    idx_x = list(range(len(y_train)))
    comb = list(it.combinations(idx_x, 2))
    pick_comb = random.sample(comb, n_new)
    
    alpha=0.2
    #lambda (for every new sample):
    l_list = sample_beta_distribution(n_new, alpha, alpha).numpy()
    #change points that are 1 (it would replicate one of the samples) 
    where_1 = np.where(l_list==1)
    l_list[where_1] = 0.99  
    l_list=list(l_list)
    
    X_arr = np.array(X_train)
    X_arr = np.reshape(X_arr, (np.shape(X_arr)[1], np.shape(X_arr)[0],
                               np.shape(X_arr)[2], np.shape(X_arr)[3]))
    
    y=y_train.copy()
    
    for idx1, idx2 in pick_comb:
        sample1 = X_arr[idx1]
        sample2 = X_arr[idx2]
        
        label1 = y_train[idx1]
        label2 = y_train[idx2]
        
        #lambda for this instance
        l=l_list.pop()
        
        new_label = mix_up(label1, label2, l)
        y.append(new_label)
        
        inst_vec = list()
        for inst1, inst2 in zip(sample1, sample2):
            sec_vec = list()
            for sec1, sec2 in zip(inst1, inst2):
                new_sec = mix_up(sec1, sec2, l)
                
                sec_vec.append(new_sec)
                
            inst_vec.append(sec_vec)
        
        new_sample = np.array(inst_vec)        
        X_arr = np.concatenate((X_arr, [new_sample]))
        
    X = np.reshape(X_arr, (np.shape(X_arr)[1], np.shape(X_arr)[0],
                               np.shape(X_arr)[2], np.shape(X_arr)[3]))
    
    return X, np.array(y)


def brute_oversample(X, y):
    X=np.array(X)
    
    
    unique, counts = np.unique(y, return_counts=True)
    
    # Find the minority class and majority class
    minority_class = unique[counts == np.min(counts)][0]
    majority_class = unique[counts == np.max(counts)][0]
    
    # Get the indices of the minority and majority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Calculate the number of samples needed for the minority class to match the majority class
    number_of_samples = len(majority_indices) - len(minority_indices)
    
    # Duplicate the samples from the minority class until we reach the desired number of samples
    random_indices = np.random.choice(minority_indices, size=number_of_samples, replace=True)
    X_minority_over = X[random_indices]
    X_new = np.concatenate((X, X_minority_over), axis=0)
    y_new = np.concatenate((y, np.full(number_of_samples, minority_class)), axis=0)
    
    return X_new, y_new

def brute_oversample_old(X_train, y_train):
    idx_pos = np.where(np.array(y_train)==1)[0]
    idx_neg = np.where(np.array(y_train)==0)[0]
    
    if len(idx_pos)< len(idx_neg):
        under_class = 1
        idx_under = idx_pos
        idx_over = idx_neg
    else:
        under_class = 0
        idx_under = idx_neg
        idx_over = idx_neg
        
    X=[]
    
    for inst in X_train:
        y=y_train
        X_under = inst[idx_under]
        while len(idx_over)/len(inst)>0.5:
            inst = np.concatenate((inst, X_under))
        X.append(inst)
        y = np.append(y, np.ones(len(X[0])-len(y_train))*under_class)
    
    print('hey')
    return X, y

'''
def oversample_combinations(Pop, scores , n):
    
    for ind, score in zip(Pop, score):
        if score==1:
            
        
    
    return Pop, scores
'''

def Pad_zeros(Pop, Scores, features):
    max_len = max([arr.shape[1] for arr in Pop])
    add_feat = np.zeros(np.shape(features[0]), dtype=float)
    features.append([np.squeeze(add_feat)])
    idx_last_feat = len(features)-1
    
    pad_section = np.array([idx_last_feat, idx_last_feat, idx_last_feat]).T
    
    new_Pop=[]
    new_Scores=[]
    
    for ind, score in zip(Pop, Scores):
        n_sec=ind.shape[1]
        if n_sec<max_len:
            n_add = max_len-n_sec
            
            for i in range(n_add):
               ind = np.concatenate((ind,np.expand_dims(pad_section, axis=1)), axis=1)
            new_Pop.append(ind)
            new_Scores.append(score)
            
    return new_Pop, new_Scores, features
