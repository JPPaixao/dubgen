"""
Created on Mon May 23 17:45:05 2022

@author: JP

@brief: Functions that contribute to NN model. Creates Input Feature Matrices. Balances and Standartizes Data
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

#Creates Input Feature Matrices for CNN. Performs standartization and oversampling
def Training_data_for_GA_CNN(Pop, features, Scores, test, scalers, pca=0):
    feat_data = features
    
    array_data = np.array(feat_data)
    array_data = np.squeeze(array_data)
    
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

    X = list()
    for i in range(len(Pop)):
        mfcc_seq = list()  #(5 sections, 12 time steps, 12 mfccs)
        for s in range(np.shape(Pop[i])[1]):
            mfcc_panel = np.array([]) #(12,12) frames for each instrument concatenated (3inst*4timesteps, 12 coefs)
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
            
        X.append(mfcc_array)

    y = Scores
    y = transform_score(np.array(y))            
                
    if test!=1:
        X, y = brute_oversample(X, y)
        X=list(X)
    
    return X, np.array(y), scalers, pca


#forces original score to range [0,1]
def transform_score(x):
   return (0.5*x+0.5)


# applies standard scaler to data
def Norm_and_Scale(data):
    data=np.array(data)
    
    sc = StandardScaler()
    sc.fit(data)
    std = sc.transform(data)

    return std, sc


# applies standard scaler to CNN data
def Norm_and_Scale_CNN_flat(data):
    n_examples, mfccs, timesteps = np.shape(data)
    data=np.transpose(data,(1,2,0))
    scalers=[]
    for i in range(data.shape[0]):
        # Extract the current frequency band
        frequency_band = data[i].T
        # Initialize the StandardScaler object for the current frequency band
        scaler = StandardScaler()
        # Fit the scaler to the frequency band
        scaler.fit(frequency_band)
        # Transform all frames in the frequency band using the same scaler
        frequency_band_scaled = scaler.transform(frequency_band)
        # Store the transformed frequency
        data[i] = frequency_band_scaled.T
        # Store the scaler for this frequency band
        scalers.append(scaler)
        
    data =np.transpose(data,(2,0,1)) #transpose back to original shape
    return data, scalers
    


# balances input data by duplicating data point in the minority class
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

