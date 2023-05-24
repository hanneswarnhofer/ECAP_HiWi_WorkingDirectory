# Author:
# Hannes Warnhofer
# hannes.warnhofer@fau.de

# ECAP HiWi Project 
# Alison Mitchell
# Samuel Spencer

# CNN image classification for 4 images with late fusion

# import relevant (and irrelevant) packages 
import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model, Sequential
import glob
import pickle
import sys
import argparse


# Specifying test data directory and filepath

var1 = "9"
dirPath = "../Test_Data/"

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
#fnr = "_2023-02-27_" + var1 +'_only_earlystopping'

# Create file paths from the argument that specifies which data files to include
crit = '[' + var1 + ']*.hdf5'
file_paths = [f for f in glob.glob(os.path.join(dirPath, crit))]

squared_training = []
peak_times = []
event_labels = []


for i in file_paths:
    with tables.open_file(i, mode="r") as x:
        # ignore the electron events
        mask_e = x.root.event_label[:] != 2 
        event_labels_temp = x.root.event_label[:][mask_e].reshape((-1, 1))
        squared_training_temp = x.root.squared_training[:,:,:,:][mask_e]
        peak_times_temp = x.root.peak_times[:,:,:,:][mask_e]

        # append all the data to a common array
        squared_training.append(squared_training_temp)
        peak_times.append(peak_times_temp)
        event_labels.append(event_labels_temp)
    
    

squared_training = np.concatenate(squared_training, axis=0)
peak_times = np.concatenate(peak_times, axis=0)
event_labels = np.concatenate(event_labels, axis=0)

# some reshaping for the further use of the timing data in the CNN
peak_times = peak_times.reshape((*np.shape(peak_times),1))

# overview about the important data array for later usage
print(np.shape(peak_times)[0], " events with 4 images each are available \n")
print("Shape of 'event_labels': ",np.shape(event_labels))
print("Shape of 'squared_training': ",np.shape(squared_training))
print("Shape of 'peak_times': ",np.shape(peak_times),"\n")

# split into random training data (80%) and test data (20%)
train_data, test_data, train_labels, test_labels = [], [], [], []
random_selection = np.random.rand(np.shape(peak_times)[0]) <= 0.8
train_data.append(peak_times[random_selection])
test_data.append(peak_times[~random_selection])
train_labels.append(event_labels[random_selection])
test_labels.append(event_labels[~random_selection])

# free some memory space
del peak_times
del event_labels

# convert to numpy array and reshape 
train_data = np.array(train_data)
train_data = train_data.reshape(np.shape(train_data[0]))
test_data = np.array(test_data)
test_data = test_data.reshape(np.shape(test_data[0]))

train_labels = np.array(train_labels)
train_labels = train_labels.reshape(np.shape(train_labels[0]))
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(np.shape(test_labels[0]))

train_labels_multishape = np.zeros_like(train_data)
test_labels_multishape = np.zeros_like(test_data)

len_train = np.shape(train_data)[0]
len_test = np.shape(test_data)[0]

for i in range(0,len_train):
    train_labels_multishape[i,:,:,:] = train_labels[i]

for k in range(0,len_test):
    test_labels_multishape[k,:,:,:] = test_labels[k]

# overvew about the splitting into training and test data
print("Split into Training and Test Data")
print("Train data shape:", np.shape(train_data) , "-->",round(100*len_train/(len_train+len_test),2),"%")
print("Test data shape:", np.shape(test_data), "-->",round(100*len_test/(len_train+len_test),2), "%")
print("Train labels shape:", np.shape(train_labels))
print("Test labels shape:", np.shape(test_labels))

## Testtest

print("Test")