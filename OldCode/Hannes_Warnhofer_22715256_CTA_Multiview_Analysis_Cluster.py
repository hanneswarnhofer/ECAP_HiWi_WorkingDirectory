# Author:
# Hannes Warnhofer
# 22715256
# hannes.warnhofer@fau.de

# Test if GitHub Connection works as intended
lll = 1
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

# Allow for arguments to be passed to the python file, when running it. This enables to define a series of runs with different model properties, by defining this in the respective batch script.
# Example for usage: srun python /home/hpc/b129dc/b129dc26/MoDA_Project/Hannes_Warnhofer_22715256_CTA_Multiview_Analysis_Cluster.py -n "345" -e 50 -b 512 -r 0.25 -reg 0.001
# This uses all the data that starts with 3,4 or 5, runs over 50 epochs, has a batch size of 512 and a dropout rate of .25 and a regularization of 0.001
# The following installations have to be done when running a batch script on Alex:
# module load python/tensorflow-2.7.0py3.9
# pip install --user tables
# pip install --user pandas
# pip install --user pickle
# pip install --user glob
# pip install --user sys
# pip install --user argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--numbers", type=str)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-r", "--rate", type=float)
parser.add_argument("-reg", "--regulization", type=float)

args = parser.parse_args()
var1 = args.numbers
num_epochs = args.epochs
batch_size = args.batch_size
rate = args.rate
reg = args.regulization
patience = 5

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "_2023-02-27_" + var1 +'_only_earlystopping'

# Create file paths from the argument that specifies which data files to include
crit = '[' + var1 + ']*.hdf5'
file_paths = [f for f in glob.glob(os.path.join(cluster_filePath, crit))]


# load in the data
####################################

# define empty arrays
squared_training = []
peak_times = []
event_labels = []

# go through the file_paths of the includes files
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

# split up different "telescopes" for the usage in the seperate single view CNNs (probably in the most long-winded way possible, but lets just ignore that)
train_data_1 = train_data[:,0,:,:] 
train_data_2 = train_data[:,1,:,:] 
train_data_3 = train_data[:,2,:,:] 
train_data_4 = train_data[:,3,:,:] 

test_data_1 = test_data[:,0,:,:]
test_data_2 = test_data[:,1,:,:]
test_data_3 = test_data[:,2,:,:]
test_data_4 = test_data[:,3,:,:]

train_labels_1 = train_labels_multishape[:,0,:,:]
train_labels_2 = train_labels_multishape[:,1,:,:]
train_labels_3 = train_labels_multishape[:,2,:,:]
train_labels_4 = train_labels_multishape[:,3,:,:]

test_labels_1 = test_labels_multishape[:,0,:,:]
test_labels_2 = test_labels_multishape[:,1,:,:]
test_labels_3 = test_labels_multishape[:,2,:,:]
test_labels_4 = test_labels_multishape[:,3,:,:]

print("Train data 1 shape:", np.shape(train_data_1))
print("Train labels 1 shape:", np.shape(train_labels_1))

print("Test data 1 shape:", np.shape(test_data_1))
print("Test labels 1 shape:", np.shape(test_labels_1))

input_shape = (48, 48, 1)
pool_size = 2
kernel_size = 4

# Define the model for the single-view CNNs
def create_cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=25, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=100, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    return model

# Define the model for the combination of the previous CNNs and the final CNN for classification

def run_multiview_model(models,inputs):

    merged = concatenate(models)

    Dropout1 = Dropout(rate)(merged)
    Conv_merged1 = Conv2D(filters=25,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout1)
    MaxPool_merged1 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged1)

    Dropout2 = Dropout(rate)(MaxPool_merged1)
    Conv_merged2 = Conv2D(filters=50,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout2)
    MaxPool_merged2 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged2)

    Dropout3 = Dropout(rate)(MaxPool_merged2)
    Conv_merged3 = Conv2D(filters=100,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout3)
    MaxPool_merged3 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged3)

    Flat_merged1 = Flatten()(MaxPool_merged3)
    Dropout4 = Dropout(rate)(Flat_merged1)
    dense_layer_merged1 = Dense(units=100, activation='relu')(Dropout4)

    Dropout5 = Dropout(rate)(dense_layer_merged1)
    dense_layer_merged2 = Dense(units=50, activation='relu')(Dropout5)

    Dropout6 = Dropout(rate)(dense_layer_merged2)
    dense_layer_merged3 = Dense(units=1, activation='sigmoid')(Dropout6)

    model = Model(inputs=inputs, outputs=dense_layer_merged3)
    return model

# Create four separate CNN models
input_1 = Input(shape=input_shape)
cnn_model_1 = create_cnn_model(input_shape)(input_1)

input_2 = Input(shape=input_shape)
cnn_model_2 = create_cnn_model(input_shape)(input_2)

input_3 = Input(shape=input_shape)
cnn_model_3 = create_cnn_model(input_shape)(input_3)

input_4 = Input(shape=input_shape)
cnn_model_4 = create_cnn_model(input_shape)(input_4)

# include early_stopping here, to see how it changes compared to previous model designs
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

model_multi = run_multiview_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
model_multi.summary()
model_multi.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 

history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping])

# Create the filename, which is used for saving the Accuracy and Loss plots and the history files
str_num_epochs = '{}'.format(num_epochs)
str_batch_size = '{}'.format(batch_size)
str_rate = '{}'.format(rate*100)
str_reg = '{}'.format(reg)

history_name = "history_" + str_num_epochs + "epochs" + str_batch_size + "batchsize" + str_rate + "rate" + str_reg + "reg" + fnr + ".pkl"

# Save the history files for later usage in other scripts
with open(history_name, 'wb') as file:
    pickle.dump(history.history, file)

# Create plots for quick overview
fig, ax = plt.subplots(1,2, figsize = (9,3))
ax[0].plot(history.history['accuracy'], label='Testing Data',lw=2,c="darkorange")
ax[0].plot(history.history['val_accuracy'], label = 'Validation Data',lw=2,c="firebrick")
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_ylim([0.5, 1])
ax[0].legend(loc='lower right')

ax[1].plot(history.history['loss'],lw=2,c="darkorange")
ax[1].plot(history.history['val_loss'],lw=2,c="firebrick")
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')

print("Image created")

filename_savefig = "Test_Cluster_"+ str_num_epochs + "epochs" + str_batch_size + "batchsize" + str_rate + "rate" + fnr +".png"
fig.savefig(filename_savefig, bbox_inches='tight')

print("Image saved")

 