# Processing HESS data for usage in CNN code from MoDA project
# Author: Hannes Warnhofer
# hannes.warnhofer@fau.de

import tables
import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import random

import fnmatch
import os
#import h5
import glob
import pickle
import sys
import argparse
import h5py
import os.path
import inspect
import json

from ctapipe.io import EventSource
from ctapipe import utils
from ctapipe.instrument.camera import CameraGeometry

from dl1_data_handler.reader import DL1DataReader
from dl1_data_handler.image_mapper import ImageMapper

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model, Sequential


class DataManager():
    """ Data class used to manage the HDF5 data files (simulations + Auger data).
        data_path: data_path of HDF5 file, (hint: use blosc compression to ensure adequate decompression speed,
        to mitigate training bottleneck due to a slow data pipeline)
        params:
            data_path = path to HDF5 datset
        optional params:
            stats: data statistics (stats.json - needed for scaling the dataset)
            tasks: list of tasks to be included (default: ['axis', 'core', 'energy', 'xmax'])
            generator_fn: generator function used for looping over data, generator function needs to have indices and
                          shuffle args.
            ad_map_fn: "advanced mapping function" the function used to map the final dataset. Here an additional
                       preprocessing can be implemented which is mapped during training on the
                       cpu (based on tf.data.experimental.map_and_batch)
    """

    def __init__(self, data_path, stats=None, tasks=['axis', 'impact', 'energy', 'classification']):
        ''' init of DataManager class, to manage simulated (CORSIKA/Offline) and measured dataset '''
        np.random.seed(1)
        self.data_path = data_path

    def open_ipython(self):
        from IPython import embed
        embed()

    @property
    def is_data(self):
        return self.type == "Data"

    @property
    def is_mc(self):
        return self.type == "MC"

    def get_h5_file(self):
        return h5py.File(self.data_path, "r")

    def walk_tree(self, details=True):
        """ Draw the tree of yout HDF5 file to see the hierachy of your dataset
            params: detail(activate details to see shapes and used compression ops, Default: True)
        """

        def walk(file, iter_str=''):
            try:
                keys = file.keys()
            except AttributeError:
                keys = []

            for key in keys:
                try:
                    if details:
                        file[key].dtype
                        print(iter_str + str(file[key]))
                    else:
                        print(iter_str + key)
                except AttributeError:
                    print(iter_str + key)
                    walk(file[key], "   " + iter_str)

        with h5py.File(self.data_path, "r") as file:
            print("filename:", file.filename)
            for key in file.keys():
                print(' - ' + key)
                walk(file[key], iter_str='   - ')

    def extract_info(self, path):
        with self.get_h5_file() as f:
            data = f[path]
            y = np.stack(data[:].tolist())

        return {k: y[:, i] for i, k in enumerate(data.dtype.names)}, dict(data.dtype.descr)

    def make_mc_data(self):
        return self.extract_info("simulation/event/subarray/shower")


class MyGenerator(keras.utils.Sequence):

    def __init__(self,images_1,images_2,images_3,images_4,labels,batch_size=64):
        self.batch_size = batch_size
        self.images_1 = images_1
        self.images_2 = images_2
        self.images_3 = images_3
        self.images_4 = images_4
        self.labels = labels
        self.sample_count = len(labels[:])
        self.batch_count = int(self.sample_count/batch_size)
        self.current_batch = 0
        self.index = 0

    def __len__(self):
        return self.batch_count
    
    def __getitem__(self,index):
        
        X = [self.images_1[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size,:,:,:],self.images_2[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size,:,:,:],self.images_3[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size,:,:,:],self.images_4[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size,:,:,:]]
        y = self.labels[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size,:]

        self.current_batch +=1 
        self.data = (X,y)

        return self.data
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.sample_count:
            raise StopIteration
        result = self.__getitem__(self.index) 
        self.index += 1
        return result
    
    
    def plot_batch(self, event_nr=100):
        image_batch = [np.array(self.images_1[event_nr]),
                       np.array(self.images_2[event_nr]),
                       np.array(self.images_3[event_nr]),
                       np.array(self.images_4[event_nr])]
        label_batch = np.array(self.labels[event_nr])
        plot_image_2by2(image=image_batch, labels=label_batch, event_nr=event_nr)

    def reset_counters(self): 
        self.current_batch = 0 

        
    def on_epoch_end(self):
        self.reset_counters()

class OnEpochBegin(keras.callbacks.Callback): # Callback class called on epoch begin to reset counters
    def on_epoch_begin(self, epoch, logs=None):
        training_generator.reset_counters()
        testing_generator.reset_counters()
        print("Epoch Begin")


def plot_image(image, name=None):
    fig, ax = plt.subplots(1)
    ax.set_aspect(1)
    ax.pcolor(np.flip(image[:, :, 0], axis=(0)), cmap='viridis', vmin=-5)
    plt.show()
    fig.savefig("./binned_image%s.png" % name)


def re_index_ct14(image):
    return image[5:, :, :]

def make_hess_geometry(file=None):
    # quick fix for dl1 data handler to circumvent to use ctapipe
    if file is None:
        with open(os.path.join(os.getcwd(), "geometry2d3.json")) as f: 
            attr_dict = json.load(f)

        data_ct14 = attr_dict["ct14_geo"]
        data_ct5 = attr_dict["ct5_geo"]
    else:
        data_ct14 = file["configuration/instrument/telescope/camera/geometry_0"][:].tolist()
        data_ct5 = file["configuration/instrument/telescope/camera/geometry_1"][:].tolist()

    class Geometry():
        def __init__(self, data):
            self.pix_id, self.pix_x, self.pix_y, self.pix_area = np.stack(data).T.astype(np.float32)
            self.pos_x = self.pix_x
            self.pos_y = self.pix_y

        def get_pix_pos(self):
            return np.column_stack([self.pix_x, self.pix_y]).T

    return Geometry(data_ct14), Geometry(data_ct5)

def get_current_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(filename))


def rotate(pix_pos, rotation_angle=0):
    rotation_angle = rotation_angle * np.pi / 180.0
    rotation_matrix = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                 [np.sin(rotation_angle), np.cos(rotation_angle)], ], dtype=float)

    pixel_positions = np.squeeze(np.asarray(np.dot(rotation_matrix, pix_pos)))
    return pixel_positions

def plot_image_2by2(image,event_nr,labels):
    image1 = image[:,0,:,:][event_nr]
    image2 = image[:,1,:,:][event_nr]
    image3 = image[:,2,:,:][event_nr]
    image4 = image[:,3,:,:][event_nr]
    fig, ax = plt.subplots(2,2)
    ax[0,0].set_aspect(1)
    ax[0,0].pcolor(np.flip(image1[:,:,0],axis=(0)), cmap='viridis',vmin=-5)
    ax[0,1].set_aspect(1)
    ax[0,1].pcolor(np.flip(image2[:,:,0],axis=(0)), cmap='viridis',vmin=-5)
    ax[1,0].set_aspect(1)
    ax[1,0].pcolor(np.flip(image3[:,:,0],axis=(0)), cmap='viridis',vmin=-5)
    ax[1,1].set_aspect(1)
    ax[1,1].pcolor(np.flip(image4[:,:,0],axis=(0)), cmap='viridis',vmin=-5) 

    label1 = labels[:,0,:,:][event_nr].ravel()[0]
    label2 = labels[:,1,:,:][event_nr].ravel()[0]
    label3 = labels[:,2,:,:][event_nr].ravel()[0]
    label4 = labels[:,3,:,:][event_nr].ravel()[0]

    str_label1 = '{}'.format(label1)
    str_label2 = '{}'.format(label2)
    str_label3 = '{}'.format(label3)
    str_label4 = '{}'.format(label4)

    ax[0, 0].text(0.05, 0.95, str_label1, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[0, 1].text(0.05, 0.95, str_label2, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 0].text(0.05, 0.95, str_label3, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 1].text(0.05, 0.95, str_label4, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))

    str_evnr = '{}'.format(event_nr)
    name = "Test_figure_2023-07-21_evnr_" + str_evnr + ".png"
    fig.savefig(name)


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-r", "--rate", type=float)
parser.add_argument("-reg", "--regulization", type=float)

args = parser.parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
rate = args.rate
reg = args.regulization
#patience = 5

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "_2023-07-18_newSet" 
num_events = 100000

#filePath_gamma="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
#filePath_gamma = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
filePath_gamma = "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"

#filePath_proton="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
#filePath_proton = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
filePath_proton="../../../../wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"


dm_gamma = DataManager(filePath_gamma)
f_g = dm_gamma.get_h5_file()
#e = tables.open_file(path, mode="r")

tel1g_raw = f_g["dl1/event/telescope/images/tel_001"][:]
tel2g_raw = f_g["dl1/event/telescope/images/tel_002"][:]
tel3g_raw = f_g["dl1/event/telescope/images/tel_003"][:]
tel4g_raw = f_g["dl1/event/telescope/images/tel_004"][:]

#data_g = tables.open_file(filePath_gamma, mode="r")

print("Successfully opened gamma data!")
#print(data_g)


# Assigning telescope data to different arrays
#tel1g_raw = data_g.get_node('/dl1/event/telescope/images/tel_001').read()
#tel2g_raw = data_g.get_node('/dl1/event/telescope/images/tel_002').read()
#tel3g_raw = data_g.get_node('/dl1/event/telescope/images/tel_003').read()
#tel4g_raw = data_g.get_node('/dl1/event/telescope/images/tel_004').read()

# Reshaping arrays and extracting the data
#tel1g = np.stack([data[-1] for data in tel1g_raw])
#tel2g = np.stack([data[-1] for data in tel2g_raw])
#tel3g = np.stack([data[-1] for data in tel3g_raw])
#tel4g = np.stack([data[-1] for data in tel4g_raw])

labelsg = np.stack([data[2] for data in tel1g_raw])
labelsg_ones = np.ones_like(labelsg)

#del tel1g_raw
#del tel2g_raw
#del tel3g_raw
#del tel4g_raw

f_g.close()

dm_proton = DataManager(filePath_proton)
f_p = dm_proton.get_h5_file()
#e = tables.open_file(path, mode="r")

tel1p_raw = f_p["dl1/event/telescope/images/tel_001"][:]
tel2p_raw = f_p["dl1/event/telescope/images/tel_002"][:]
tel3p_raw = f_p["dl1/event/telescope/images/tel_003"][:]
tel4p_raw = f_p["dl1/event/telescope/images/tel_004"][:]

#data_p = tables.open_file(filePath_proton, mode="r")

print("Successfully opened proton data!")
#print(data_p)

# Assigning telescope data to different arrays
#tel1p_raw = data_p.get_node('/dl1/event/telescope/images/tel_001').read()
#tel2p_raw = data_p.get_node('/dl1/event/telescope/images/tel_002').read()
#tel3p_raw = data_p.get_node('/dl1/event/telescope/images/tel_003').read()
#tel4p_raw = data_p.get_node('/dl1/event/telescope/images/tel_004').read()

# Reshaping arrays and extracting the data
#tel1p = np.stack([data[-1] for data in tel1p_raw])
#tel2p = np.stack([data[-1] for data in tel2p_raw])
#tel3p = np.stack([data[-1] for data in tel3p_raw])
#tel4p = np.stack([data[-1] for data in tel4p_raw])

labelsp = np.stack([data[2] for data in tel1p_raw])
labelsp_zeros = np.zeros_like(labelsp)

tel1 = np.concatenate((tel1g_raw,tel1p_raw),axis=0)
tel2 = np.concatenate((tel2g_raw,tel2p_raw),axis=0)
tel3 = np.concatenate((tel3g_raw,tel3p_raw),axis=0)
tel4 = np.concatenate((tel4g_raw,tel4p_raw),axis=0)
labels = np.concatenate((labelsg_ones,labelsp_zeros),axis=0)

del tel1g_raw
del tel2g_raw
del tel3g_raw
del tel4g_raw

del tel1p_raw
del tel2p_raw
del tel3p_raw
del tel4p_raw

f_p.close()

#tel1 = np.concatenate((tel1g,tel1p),axis=0)
#tel2 = np.concatenate((tel2g,tel2p),axis=0)
#tel3 = np.concatenate((tel3g,tel3p),axis=0)
#tel4 = np.concatenate((tel4g,tel4p),axis=0)
#labels = np.concatenate((labelsg_ones,labelsp_zeros),axis=0)

#tel1 = np.vstack((tel1g, tel1p))
#tel2 = np.vstack((tel2g, tel2p))
#tel3 = np.vstack((tel3g, tel3p))
#tel4 = np.vstack((tel4g, tel4p))
#labels = np.vstack((labelsg_ones, labelsp_zeros))

#del tel1p
#del tel1g
#del tel2p
#del tel2g
#del tel3p
#del tel3g
#del tel4p
#del tel4g
del labelsp
del labelsg
del labelsp_zeros
del labelsg_ones

print(np.shape(tel1))
print(np.shape(tel2))
print(np.shape(tel3))
print(np.shape(tel4))
print(np.shape(labels))
print(labels)

geo_ct14, geo_ct5 = make_hess_geometry()
print(os.getcwd())
ct_14_mapper = ImageMapper(camera_types=["HESS-I"], pixel_positions={"HESS-I": rotate(geo_ct14.get_pix_pos())}, mapping_method={"HESS-I": "axial_addressing"})

#test_img_ct14 = x3[50][3][:,np.newaxis]

#image_ct14 = ct_14_mapper.map_image(test_img_ct14, "HESS-I")
#image_ct14 = re_index_ct14(image_ct14)
#plot_image(image_ct14, name="ct14")

#num_events = 100 #len(labels) # Takes very long with many events on my PC, for testing: num_events = 10000 (len(test_pixel_values)=106319)

'''
# Define the camera types and mapping methods: HESS-I only

hex_cams = ['HESS-I']
camera_types = hex_cams 
#hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
#               'bilinear_interpolation', 'bicubic_interpolation', 
#               'image_shifting', 'axial_addressing']
hex_methods = ['axial_addressing']
#Load the image mappers
mappers = {}
print("Start Initializing Mappers...")
print(os.system("pwd")) 
current_directory = os.getcwd()
print(current_directory)
#raise KeyboardInterrupt 
print("Initialization time (total for all telescopes):")
for method in hex_methods:
    print(method)
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method,camera_types=["HESS-I"])
print("... Finished Initializing Mappers")
# Reshape arrays for mapping
# Defining how many events should be mapped and used later on
#num_events = 1000 #len(labels) # Takes very long with many events on my PC, for testing: num_events = 10000 (len(test_pixel_values)=106319)

# Defining image shape and mapper type
default_mapper = ImageMapper(camera_types=['HESS-I'])
#padding_mapper = ImageMapper(padding={cam: 10 for cam in camera_types}, camera_types=["HESS-I"])
#image_shape = default_mapper.map_image(tel1[0], 'HESS-I').shape
'''
# Creating empty arrays for mapped images and the associated labels
mapped_images_1 = np.empty((num_events, 41,41,1))
mapped_images_2 = np.empty((num_events, 41,41,1))
mapped_images_3 = np.empty((num_events, 41,41,1))
mapped_images_4 = np.empty((num_events, 41,41,1))
mapped_labels = np.empty(num_events)

# Using the map_image function for mapping the data from the different telescopes to the associated empty array
# Drawing radom num_events events from all the data 
length = num_events
max_value = len(tel1)
#random_list = random.sample(range(max_value),length) 
random_list = np.random.randint(max_value, size=length)
image_nr = 0
'''
print("Start Mapping...")
for event_nr in random_list:
    test_pixel_values_1 = np.expand_dims(tel1[event_nr], axis=1)
    mapped_images_1[image_nr] = default_mapper.map_image(test_pixel_values_1, 'HESS-I')
    test_pixel_values_2 = np.expand_dims(tel2[event_nr], axis=1)
    mapped_images_2[image_nr] = default_mapper.map_image(test_pixel_values_2, 'HESS-I')
    test_pixel_values_3 = np.expand_dims(tel3[event_nr], axis=1)
    mapped_images_3[image_nr] = default_mapper.map_image(test_pixel_values_3, 'HESS-I')        
    test_pixel_values_4 = np.expand_dims(tel4[event_nr], axis=1)
    mapped_images_4[image_nr] = default_mapper.map_image(test_pixel_values_4, 'HESS-I')
    mapped_labels[image_nr] = labels[event_nr]
    image_nr=image_nr+1

print("... Finished Mapping")
'''

print("Start Mapping...")
for event_nr in random_list:
    #test_pixel_values_1 = np.expand_dims(tel1[event_nr], axis=1)
    mapped_images_1[image_nr] = ct_14_mapper.map_image(tel1[event_nr][3][:, np.newaxis], 'HESS-I')
    #mapped_images_1 = re_index_ct14(mapped_images_1)
    #test_pixel_values_2 = np.expand_dims(tel2[event_nr][3][:, np.newaxis], axis=1)
    mapped_images_2[image_nr] = ct_14_mapper.map_image(tel2[event_nr][3][:, np.newaxis], 'HESS-I')
    #mapped_images_2 = re_index_ct14(mapped_images_2)
    #test_pixel_values_3 = np.expand_dims(tel3[event_nr][3][:, np.newaxis], axis=1)
    mapped_images_3[image_nr] = ct_14_mapper.map_image(tel3[event_nr][3][:, np.newaxis], 'HESS-I')   
    #mapped_images_3 = re_index_ct14(mapped_images_3)     
    #test_pixel_values_4 = np.expand_dims(tel4[event_nr][3][:, np.newaxis], axis=1)
    mapped_images_4[image_nr] = ct_14_mapper.map_image(tel4[event_nr][3][:, np.newaxis], 'HESS-I')
    #mapped_images_4 = re_index_ct14(mapped_images_4)
    mapped_labels[image_nr] = labels[event_nr]
    image_nr=image_nr+1

print("... Finished Mapping")

mapped_images = np.array([mapped_images_1,mapped_images_2,mapped_images_3,mapped_images_4])
print(np.shape(mapped_images_1))
print(np.shape(mapped_images))

# Reshape the final array, so it is present in the same way as MoDAII data
mapped_images = np.transpose(mapped_images, (1, 0, 2, 3, 4))
mapped_images = np.squeeze(mapped_images, axis=-1)
mapped_labels = mapped_labels[:,np.newaxis]

print(np.shape(mapped_images))
print(np.shape(mapped_labels))




########################################################
# START WITH CNN STUFF

#num_epochs = 20
#batch_size = 512
#rate = 0.2
#reg = 0.001
patience = 4

input_shape = (41, 41, 1)
pool_size = 2
kernel_size = 4

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
#fnr = "_2023-07-03_"

peak_times = mapped_images
event_labels = mapped_labels

# some reshaping for the further use of the timing data in the CNN
peak_times = peak_times.reshape((*np.shape(peak_times),1))

# overview about the important data array for later usage
print(np.shape(peak_times)[0], " events with 4 images each are available \n")
print("Shape of 'event_labels': ",np.shape(event_labels))
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


plot_image_2by2(train_data,4,train_labels_multishape)
plot_image_2by2(train_data,40,train_labels_multishape)
plot_image_2by2(train_data,400,train_labels_multishape)
plot_image_2by2(train_data,4000,train_labels_multishape)


# Define the model for the single-view CNNs
def create_cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=40, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    #print("Before first Dropout")

    model.add(Dropout(rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    #print("After first Dropout")

    model.add(Dropout(rate))
    model.add(Conv2D(filters=60, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    #print("After second Dropout")

    model.add(Dropout(rate))
    model.add(Conv2D(filters=100, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    #print("After first Dropout")

    model.add(Dropout(rate))
    model.add(Conv2D(filters=150, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))


    return model

# Define the model for the combination of the previous CNNs and the final CNN for classification
input_shape = (41, 41, 1)

def run_multiview_model(models,inputs):

    merged = concatenate(models)

    Dropout1 = Dropout(rate)(merged)
    Conv_merged1 = Conv2D(filters=40,kernel_size=[2,2],activation='relu',padding='same',input_shape=input_shape)(Dropout1)
    MaxPool_merged1 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged1)

    Dropout2 = Dropout(rate)(MaxPool_merged1)
    Conv_merged2 = Conv2D(filters=50,kernel_size=[2,2],activation='relu',padding='same',input_shape=input_shape)(Dropout2)
    MaxPool_merged2 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged2)

    Dropout3 = Dropout(rate)(MaxPool_merged2)
    Conv_merged3 = Conv2D(filters=80,kernel_size=[2,2],activation='relu',padding='same',input_shape=input_shape)(Dropout3)
    MaxPool_merged3 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged3)

    Dropout31 = Dropout(rate)(MaxPool_merged3)
    Conv_merged31 = Conv2D(filters=140,kernel_size=[2,2],activation='relu',padding='same',input_shape=input_shape)(Dropout31)
    MaxPool_merged31 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged31)

    Flat_merged1 = Flatten()(MaxPool_merged31)
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
#early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

model_multi = run_multiview_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
model_multi.summary()
model_multi.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 


global training_generator
global testing_generator

training_generator = MyGenerator(train_data_1,train_data_2,train_data_3,train_data_4,train_labels)
testing_generator = MyGenerator(test_data_1,test_data_2,test_data_3,test_data_4,test_labels)

# Generate data and plot the first batch
batch_index = 0  
training_generator.plot_batch(batch_index)

testing_generator.reset_counters()
testing_generator.reset_counters()


#also add some checkpoints
early_stopping_callback_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,verbose=1,mode='min')
#checkpoints_2=tf.keras.callbacks.ModelCheckpoint('/home/hpc/b129dc/b129dc26/'+fnr+'checkpoints.ckpt',save_weights_only=True,save_best_only=True)
#callbacks_fixed = [OnEpochBegin(),early_stopping_callback_1,checkpoints_2]

print("Starting the Fitting ...")
#early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
#history = model_multi.fit(training_generator, epochs=num_epochs, steps_per_epoch=num_steps, validation_data=testing_generator, validation_steps=num_val_steps, callbacks=[early_stopping_callback_1])
history = model_multi.fit(training_generator, epochs=num_epochs, batch_size= batch_size,validation_data=testing_generator, callbacks=[early_stopping_callback_1])
#history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels, epochs=num_epochs, batch_size= batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping_callback_1])


str_batch_size = '{}'.format(batch_size)
str_rate = '{}'.format(rate*100)
str_reg = '{}'.format(reg)
str_num_epochs = '{}'.format(num_epochs)

history_name = "history_" + str_num_epochs + "epochs" + str_batch_size + "batchsize" + str_rate + "rate" + str_reg + "reg" + fnr + ".pkl"


#######################################
# Try Generator Stuff

####################################################


######################################################

#history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping])
print("... Finished the Fitting")
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
