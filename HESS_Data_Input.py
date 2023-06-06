# Processing HESS data for usage in CNN code from MoDA project
# Author: Hannes Warnhofer
# hannes.warnhofer@fau.de

import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import fnmatch
import os
import h5py
import glob
import pickle
import sys
import argparse

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

#filePath_gamma="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
filePath_gamma = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
data_g = tables.open_file(filePath_gamma, mode="r")

print("Successfully opened gamma data!")
#print(data_g)


# Assigning telescope data to different arrays
tel1g_raw = data_g.get_node('/dl1/event/telescope/images/tel_001').read()
tel2g_raw = data_g.get_node('/dl1/event/telescope/images/tel_002').read()
tel3g_raw = data_g.get_node('/dl1/event/telescope/images/tel_003').read()
tel4g_raw = data_g.get_node('/dl1/event/telescope/images/tel_004').read()

# Reshaping arrays and extracting the data
tel1g = np.stack([data[-1] for data in tel1g_raw])
tel2g = np.stack([data[-1] for data in tel2g_raw])
tel3g = np.stack([data[-1] for data in tel3g_raw])
tel4g = np.stack([data[-1] for data in tel4g_raw])

labelsg = np.stack([data[2] for data in tel1g_raw])
labelsg_ones = np.ones_like(labelsg)

del tel1g_raw
del tel2g_raw
del tel3g_raw
del tel4g_raw

data_g.close()

filePath_proton="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
#filePath_proton = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
data_p = tables.open_file(filePath_proton, mode="r")

print("Successfully opened proton data!")
#print(data_p)

# Assigning telescope data to different arrays
tel1p_raw = data_p.get_node('/dl1/event/telescope/images/tel_001').read()
tel2p_raw = data_p.get_node('/dl1/event/telescope/images/tel_002').read()
tel3p_raw = data_p.get_node('/dl1/event/telescope/images/tel_003').read()
tel4p_raw = data_p.get_node('/dl1/event/telescope/images/tel_004').read()

# Reshaping arrays and extracting the data
tel1p = np.stack([data[-1] for data in tel1p_raw])
tel2p = np.stack([data[-1] for data in tel2p_raw])
tel3p = np.stack([data[-1] for data in tel3p_raw])
tel4p = np.stack([data[-1] for data in tel4p_raw])

labelsp = np.stack([data[2] for data in tel1p_raw])
labelsp_zeros = np.zeros_like(labelsp)

del tel1p_raw
del tel2p_raw
del tel3p_raw
del tel4p_raw

data_p.close()

tel1 = np.concatenate((tel1g,tel1p),axis=0)
tel2 = np.concatenate((tel2g,tel2p),axis=0)
tel3 = np.concatenate((tel3g,tel3p),axis=0)
tel4 = np.concatenate((tel4g,tel4p),axis=0)
labels = np.concatenate((labelsg_ones,labelsp_zeros),axis=0)

del tel1p
del tel1g
del tel2p
del tel2g
del tel3p
del tel3g
del tel4p
del tel4g
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

# Define the camera types and mapping methods
hex_cams = ['HESS-I']
camera_types = hex_cams 
hex_methods = ['oversampling', 'rebinning', 'nearest_interpolation',
               'bilinear_interpolation', 'bicubic_interpolation', 
               'image_shifting', 'axial_addressing']

#Load the image mappers
mappers = {}
print("Initialization time (total for all telescopes):")
for method in hex_methods:
    print(method)
    mapping_method = {cam: method for cam in hex_cams}
    mappers[method] = ImageMapper(mapping_method=mapping_method,camera_types=["HESS-I"])

# Reshape arrays for mapping
num_pixels = len(CameraGeometry.from_name('HESS-I').pix_id)
test_pixel_values = np.empty((len(tel1),num_pixels))
print(np.shape(test_pixel_values))
test_pixel_values_1 = test_pixel_values
test_pixel_values_2 = test_pixel_values
test_pixel_values_3 = test_pixel_values
test_pixel_values_4 = test_pixel_values

test_pixel_values_1[:] = tel1[:]
test_pixel_values_2[:] = tel2[:]
test_pixel_values_3[:] = tel3[:]
test_pixel_values_4[:] = tel4[:]

test_pixel_values_1 = np.expand_dims(test_pixel_values_1, axis=2)
test_pixel_values_2 = np.expand_dims(test_pixel_values_2, axis=2)
test_pixel_values_3 = np.expand_dims(test_pixel_values_3, axis=2)
test_pixel_values_4 = np.expand_dims(test_pixel_values_4, axis=2)

# Mapping the images
num_events = 10000 #len(test_pixel_values) # Takes very long with many events on my PC, for testing: num_events = 10000 (len(test_pixel_values)=106319)
default_mapper = ImageMapper(camera_types=['HESS-I'])
padding_mapper = ImageMapper(padding={cam: 10 for cam in camera_types}, camera_types=["HESS-I"])

image_shape = default_mapper.map_image(test_pixel_values_1[0], 'HESS-I').shape
mapped_images_1 = np.empty((num_events,) + image_shape)
mapped_images_2 = mapped_images_1
mapped_images_3 = mapped_images_1
mapped_images_4 = mapped_images_1
mapped_labels = np.empty(num_events)

length = num_events
max_value = len(test_pixel_values)
random_list = random.sample(range(max_value),length) 
image_nr = 0
for event_nr in random_list:
    image = default_mapper.map_image(test_pixel_values_1[event_nr], 'HESS-I')
    mapped_images_1[image_nr] = image
    image = default_mapper.map_image(test_pixel_values_2[event_nr], 'HESS-I')
    mapped_images_2[image_nr] = image
    image = default_mapper.map_image(test_pixel_values_3[event_nr], 'HESS-I')
    mapped_images_3[image_nr] = image    
    image = default_mapper.map_image(test_pixel_values_4[event_nr], 'HESS-I')
    mapped_images_4[image_nr] = image
    mapped_labels[image_nr] = labels[event_nr]
    image_nr=image_nr+1

mapped_images = np.array([mapped_images_1,mapped_images_2,mapped_images_3,mapped_images_4])
print(np.shape(mapped_images_1))
print(np.shape(mapped_images))

# Reshape the final array, so it is present in the same way as MoDAII data
mapped_images = np.transpose(mapped_images, (1, 0, 2, 3, 4))
mapped_images = np.squeeze(mapped_images, axis=-1)
mapped_labels = mapped_labels[:,np.newaxis]

print(np.shape(mapped_images))
print(np.shape(mapped_labels))