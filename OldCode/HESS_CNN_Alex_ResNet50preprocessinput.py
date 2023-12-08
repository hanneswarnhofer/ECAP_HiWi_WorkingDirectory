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

from datetime import datetime
import time

from ctapipe.io import EventSource
from ctapipe import utils
from ctapipe.instrument.camera import CameraGeometry

from dl1_data_handler.reader import DL1DataReader
from dl1_data_handler.image_mapper import ImageMapper

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization, LeakyReLU
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
        current_timestamp = int(time.time())
        np.random.seed(current_timestamp)
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

    def __init__(self,images_1,images_2,images_3,images_4,labels,batch_size):
        self.batch_size = batch_size
        self.images_1 = images_1
        self.images_2 = images_2
        self.images_3 = images_3
        self.images_4 = images_4
        self.labels = labels
        self.sample_count = len(labels[:])
        self.batch_count = int(self.sample_count/self.batch_size)
        self.current_batch = 0
        self.index = 0

    def __len__(self):
        return self.batch_count



    def __getitem__(self,index):
        idx_low  = self.current_batch*self.batch_size
        idx_high = (self.current_batch+1)*self.batch_size
        X = np.array([self.images_1[idx_low:idx_high],self.images_2[idx_low:idx_high],self.images_3[idx_low:idx_high],self.images_4[idx_low:idx_high]])
        y = np.array(self.labels[idx_low:idx_high])

        self.current_batch +=1 
        self.data = (X,y)

        return self.data
        
    ''' 

    def __getitem__(self,index):
        idx_low  = self.current_batch*self.batch_size
        idx_high = (self.current_batch+1)*self.batch_size
        images_batch_1 = self.images_1[idx_low:idx_high]
        images_batch_2 = self.images_2[idx_low:idx_high]
        images_batch_3 = self.images_3[idx_low:idx_high]
        images_batch_4 = self.images_4[idx_low:idx_high]

        labels_batch = np.array(self.labels[idx_low:idx_high])

        # Assuming your images are of shape (41, 41, 1)
        images_batch_1 = np.expand_dims(images_batch_1, axis=-1)
        images_batch_2 = np.expand_dims(images_batch_2, axis=-1)
        images_batch_3 = np.expand_dims(images_batch_3, axis=-1)
        images_batch_4 = np.expand_dims(images_batch_4, axis=-1)

        self.current_batch +=1 
        self.data = (np.array([images_batch_1, images_batch_2, images_batch_3, images_batch_4]), labels_batch)

        # MAYBE CHECK: X = np.array([...]) and y = labels_batch

        return self.data
    '''
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.sample_count:
            raise StopIteration
        result = self.__getitem__(self.index) 
        self.index += 1
        return result

    def plot_batch(self, event_nr=100):
        print(np.shape(self.images_1))
        print(np.shape(self.images_1[event_nr]))
        image_batch_1 = self.images_1
        image_batch_2 = self.images_2
        image_batch_3 = self.images_3
        image_batch_4 = self.images_4
        label_batch = self.labels
        plot_image_2by2(image_batch_1,image_batch_2,image_batch_3,image_batch_4, labels=label_batch, event_nr=event_nr,string="generator")

    def __shape__(self):
        print("Shape of self.images1: ",np.shape(self.images_1))
        print("Shape of first event: ",np.shape(self.images_1[0]))

    def reset_counters(self): 
        self.current_batch = 0 

        
    def on_epoch_end(self):
        self.reset_counters()

class OnEpochBegin(keras.callbacks.Callback): # Callback class called on epoch begin to reset counters
    def on_epoch_begin(self, epoch, logs=None):
        training_generator.reset_counters()
        testing_generator.reset_counters()
        print("Epoch Begin")


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

def plot_image_2by2(train_data,event_nr,labels,string,dt):

    

    

    print("Plotting Example Event. Event Nr: ", event_nr)

    image1 = train_data[:,0,:,:] 
    image2 = train_data[:,1,:,:] 
    image3 = train_data[:,2,:,:] 
    image4 = train_data[:,3,:,:] 

    pltimage1 = image1[event_nr]
    pltimage2 = image2[event_nr]
    pltimage3 = image3[event_nr]
    pltimage4 = image4[event_nr]

    fig, ax = plt.subplots(2,2)

    im1 = ax[0,0].imshow(pltimage1[:,:,0], cmap='viridis',vmin=0)
    im2 = ax[0,1].imshow(pltimage2[:,:,0], cmap='viridis',vmin=0)
    im3 = ax[1,0].imshow(pltimage3[:,:,0], cmap='viridis',vmin=0)
    im4 = ax[1,1].imshow(pltimage4[:,:,0], cmap='viridis',vmin=0)

    cbar1 = fig.colorbar(im1, ax=ax[0, 0], orientation='vertical')
    cbar2 = fig.colorbar(im2, ax=ax[0, 1], orientation='vertical')
    cbar3 = fig.colorbar(im3, ax=ax[1, 0], orientation='vertical')
    cbar4 = fig.colorbar(im4, ax=ax[1, 1], orientation='vertical')


    label1 = labels[event_nr].ravel()[0]
    label2 = labels[event_nr].ravel()[1]
    label3 = labels[event_nr].ravel()[2]
    label4 = labels[event_nr].ravel()[3]

    str_label1 = '{}'.format(label1)
    str_label2 = '{}'.format(label2)
    str_label3 = '{}'.format(label3)
    str_label4 = '{}'.format(label4)

    ax[0, 0].text(0.05, 0.95, str_label1, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[0, 1].text(0.05, 0.95, str_label2, transform=ax[0, 1].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 0].text(0.05, 0.95, str_label3, transform=ax[1, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 1].text(0.05, 0.95, str_label4, transform=ax[1, 1].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    #plt.show()

    print("Min. and Max. Value for Image 1: ", np.min(pltimage1), " - " , np.max(pltimage1) , ". Sum: ", np.sum(pltimage1))
    print("Min. and Max. Value for Image 2: ", np.min(pltimage2), " - " , np.max(pltimage2), ". Sum: ", np.sum(pltimage2))
    print("Min. and Max. Value for Image 3: ", np.min(pltimage3), " - " , np.max(pltimage3), ". Sum: ", np.sum(pltimage3))
    print("Min. and Max. Value for Image 4: ", np.min(pltimage4), " - " , np.max(pltimage4), ". Sum: ", np.sum(pltimage4))


    str_evnr = '{}'.format(event_nr)
    name = "Test_images/Test_figure_evnr_" + str_evnr + "_" + string + "_" + dt + ".png"
    fig.savefig(name)

def plot_image_2by2_v2(image1,image2,image3,image4,event_nr,labels,string,dt):

    

    

    print("Plotting Example Event. Event Nr: ", event_nr)


    pltimage1 = image1[event_nr]
    pltimage2 = image2[event_nr]
    pltimage3 = image3[event_nr]
    pltimage4 = image4[event_nr]

    fig, ax = plt.subplots(2,2)

    im1 = ax[0,0].imshow(pltimage1[:,:,0], cmap='viridis',vmin=0)
    im2 = ax[0,1].imshow(pltimage2[:,:,0], cmap='viridis',vmin=0)
    im3 = ax[1,0].imshow(pltimage3[:,:,0], cmap='viridis',vmin=0)
    im4 = ax[1,1].imshow(pltimage4[:,:,0], cmap='viridis',vmin=0)

    cbar1 = fig.colorbar(im1, ax=ax[0, 0], orientation='vertical')
    cbar2 = fig.colorbar(im2, ax=ax[0, 1], orientation='vertical')
    cbar3 = fig.colorbar(im3, ax=ax[1, 0], orientation='vertical')
    cbar4 = fig.colorbar(im4, ax=ax[1, 1], orientation='vertical')


    label1 = labels[event_nr].ravel()[0]
    label2 = labels[event_nr].ravel()[1]
    label3 = labels[event_nr].ravel()[2]
    label4 = labels[event_nr].ravel()[3]

    str_label1 = '{}'.format(label1)
    str_label2 = '{}'.format(label2)
    str_label3 = '{}'.format(label3)
    str_label4 = '{}'.format(label4)

    ax[0, 0].text(0.05, 0.95, str_label1, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[0, 1].text(0.05, 0.95, str_label2, transform=ax[0, 1].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 0].text(0.05, 0.95, str_label3, transform=ax[1, 0].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 1].text(0.05, 0.95, str_label4, transform=ax[1, 1].transAxes, color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.7))
    #plt.show()

    print("Min. and Max. Value for Image 1: ", np.min(pltimage1), " - " , np.max(pltimage1) , ". Sum: ", np.sum(pltimage1))
    print("Min. and Max. Value for Image 2: ", np.min(pltimage2), " - " , np.max(pltimage2), ". Sum: ", np.sum(pltimage2))
    print("Min. and Max. Value for Image 3: ", np.min(pltimage3), " - " , np.max(pltimage3), ". Sum: ", np.sum(pltimage3))
    print("Min. and Max. Value for Image 4: ", np.min(pltimage4), " - " , np.max(pltimage4), ". Sum: ", np.sum(pltimage4))


    str_evnr = '{}'.format(event_nr)
    name = "Test_images/Test_figure_evnr_" + str_evnr + "_" + string + "_" + dt + ".png"
    fig.savefig(name)

print("Functions Defined.")


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-r", "--rate", type=float)
parser.add_argument("-reg", "--regulization", type=float)
parser.add_argument("-t", "--threshold", type=float)
parser.add_argument("-c", "--cut", type=int)
parser.add_argument("-ne", "--numevents", type=int)

args = parser.parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
rate = args.rate
reg = args.regulization
sum_threshold = args.threshold
cut_nonzero = args.cut
num_events = args.numevents

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "ResNet50_pp" 

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
print("Date-Time: ", formatted_datetime)

#num_events = 100000
amount = int(num_events * 2)
#filePath_gamma="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
#filePath_gamma = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
filePath_gamma = "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"

#filePath_proton="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
#filePath_proton = "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
filePath_proton="../../../../wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"

dm_gamma = DataManager(filePath_gamma)
f_g = dm_gamma.get_h5_file()

if num_events >= len(f_g["dl1/event/telescope/images/tel_001"][:]) : num_events = len(f_g["dl1/event/telescope/images/tel_001"][:]) - 2
if amount >= len(f_g["dl1/event/telescope/images/tel_001"][:]) : amount = len(f_g["dl1/event/telescope/images/tel_001"][:]) - 1

tel1g_raw = f_g["dl1/event/telescope/images/tel_001"][0:amount]
tel2g_raw = f_g["dl1/event/telescope/images/tel_002"][0:amount]
tel3g_raw = f_g["dl1/event/telescope/images/tel_003"][0:amount]
tel4g_raw = f_g["dl1/event/telescope/images/tel_004"][0:amount]
#tel5g_raw = f_g["dl1/event/telescope/images/tel_005"][0:amount]

print("Successfully opened gamma data!")

labelsg = np.stack([data[2] for data in tel1g_raw])
labelsg_ones = np.ones_like(labelsg)

f_g.close()

dm_proton = DataManager(filePath_proton)
f_p = dm_proton.get_h5_file()

tel1p_raw = f_p["dl1/event/telescope/images/tel_001"][0:amount]
tel2p_raw = f_p["dl1/event/telescope/images/tel_002"][0:amount]
tel3p_raw = f_p["dl1/event/telescope/images/tel_003"][0:amount]
tel4p_raw = f_p["dl1/event/telescope/images/tel_004"][0:amount]
#tel5p_raw = f_p["dl1/event/telescope/images/tel_005"][0:amount]

print("Successfully opened proton data!")

labelsp = np.stack([data[2] for data in tel1p_raw])
labelsp_zeros = np.zeros_like(labelsp)

tel1 = np.concatenate((tel1g_raw,tel1p_raw),axis=0)
tel2 = np.concatenate((tel2g_raw,tel2p_raw),axis=0)
tel3 = np.concatenate((tel3g_raw,tel3p_raw),axis=0)
tel4 = np.concatenate((tel4g_raw,tel4p_raw),axis=0)
#tel5 = np.concatenate((tel5g_raw,tel5p_raw),axis=0)
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

del labelsp
del labelsg
del labelsp_zeros
del labelsg_ones

print("Shape of Tel1: ",np.shape(tel1))
print("Shape of Tel2: ",np.shape(tel2))
print("Shape of Tel3: ",np.shape(tel3))
print("Shape of Tel4: ",np.shape(tel4))
#print("Shape of Tel5: ",np.shape(tel5))
print("Shape of Labels: ",np.shape(labels))
print("Labels: ",labels)

geo_ct14, geo_ct5 = make_hess_geometry()
print(os.getcwd())
ct_14_mapper = ImageMapper(camera_types=["HESS-I"], pixel_positions={"HESS-I": rotate(geo_ct14.get_pix_pos())}, mapping_method={"HESS-I": "axial_addressing"})
#ct_5_mapper = ImageMapper(camera_types=["HESS-II"], pixel_positions={"HESS-II": rotate(geo_ct5.get_pix_pos())}, mapping_method={"HESS-II": "axial_addressing"})

mapped_images_1 = np.empty((num_events, 41,41,1))
mapped_images_2 = np.empty((num_events, 41,41,1))
mapped_images_3 = np.empty((num_events, 41,41,1))
mapped_images_4 = np.empty((num_events, 41,41,1))
#mapped_images_4 = np.empty((num_events, 41,41,1))
mapped_labels = np.empty(num_events)

length = num_events
max_value = len(tel1)
random_list = np.random.randint(max_value, size=length)
image_nr = 0

print(random_list[0:10])


threshold_value = 0.0001  # Adjust this threshold value as needed

print("Start Mapping...")
for event_nr in random_list:


    
    image_1 = ct_14_mapper.map_image(tel1[event_nr][3][:, np.newaxis], 'HESS-I')
    image_2 = ct_14_mapper.map_image(tel2[event_nr][3][:, np.newaxis], 'HESS-I')
    image_3 = ct_14_mapper.map_image(tel3[event_nr][3][:, np.newaxis], 'HESS-I')   
    image_4 = ct_14_mapper.map_image(tel4[event_nr][3][:, np.newaxis], 'HESS-I')
    #image_5 = ct_5_mapper.map_image(tel5[event_nr][3][:, np.newaxis], 'HESS-II')   

    # Apply threshold on the sum of pixel values
    #sum_threshold = 60  # Adjust this value to your desired threshold
    sum_threshold = args.threshold

    if np.sum(image_1) < sum_threshold:
        image_1[:] = 0
    if np.sum(image_2) < sum_threshold:
        image_2[:] = 0
    if np.sum(image_3) < sum_threshold:
        image_3[:] = 0
    if np.sum(image_4) < sum_threshold:
        image_4[:] = 0
     
    # Set all pixels lower than the threshold value to zero
    image_1[image_1 < threshold_value] = 0
    image_2[image_2 < threshold_value] = 0
    image_3[image_3 < threshold_value] = 0
    image_4[image_4 < threshold_value] = 0
    #image_5[image_5 < threshold_value] = 0

    non_zero_count = sum(1 for img in [image_1, image_2, image_3, image_4] if np.sum(img) > 0)
    if non_zero_count >= cut_nonzero:
        mapped_images_1[image_nr] = image_1
        mapped_images_2[image_nr] = image_2
        mapped_images_3[image_nr] = image_3
        mapped_images_4[image_nr] = image_4
        #mapped_images_5[image_nr] = image_5
        mapped_labels[image_nr] = labels[event_nr]
        image_nr += 1
    
print("... Finished Mapping")

'''
print("Start Mapping...")
for event_nr in random_list:
    mapped_images_1[image_nr] = ct_14_mapper.map_image(tel1[event_nr][3][:, np.newaxis], 'HESS-I')
    mapped_images_2[image_nr] = ct_14_mapper.map_image(tel2[event_nr][3][:, np.newaxis], 'HESS-I')
    mapped_images_3[image_nr] = ct_14_mapper.map_image(tel3[event_nr][3][:, np.newaxis], 'HESS-I')   
    mapped_images_4[image_nr] = ct_14_mapper.map_image(tel4[event_nr][3][:, np.newaxis], 'HESS-I')
    mapped_labels[image_nr] = labels[event_nr]
    image_nr=image_nr+1
print("... Finished Mapping")
'''
#########################################   MAYBE TRY CONVERTING ALL "EMPTY" IMAGES IN SUCH A WAY
#########################################   THAT ALL PIXELS BECOME ZERO? IF SUM < 0 -> ALL_PIXELS=0
#########################################   


mapped_images = np.array([mapped_images_1,mapped_images_2,mapped_images_3,mapped_images_4]) #mapped_images_5])
print("Shape of mapped_images_1: ",np.shape(mapped_images_1))
print("Shape of mapped_images: ",np.shape(mapped_images))

del tel1
del tel2
del tel3
del tel4
del labels

del mapped_images_1
del mapped_images_2
del mapped_images_3
del mapped_images_4

# Reshape the final array, so it is present in the same way as MoDAII data
mapped_images = np.transpose(mapped_images, (1, 0, 2, 3, 4))
mapped_images = np.squeeze(mapped_images, axis=-1)
mapped_labels = mapped_labels[:,np.newaxis]

print("New shape of mapped_images: ",np.shape(mapped_images))
print("New shape of mapped_labels: ",np.shape(mapped_labels))


########################################################
# START WITH CNN STUFF


patience = 5
input_shape = (41, 41, 3)
#input_shape5 = (72,72,1)
pool_size = 2
kernel_size = 2

# some reshaping for the further use of the timing data in the CNN
mapped_images = mapped_images.reshape((*np.shape(mapped_images),1))

# overview about the important data array for later usage
print(np.shape(mapped_images)[0], " events with 4 images each are available \n")
print("Shape of 'event_labels': ",np.shape(mapped_labels))
print("Shape of 'peak_times': ",np.shape(mapped_images),"\n")

# split into random training data (80%) and test data (20%)
train_data = []
test_data = []
train_labels = []
test_labels = [] 

#data_dummy = mapped_images

random_selection = np.random.rand(np.shape(mapped_images)[0]) <= 0.8


train_data.append(mapped_images[random_selection])
test_data.append(mapped_images[~random_selection])
train_labels.append(mapped_labels[random_selection])
test_labels.append(mapped_labels[~random_selection])

#mapped_images = data_dummy
#del data_dummy

print(random_selection[0:10])

# free some memory space
del mapped_images
del mapped_labels

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
#train_data_5 = train_data[:,4,:,:] 

test_data_1 = test_data[:,0,:,:]
test_data_2 = test_data[:,1,:,:]
test_data_3 = test_data[:,2,:,:]
test_data_4 = test_data[:,3,:,:]
#test_data_5 = test_data[:,4,:,:]

train_labels_1 = train_labels_multishape[:,0,:,:]
train_labels_2 = train_labels_multishape[:,1,:,:]
train_labels_3 = train_labels_multishape[:,2,:,:]
train_labels_4 = train_labels_multishape[:,3,:,:]
#train_labels_5 = train_labels_multishape[:,4,:,:]

test_labels_1 = test_labels_multishape[:,0,:,:]
test_labels_2 = test_labels_multishape[:,1,:,:]
test_labels_3 = test_labels_multishape[:,2,:,:]
test_labels_4 = test_labels_multishape[:,3,:,:]
#test_labels_5 = test_labels_multishape[:,4,:,:]

print("Train data 1 shape:", np.shape(train_data_1))
print("Train labels 1 shape:", np.shape(train_labels_1))

print("Test data 1 shape:", np.shape(test_data_1))
print("Test labels 1 shape:", np.shape(test_labels_1))


'''
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.resnet50.preprocess_input(x)
core = tf.keras.applications.resnet50()
x = core(x)
model = tf.keras.Model(inputs=[i], outputs=[x])

train_data_1 = model(train_data_1)
'''

plot_image_2by2(train_data,4,train_labels_multishape,string="train",dt=formatted_datetime)
#plot_image_2by2(train_data,40,train_labels_multishape,string="train",dt=formatted_datetime)
#plot_image_2by2(train_data,400,train_labels_multishape,string="train",dt=formatted_datetime)
#plot_image_2by2(train_data,4000,train_labels_multishape,string="train",dt=formatted_datetime)

plot_image_2by2(test_data,4,test_labels_multishape,string="test",dt=formatted_datetime)
#plot_image_2by2(test_data,40,test_labels_multishape,string="test",dt=formatted_datetime)
#plot_image_2by2(test_data,400,test_labels_multishape,string="test",dt=formatted_datetime)
#plot_image_2by2(test_data,4000,test_labels_multishape,string="test",dt=formatted_datetime)




#Define the model for the single-view CNNs
def create_cnn_model(input_shape):
    input_layer = Input(shape=input_shape)
    #x = Conv2D(3, (1, 1))(input_layer)
    #preprocessed_input = preprocess_input(x)
    #Seq_model = Sequential 
    ResNet50_model = ResNet50(include_top=False, weights=None, input_tensor=input_layer)

    Conv1 = Conv2D(filters=200, kernel_size=kernel_size, padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,)(ResNet50_model.output)
    LeakyRelu1 = LeakyReLU(alpha=0.1)(Conv1)
    MaxPool1 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu1)

    #print("Before first Dropout")

    Dropout1 = Dropout(rate)(MaxPool1)
    Conv2 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout1)
    LeakyRelu2 = LeakyReLU(alpha=0.1)(Conv2) 
    MaxPool2 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu2)

    Dropout2 = Dropout(rate)(MaxPool2)
    Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
    LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
    MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

    Dropout3 = Dropout(rate)(MaxPool3)
    Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
    LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
    MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

    Dropout4 = Dropout(rate)(MaxPool4)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(rate)(MaxPool6)
    Conv7 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    model = Model(inputs=input_layer, outputs=MaxPool7)
    return model

# Define the model for the combination of the previous CNNs and the final CNN for classification

def run_multiview_model(models,inputs):

    merged = concatenate(models)

    Dropout1 = Dropout(rate)(merged)
    Conv_merged1 = Conv2D(filters=100,kernel_size=[2,2],activation='relu',padding='same',input_shape=input_shape)(Dropout1)
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

    Dropout6 = Dropout(rate)(dense_layer_merged1)
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


def preprocess_input_resnet(data):
    # Expand single-channel image to three channels
    rgb_data = np.repeat(data,3,axis=-1)
    mean = [103.939, 116.779, 123.68]
    std = None

    rgb_data[:,:,:,:,0] -= mean[0]
    rgb_data[:,:,:,:,1] -= mean[1]
    rgb_data[:,:,:,:,2] -= mean[2]
    return rgb_data

def custom_preprocess_input(data):
    normalized_data = np.empty(data.shape)
    # Perform custom preprocessing (e.g., scaling)
    #max_values = np.max(data,axis=(2,3,4),keepdims=True)
    for event in range(data.shape[0]):
        #max_values = np.max(data,axis=(1,2,3,4))
        #print(max_values)
        #normalized_data = np.where(max_values == 0, data, data / max_values[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])  
        max_value = np.max(data[event]) 
        if max_value == 0:
            normalized_data[event] = data[event]
        else:
            data[event][np.isnan(data[event])] = 0
            normalized_data[event] = data[event]/max_value     
    return normalized_data

#cpp_train_data = custom_preprocess_input(train_data)
#cpp_test_data = custom_preprocess_input(test_data)

pp_train_data = preprocess_input_resnet(train_data)
pp_test_data = preprocess_input_resnet(test_data)

'''
cpp_test_data_1 = custom_preprocess_input(test_data_1)
cpp_test_data_2 = custom_preprocess_input(test_data_2)
cpp_test_data_3 = custom_preprocess_input(test_data_3)
cpp_test_data_4 = custom_preprocess_input(test_data_4)

cpp_train_data_1 = custom_preprocess_input(train_data_1)
cpp_train_data_2 = custom_preprocess_input(train_data_2)
cpp_train_data_3 = custom_preprocess_input(train_data_3)
cpp_train_data_4 = custom_preprocess_input(train_data_4)


#pp_test_data_1 = preprocess_input(convert_to_rgb(test_data_1.reshape(test_data_1.shape[:-1])))
#pp_test_data_2 = preprocess_input(convert_to_rgb(test_data_2.reshape(test_data_2.shape[:-1])))
#pp_test_data_3 = preprocess_input(convert_to_rgb(test_data_3.reshape(test_data_3.shape[:-1])))
#pp_test_data_4 = preprocess_input(convert_to_rgb(test_data_4.reshape(test_data_4.shape[:-1])))

#pp_train_data_1 = preprocess_input(convert_to_rgb(train_data_1.reshape(train_data_1.shape[:-1])))
#pp_train_data_2 = preprocess_input(convert_to_rgb(train_data_2.reshape(train_data_2.shape[:-1])))
#pp_train_data_3 = preprocess_input(convert_to_rgb(train_data_3.reshape(train_data_3.shape[:-1])))
#pp_train_data_4 = preprocess_input(convert_to_rgb(train_data_4.reshape(train_data_4.shape[:-1])))
'''
'''
cpp_train_data = np.zeros_like(train_data)
cpp_train_data[:,0,:,:]=cpp_train_data_1
cpp_train_data[:,1,:,:]=cpp_train_data_2
cpp_train_data[:,2,:,:]=cpp_train_data_3
cpp_train_data[:,3,:,:]=cpp_train_data_4

pp_train_data = np.zeros_like(convert_to_rgb(train_data.reshape(train_data.shape[:-1])))
pp_train_data[:,0,:,:]=pp_train_data_1
pp_train_data[:,1,:,:]=pp_train_data_2
pp_train_data[:,2,:,:]=pp_train_data_3
pp_train_data[:,3,:,:]=pp_train_data_4
'''

#plot_image_2by2_v2(cpp_train_data_1,cpp_train_data_2,cpp_train_data_3,cpp_train_data_4,4,train_labels_multishape,string="cpptrain",dt=formatted_datetime)
#plot_image_2by2_v2(pp_train_data_1,pp_train_data_2,pp_train_data_3,pp_train_data_4,4,train_labels_multishape,string="pptrain",dt=formatted_datetime)

plot_image_2by2(pp_train_data,4,train_labels_multishape,string="cpptrain",dt=formatted_datetime)
plot_image_2by2(pp_test_data,4,test_labels_multishape,string="cpptest",dt=formatted_datetime)

#print("Max:",np.max(train_data_1))
#print("Max_pp:",np.max(pp_train_data_1))

# include early_stopping here, to see how it changes compared to previous model designs
#early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

model_multi = run_multiview_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
model_multi.summary()
#model_multi.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'],from_logits=True) 
from keras.losses import BinaryCrossentropy

# Create the loss function with from_logits=True
loss_fn = BinaryCrossentropy(from_logits=True)

# Compile the model using the created loss function
model_multi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#global training_generator
#global testing_generator

#training_generator = MyGenerator(train_data_1,train_data_2,train_data_3,train_data_4,train_labels,batch_size=batch_size)
#testing_generator = MyGenerator(test_data_1,test_data_2,test_data_3,test_data_4,test_labels,batch_size=batch_size)

#training_generator = MyGenerator(train_data_1,train_data_2,train_data_3,train_data_4,train_labels,batch_size=batch_size)
#testing_generator = MyGenerator(test_data_1,test_data_2,test_data_3,test_data_4,test_labels,batch_size=batch_size)

# Generate data and plot the first batch
#batch_index = 0  
#training_generator.plot_batch(batch_index)

#testing_generator.reset_counters()
#testing_generator.reset_counters()
early_stopping_callback_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,verbose=1,mode='min')

print("Starting the Fitting ...")

'''
history = model_multi.fit(
    x=([training_generator.images_1, training_generator.images_2, training_generator.images_3, training_generator.images_4], training_generator.labels),
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(
        [testing_generator.images_1, testing_generator.images_2, testing_generator.images_3, testing_generator.images_4],
        testing_generator.labels
    ),
    callbacks=[early_stopping_callback_1]
)
'''
#history = model_multi.fit(training_generator, epochs=num_epochs, batch_size= batch_size,validation_data=testing_generator, callbacks=[early_stopping_callback_1])
#history = model_multi.fit([cpp_train_data_1,cpp_train_data_2,cpp_train_data_3,cpp_train_data_4],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([cpp_test_data_1,cpp_test_data_2,cpp_test_data_3,cpp_test_data_4], test_labels), callbacks=[early_stopping_callback_1])
#history_pp = history = model_multi.fit([pp_train_data_1,pp_train_data_2,pp_train_data_3,pp_train_data_4],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([pp_test_data_1,pp_test_data_2,pp_test_data_3,pp_test_data_4], test_labels), callbacks=[early_stopping_callback_1])


history = model_multi.fit([pp_train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([pp_test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping_callback_1])
str_batch_size = '{}'.format(batch_size)
str_rate = '{}'.format(rate*100)
str_reg = '{}'.format(reg)
str_num_epochs = '{}'.format(num_epochs)
str_thr = '{}'.format(sum_threshold)
str_cnz = '{}'.format(cut_nonzero)

name_str = fnr + "_" + str_num_epochs + "epochs" + str_batch_size + "batchsize" + str_rate + "rate" + str_reg + "reg" + str_thr + "threshold" + str_cnz + "nonzerocut" + "_" + formatted_datetime 



print("... Finished the Fitting")

# Save the history files for later usage in other scripts

history_name = "HistoryFiles/history_" + name_str + ".pkl"

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


filename_savefig = "Images/Test_Cluster_" + name_str + ".png"
fig.savefig(filename_savefig, bbox_inches='tight')

print("Image saved")

