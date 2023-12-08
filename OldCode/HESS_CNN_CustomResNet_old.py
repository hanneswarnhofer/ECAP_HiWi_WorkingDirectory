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
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Lambda,Reshape,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization, LeakyReLU
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

def preprocess_data(train_data, train_labels, threshold=0.000001):
    # Get the dimensions of the input data
    num_events, num_views, height, width, channels = train_data.shape
    
    # Reshape train_data into a single array with shape (num_events * num_views, height, width, channels)
    reshaped_data = train_data.reshape(-1, height, width, channels)
    reshaped_labels = np.repeat(train_labels,4,axis=0)


    # Compute the sum of pixel values along the last axis (assuming your images are grayscale)
    pixel_sums = np.sum(reshaped_data, axis=(1, 2, 3))
    
    # Find indices of images with sums greater than or equal to the threshold
    valid_indices = np.where(pixel_sums >= threshold)
    
    # Filter the reshaped data based on valid_indices
    filtered_data = reshaped_data[valid_indices]
    
    # Reshape train_labels to match the reshaped data
    filtered_labels = reshaped_labels[valid_indices]

    combined_data_labels = list(zip(filtered_data, filtered_labels))

    # Shuffle the combined array
    np.random.shuffle(combined_data_labels)

    # Split the shuffled array back into filtered_data and filtered_labels
    shuffled_data, shuffled_labels = zip(*combined_data_labels)

    # Convert them back to NumPy arrays
    shuffled_data = np.array(shuffled_data)
    shuffled_labels = np.array(shuffled_labels)

    return shuffled_data, shuffled_labels


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
    plt.show()

    print("Min. and Max. Value for Image 1: ", np.min(pltimage1), " - " , np.max(pltimage1) , ". Sum: ", np.sum(pltimage1))
    print("Min. and Max. Value for Image 2: ", np.min(pltimage2), " - " , np.max(pltimage2), ". Sum: ", np.sum(pltimage2))
    print("Min. and Max. Value for Image 3: ", np.min(pltimage3), " - " , np.max(pltimage3), ". Sum: ", np.sum(pltimage3))
    print("Min. and Max. Value for Image 4: ", np.min(pltimage4), " - " , np.max(pltimage4), ". Sum: ", np.sum(pltimage4))

    #str_evnr = '{}'.format(event_nr)
    #name = "Test_images/Test_figure_evnr_" + str_evnr + "_" + string + "_" + dt + ".png"
    #fig.savefig(name)

print("Functions Defined.")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-r", "--rate", type=float)
parser.add_argument("-reg", "--regulization", type=float)
parser.add_argument("-t", "--threshold", type=float)
parser.add_argument("-c", "--cut", type=int)
parser.add_argument("-ne", "--numevents", type=int)
parser.add_argument("-ft","--fusiontype",type=str,default="latefc")
parser.add_argument("-n","--normalize",type=str,default="nonorm")
parser.add_argument("-loc","--location",type=str,default="alex")
parser.add_argument("-transfer","--transfer",type=str,default="no")
parser.add_argument("-base","--base",type=str,default='moda')

args = parser.parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
rate = args.rate
reg = args.regulization
sum_threshold = args.threshold
cut_nonzero = args.cut
num_events = args.numevents
fusiontype = args.fusiontype
normalize = args.normalize
location = args.location
transfer = args.transfer
base = args.base

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "CustomResNet" 

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
print("Date-Time: ", formatted_datetime)

#num_events = 1000
amount = int(num_events * 2)

if location == 'local':
    filePath_gamma="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
    filePath_proton="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
elif location == 'alex':    
    filePath_gamma = "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"
    filePath_proton="../../../../wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"
else: print("Wrong location specified!")

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

cut_nonzero = 3
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
    sum_threshold = 60 #args.threshold

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
input_shape = (41, 41, 1)
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


mean_values = np.mean(train_data,axis=(2,3))
max_values = np.amax(train_data,axis=(2,3))

mean = np.mean(mean_values)
max = np.max(max_values)


filtered_data,filtered_labels = preprocess_data(train_data,train_labels)
filtered_test_data,filtered_test_labels = preprocess_data(test_data,test_labels)

print(np.shape(filtered_data))
print(np.shape(filtered_labels))

#plot_image_2by2(train_data,4,train_labels_multishape,string="train",dt=formatted_datetime)
#plot_image_2by2(test_data,4,test_labels_multishape,string="test",dt=formatted_datetime)

from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model

input_shape = (41, 41, 1)
reg = 0.00001

#### CNN Models ###

def create_custom_resnet(inputs, start_layer='input',freeze=False):

    def inner_block(input, filters, strides=(1, 1),firstblock = False):
        shortcut = input
        if firstblock: 
            x = Conv2D(filters/8, (3, 3), strides=strides, padding='same')(input)
        else: 
            x = Conv2D(filters/4, (3, 3), strides=strides, padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters/4, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), strides=strides, padding='same')(input)
        x = BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    def outer_block(input,num_filters):
        x = inner_block(input,num_filters,firstblock=True)
        x = inner_block(x,num_filters)
        x = inner_block(x,num_filters)
        return x

    #inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7),strides = (2,2), padding='same')(inputs)
    x = MaxPooling2D(2, padding='same')(x)
    
    x = Conv2D(1024, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)

    b0 = outer_block(x, 1024)

    # Decide where to start the new branch
    if start_layer == 'b0':
        start_layer_output = b0
    elif start_layer == 'b1':
        b1 = outer_block(b0, 512)
        start_layer_output = b1
    elif start_layer == 'b2':
        b1 = outer_block(b0, 512)
        b2 = outer_block(b1, 1024)
        start_layer_output = b2
    elif start_layer == 'b3':
        b1 = outer_block(b0, 512)
        b2 = outer_block(b1, 1024)
        b3 = outer_block(b2, 2048)
        start_layer_output = b3
    else:
        print("Late fusion?")
        b1 = outer_block(b0, 512)
        b2 = outer_block(b1, 1024)
        b3 = outer_block(b2, 2048)
        start_layer_output = b3

    avgpool = GlobalAveragePooling2D()(start_layer_output)
    x = Dense(units=1024, activation='softmax')(avgpool)
    model = Model(inputs=inputs, outputs=x)

    if freeze:
        for layer in model.layers:
            layer.trainable = False

    return model

def create_single_model(model):
    inputs = model.input
    x = model.output
    x = Dropout(0.5)(x)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_single = Model(inputs,outputs)
    return model_single

def create_latefc_model(models,inputs):
    fusionlayer = concatenate([model.output for model in models])
    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    return model_multi

def create_latemax_model(models,inputs):
    fusionlayer = Lambda(lambda x: tf.reduce_max(x, axis=0), output_shape=input_shape)([model.output for model in models])
    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    return model_multi

def create_earlymax_model(models,inputs,block):

    if block == 0:
        nn1_block = 'b0'
        nn2_block = 'b1'
    elif block == 1:
        nn1_block = 'b1'
        nn2_block = 'b2'
    elif block == 2:
        nn1_block = 'b2'
        nn2_block = 'b3'    
    elif block == 3:
        nn1_block = 'b3'
        nn2_block = 'avgpool'
    else: print('Choose from b0 to b3 please')

    fusionlayer = concatenate([model.get_layer(nn1_block).ouput for model in models],axis=-1)
    secondpart_model = create_custom_resnet(input_shape,start_layer=nn2_block)
    new_output = secondpart_model(fusionlayer)
    x = Dropout(0.5)(new_output)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    return model_multi


inputs = Input(shape=input_shape)
base_cnn = create_custom_resnet(inputs)
single_view_model = create_single_model(base_cnn)

single_view_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
single_view_model.summary()
single_view_model.fit(filtered_data, filtered_labels, epochs=num_epochs, batch_size=batch_size,validation_data=(filtered_test_data,filtered_test_labels))

#base_cnn_weights = single_view_model.get_layer('max_pooling2d_6').get_weights()

single_view_model.save_weights('single_cnn_weights_partial.h5')

if transfer == 'yes':
    input_1 = Input(shape=input_shape)
    cnn_model_1 = create_custom_resnet(input_1, freeze=True)
    cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

    input_2 = Input(shape=input_shape)
    cnn_model_2 = create_custom_resnet(input_2, freeze=True)
    cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

    input_3 = Input(shape=input_shape)
    cnn_model_3 = create_custom_resnet(input_3, freeze=True)
    cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

    input_4 = Input(shape=input_shape)
    cnn_model_4 = create_custom_resnet(input_4, freeze=True)
    cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)
else: 
    input_1 = Input(shape=input_shape)
    cnn_model_1 = create_custom_resnet(input_1)
    input_2 = Input(shape=input_shape)
    cnn_model_2 = create_custom_resnet(input_2)
    input_3 = Input(shape=input_shape)
    cnn_model_3 = create_custom_resnet(input_3)
    input_4 = Input(shape=input_shape)
    cnn_model_4 = create_custom_resnet(input_4)     

if fusiontype == "latefc":
    model_multi = create_latefc_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
elif fusiontype == "latemax":
    model_multi = create_latemax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
else: print("Early Fusion not yet working! Choose latefc or latemax!")

model_multi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_multi.summary()

early_stopping_callback_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,verbose=1,mode='min')
history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping_callback_1])

str_batch_size = '{}'.format(batch_size)
str_rate = '{}'.format(rate*100)
str_reg = '{}'.format(reg)
str_num_epochs = '{}'.format(num_epochs)
str_thr = '{}'.format(sum_threshold)
str_cnz = '{}'.format(cut_nonzero)
str_transfer = '{}'.format(transfer)
str_base = '{}'.format(base)

name_str = fnr + "_" + fusiontype + "_" + normalize + "_" + str_num_epochs + "epochs" + str_batch_size + "batchsize" + str_rate + "rate" + str_reg + "reg" + str_thr + "threshold" + str_cnz + "nonzerocut" + "_" + str_transfer + "transfer_" + str_base + "basemodel_" + formatted_datetime 

print("... Finished the Fitting")

# Save the history files for later usage in other scripts

history_name = "HistoryFiles/history_" + name_str + ".pkl"

with open(history_name, 'wb') as file:
    pickle.dump(history.history, file)

# Create plots for quick overview
fig, ax = plt.subplots(1,2, figsize = (9,3))
ax[0].plot(history.history['accuracy'], label='Training Data',lw=2,c="darkorange")
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