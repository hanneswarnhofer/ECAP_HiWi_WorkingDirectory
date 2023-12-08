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