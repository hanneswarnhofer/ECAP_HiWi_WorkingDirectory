# Processing HESS data for usage in CNN code from MoDA project
# Author: Hannes Warnhofer
# hannes.warnhofer@fau.de

import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


filePath_gamma="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
data_g = tables.open_file(filePath_gamma, mode="r")

print("Successfully opened gamma data!")
print(data_g)

filePath_proton="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
data_p = tables.open_file(filePath_proton, mode="r")

print("Successfully opened proton data!")
print(data_p)
print("Try again with DL1DataLoader:")

reader_g = DL1DataReader([filePath_gamma])
print("Sucessfully opended gamma data with DataLoader!")
print(reader_g)

reader_p = DL1DataReader([filePath_proton])
print("Sucessfully opended proton data with DataLoader!")
print(reader_p)