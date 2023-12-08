import numpy as np
import h5py
import os
from ctapipe.instrument.camera import CameraGeometry
from ctapipe.io import EventSource

# Define some constants
NUM_EVENTS = 100
NUM_TELESCOPES = 4
IMAGE_SIZE = (41, 41)

#DATA_DIR = "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/"
#FILENAME_GAMMA = "gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"
#FILENAME_PROTON = "proton_noZBDT_noLocDist_hybrid_v2.h5"

DATA_DIR = "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/"
FILENAME_GAMMA = "phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
FILENAME_PROTON = "phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"

THRESHOLD = 0.0001

# Load telescope data
def load_telescope_data(filename):
    with h5py.File(filename, "r") as f:
        telescope_data = f["dl1/event/telescope/images"]
        selected_data = telescope_data[:NUM_EVENTS, :, 3, :, np.newaxis]
    return selected_data
tel_data_gamma = load_telescope_data(os.path.join(DATA_DIR, FILENAME_GAMMA))
tel_data_proton = load_telescope_data(os.path.join(DATA_DIR, FILENAME_PROTON))

# Combine and shuffle the data
tel_data = np.concatenate((tel_data_gamma, tel_data_proton), axis=0)
labels = np.concatenate((np.ones(NUM_EVENTS), np.zeros(NUM_EVENTS)))

shuffle_indices = np.arange(len(tel_data))
np.random.shuffle(shuffle_indices)
tel_data = tel_data[shuffle_indices]
labels = labels[shuffle_indices]

# Load camera geometry
geo_data = np.load("geometry2d3.npz")
pix_positions = geo_data["ct14_geo"]
geo_ct14 = CameraGeometry.from_pixel_positions(pix_positions)

# Map images and apply threshold
def map_images(tel_data):
    mapped_images = np.zeros((len(tel_data), NUM_TELESCOPES, *IMAGE_SIZE))
    for tel_num in range(NUM_TELESCOPES):
        tel_data_tel = tel_data[:, tel_num]
        tel_mapped = geo_ct14.map_image(tel_data_tel, "HESS-I")
        tel_mapped[tel_mapped < THRESHOLD] = 0
        mapped_images[:, tel_num] = tel_mapped
    return mapped_images

mapped_images = map_images(tel_data)

# Split data into training and test sets
split_idx = int(len(mapped_images) * 0.8)
train_data, test_data = mapped_images[:split_idx], mapped_images[split_idx:]
train_labels, test_labels = labels[:split_idx], labels[split_idx:]

# Printing shapes for confirmation
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)