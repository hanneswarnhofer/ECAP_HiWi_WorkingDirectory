#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.models.torch_cnn import StereoCNN
from models import training, evaluate
from tools.utils import config
from models.tasks import Regression

CONFIG = config()
TASKS = {'energy': Regression()}
EPOCHS = 20

path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5


hdf_loader = HESSLoader(path_proton, np_prepro_fn=prepro_fn)
train_data, val_data, test_data = hdf_loader.make_image_datasets(test_split=0.15, val_split=0.05)
# mask CT5 in hybrid events and apply stereo trigger requirement (at east 2 CT1-4 triggered)
train_data, val_data, test_data = train_data.hess_1u_stereo(), val_data.hess_1u_stereo(), test_data .hess_1u_stereo()

cnn_model = StereoCNN(tasks=TASKS)

# convert numpy data to torch dataset
train_data.torch()
val_data.torch()
test_data.torch()

my_aiact = training.Trainer(model=cnn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=64)
my_aiact.train(train_data, val_data)

evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
evaluation.observable_dep("energy", (0.5, 300, 30), log_x=True)

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
