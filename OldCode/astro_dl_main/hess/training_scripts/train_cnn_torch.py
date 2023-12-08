#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.models.torch_cnn import DummyCNN
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification, Regression, VectorRegression

CONFIG = config()
TASKS = {'primary': Classification(num_classes=2), 'energy': Regression(), 'axis': VectorRegression("angular")}
EPOCHS = 10

path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"
path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5


hdf_loader = HESSLoader([path_proton, path_gamma], np_prepro_fn=lambda x: x)
train_data, val_data, test_data = hdf_loader.make_image_datasets(test_split=0.15, val_split=0.05)

cnn_model = DummyCNN(tasks=TASKS)

train_data.torch()
val_data.torch()

my_aiact = training.Trainer(model=cnn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=64)
my_aiact.train(train_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)

test_datasets = []
for i, fn in enumerate(["mono"]):  # , "stereo", "hess_1u_stereo", "hybrid"]):
    test_datasets.append(test_data.__getattribute__(fn)())
    test_datasets[i].torch()

evaluation = evaluate.Evaluator(my_aiact.model, test_datasets, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
evaluation.observable_dep("energy", (0.01, 300, 30), log_x=True)

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
