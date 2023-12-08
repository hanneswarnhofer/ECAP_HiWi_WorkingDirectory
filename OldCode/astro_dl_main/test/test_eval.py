#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # noqa
from hess.hess_mappings import default_mapping
from hess.dataset import HESSLoader
from hess.models import tf_cnn
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification, Regression, VectorRegression


CONFIG = config()
BATCHSIZE = 128
EPOCHS = 100
TASKS = {'primary': Classification(num_classes=2), 'energy': Regression(), 'axis': VectorRegression("angular")}


# f = DataContainer((feat, labels))
# f.dtype
# f_ = HessDataContainer((feat, labels))
# f_.dtype

path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"
path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"

hdf_loader = HESSLoader([path_proton, path_gamma])
train_data, val_data, test_data = hdf_loader.make_image_datasets()

val_data.tf(transform=default_mapping)
train_data.tf(transform=default_mapping)
cnn_model = tf_cnn.get_model(train_data.feat, tasks=TASKS, stats=train_data.get_stats(), bn=True, share_ct14=True)

my_aiact = training.Trainer(model=cnn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE)
my_aiact.train(val_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)

evaluation = evaluate.Evaluator(my_aiact.model, val_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
