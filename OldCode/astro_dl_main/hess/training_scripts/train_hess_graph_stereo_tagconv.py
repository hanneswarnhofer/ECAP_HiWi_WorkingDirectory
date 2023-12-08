#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.models.gnn import SparseTAGConvStereo
import torch_geometric.transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification, Regression, VectorRegression
from plotting.style import mplrc

mplrc(True)

CONFIG = config()
BATCHSIZE = 96
NFEAT = 150
NRESNETS = 2
NHOPS = 2
EPOCHS = 200
TASKS = {'primary': Classification(num_classes=2), 'energy': Regression(),
         'axis': VectorRegression("angular"), 'impact': VectorRegression("distance")}
MAKE_SPARSE = True
LR = 5E-3
MAX_OFFSET = 72.5  # deg
MIN_OFFSET = 67.5  # deg
LOSS_WEIGHTS = {"primary": 1. / 0.68, "energy": 1. / 313., "axis": 1. / 1.7E-3, "impact": 1. / 4.E3}

path_proton = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"
path_gamma = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5


hdf_loader = HESSLoader([path_proton, path_gamma], np_prepro_fn=prepro_fn)
train_data, val_data, test_data = hdf_loader.make_graph_datasets(sparse=MAKE_SPARSE)
train_data, val_data = train_data.hess_1u_stereo(), val_data.hess_1u_stereo()

train_data.torch_geometric(graph_transform=T.KNNGraph(k=6))
val_data.torch_geometric(graph_transform=T.KNNGraph(k=6))

gcn_model = SparseTAGConvStereo(tasks=TASKS, nb_inputs=3, nb_feat=NFEAT, nb_hops=NHOPS, nb_resnets=NRESNETS, drop=0.5)
my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS,
                            epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, loss_weights=LOSS_WEIGHTS)
my_aiact.train(train_data, val_data)

test_data.cut("energy", 0.1)
test_data_no_cuts = copy.deepcopy(test_data)
test_data_no_cuts.hess_1u_stereo()
test_data_no_cuts.name = "test data no cuts"
test_data_no_cuts.torch_geometric(graph_transform=T.KNNGraph(k=6))

# only local dist cut
test_data.hess_1u_stereo(max_loc_dist=0.525)
test_data.torch_geometric(graph_transform=T.KNNGraph(k=6))

evaluation = evaluate.Evaluator(my_aiact.model, [test_data, test_data_no_cuts], TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir, "with_cuts")
test_data_no_cuts.save_labels(CONFIG.log_dir, "no_cuts")
evaluation.observable_dep("energy_reco", (0.1, 300, 21), log_x=True)
