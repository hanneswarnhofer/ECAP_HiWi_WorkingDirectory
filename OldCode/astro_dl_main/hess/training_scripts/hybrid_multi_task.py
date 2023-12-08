#!/usr/bin/env python
# -*- coding: utf-8 -*-
from plotting.style import mplrc
from models.tasks import Classification
from tools.utils import config
from models import training, evaluate
import torch_geometric.transforms as T
from hess.loading import load_hess_data
from hess.models.gnn import SparseTAGConvHybridNoTime
import copy

import numpy as np  # noqa

mplrc(False)

CONFIG = config()
BATCHSIZE = 96
NFEAT = 150
NRESNETS = 2
NHOPS = 2
EPOCHS = 200
MAKE_SPARSE = True
LR = 5E-3
MAX_OFFSET = 72.5  # deg
MIN_OFFSET = 67.5  # deg
LOSS_WEIGHTS = {"primary": 1. / 3.1} # , "energy": 1. / 727, "axis": 1. / 7.4E-3, "impact": 1. / 1.33E3}

path_proton = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"
path_gamma = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"

# path_proton = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
# path_gamma = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5

# for ev in dc():
#     a = int(ev["ct1_image"].sum()>1.) + int(ev["ct2_image"].sum()>1.) + int(ev["ct3_image"].sum()>1.) +  int(ev["ct4_image"].sum()>1.)
#     b = ev["ct5_image"].sum()
#     print(a)
#     if b<2:
#         raise NameError

    # if a==0:
    #     raise NameError
    # print(a.sum(), b.sum())    


dc = load_hess_data([path_gamma , path_proton], prepro_fn=prepro_fn)
train_data, val_data, test_data = dc.split2train_val_test()
graph_transform = T.KNNGraph(k=6, loop=True)
train_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)
val_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)

# TASKS = [Classification(train_data["primary"]), Regression(train_data["energy"]),
#        VectorRegression(train_data["shower_axis"], "angular"), VectorRegression(train_data["core"], "distance")]
TASKS = [Classification(train_data["primary"])]
gcn_model = SparseTAGConvHybridNoTime(tasks=TASKS, nb_inputs=3, nb_feat=NFEAT, nb_hops=NHOPS, nb_resnets=NRESNETS, drop=0.5)
my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS,
                            epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, loss_weights=LOSS_WEIGHTS)
my_aiact.train(train_data, val_data)

test_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)

evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir, "with_cuts")
evaluation.observable_dep("energy_reco", (0.1, 300, 21), log_x=True)
