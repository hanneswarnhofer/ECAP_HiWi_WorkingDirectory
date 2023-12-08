#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import numpy as np  # noqa
from torch_geometric import transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import VectorRegression
# from plotting.style import mplrc
from swgo.models.gnn import DynEdgeConv
from swgo.data_loading import load_data

# mplrc(True)
CONFIG = config()
BATCHSIZE = 128
EPOCHS = 40
LR = 1E-3
SPARSE = True
DELTA = 0.5

data_dir = '/home/saturn/caph/mppi067h/swgo/data/'
file_list_gamma = glob.glob("%s/gamma/*.root" % data_dir)[0:800]
file_list_proton = glob.glob("%s/proton/*.root" % data_dir)[0:800]


train_data = load_data(file_list_gamma + file_list_proton)

train_data, val_data, test_data = train_data.split2train_val_test()

TASKS = [VectorRegression(train_data["shower_axis"], vec_type="angular", normalize="default")]
gcn_model = DynEdgeConv(tasks=TASKS, nb_feat=128, drop=0.3)

graph_transform = T.KNNGraph(k=6, loop=True)
train_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)
val_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)
test_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)

# data.pyg2np()

my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, decay_factor=DELTA)
my_aiact.train(train_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)


evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
evaluation.observable_dep("energy", (30, 3E6, 15), log_x=True)

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
