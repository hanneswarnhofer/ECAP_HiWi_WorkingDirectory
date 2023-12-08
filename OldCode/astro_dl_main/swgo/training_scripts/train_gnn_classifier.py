#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import numpy as np  # noqa
from torch_geometric import transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification
from swgo.models.gnn import DynEdgeConv
from swgo.data_loading import load_data

# mplrc(True)
CONFIG = config()
BATCHSIZE = 128
EPOCHS = 100
LR = 1E-3
SPARSE = True
DELTA = 0.5

data_dir = '/home/saturn/caph/mppi067h/swgo/data/'
file_list_gamma = glob.glob("%s/gamma/*.root" % data_dir)  # [:500]
file_list_proton = glob.glob("%s/proton/*.root" % data_dir)  # [:500]


train_data = load_data(file_list_gamma + file_list_proton)

train_data, val_data, test_data = train_data.split2train_val_test()

TASKS = [Classification(train_data["primary"])]
gcn_model = DynEdgeConv(tasks=TASKS, nb_feat=128, drop=0.5)

graph_transform = T.KNNGraph(k=6, loop=True)
train_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)
val_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)
test_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=False)

# data.pyg2np()

my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, decay_factor=DELTA)
my_aiact.train(train_data, val_data, interactive=True)  # , val_data=val_aug, ad_val_data=ad_val_data)


evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
evaluation.evaluate(metric_names="auroc", log_x=True, log_y=True)
evaluation.observable_dep("energy", (30, 3E6, 16), log_x=True)

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
