#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np  # noqa
from hess.loading import load_hess_data
from hess.models.gnn import SparseEdgeConvStereo
from torch_geometric import transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification
from plotting.style import mplrc

mplrc(True)
CONFIG = config()
BATCHSIZE = 96
EPOCHS = 100
LR = 1E-3
DELTA = 0.5
SPARSE = True


path_gamma_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"
path_proton_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"

def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5


dc = load_hess_data([path_gamma_new , path_proton_new], mode="stereo", prepro_fn=prepro_fn)
train_data, val_data, test_data = dc.split2train_val_test()
TASKS = [Classification(train_data["primary"])]

# test_data.cut_with_label("energy", 0.1)

graph_transform = T.KNNGraph(k=6, loop=True)
train_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)
val_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)
test_data.point_clouds_to_pyg_graphs(graph_transform, add_positions=True, empty_graphs_as_single_node=True)

gcn_model = SparseEdgeConvStereo(tasks=TASKS, nb_feat=128, drop=0.5)

my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, decay_factor=DELTA)
my_aiact.train(train_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)


evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir, "with_cuts")
# test_data_no_cuts.save_labels(CONFIG.log_dir, "no_cuts")

evaluation.observable_dep("energy_reco", (0.1, 300, 21), log_x=True)
