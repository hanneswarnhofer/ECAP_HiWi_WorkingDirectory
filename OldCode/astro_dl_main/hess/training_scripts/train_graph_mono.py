#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.models.gnn import SparseEdgeConvMono
from torch_geometric import transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification
from plotting.style import mplrc

mplrc(True)
CONFIG = config()
BATCHSIZE = 128
EPOCHS = 120
LR = 1E-3
DELTA = 0.33
TASKS = {'primary': Classification(num_classes=2)}
SPARSE = True

path_gamma_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_mono_v2.h5"
path_proton_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_mono_v2.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5

# def prepro_fn(x):
#     mask = x < 0
#     x = np.log10(1 + np.abs(x))
#     x[mask] = -x[mask]
#     return x / x.std()


hdf_loader = HESSLoader([path_proton_new, path_gamma_new], np_prepro_fn=prepro_fn)
train_data, val_data, test_data = hdf_loader.make_graph_datasets(sparse=SPARSE)
train_data, val_data = train_data.mono(), val_data.mono()
train_data.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=True))
val_data.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=True))

train_data.n_samples + val_data.n_samples + test_data.n_samples
gcn_model = SparseEdgeConvMono(tasks=TASKS, nb_feat=96, drop=0.5)  # DummyGCN

my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE, lr=LR, decay_factor=DELTA)
my_aiact.train(train_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)


test_data_no_cuts = copy.deepcopy(test_data)
test_data_no_cuts.mono()
test_data_no_cuts.name = "test data no cuts"
test_data_no_cuts.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=True))

test_data.mono(max_loc_dist=0.72)
test_data.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=True))

evaluation = evaluate.Evaluator(my_aiact.model, [test_data, test_data_no_cuts], TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir, "with_cuts")
test_data_no_cuts.save_labels(CONFIG.log_dir, "no_cuts")

evaluation.observable_dep("energy_reco", (0.01, 300, 21), log_x=True)
