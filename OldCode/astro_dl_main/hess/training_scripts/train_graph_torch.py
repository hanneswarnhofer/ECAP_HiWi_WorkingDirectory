#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.models.gnn import TagGCNMonoDeep
import torch_geometric.transforms as T
from models import training, evaluate
from tools.utils import config
from models.tasks import Classification, Regression, VectorRegression

CONFIG = config()
BATCHSIZE = 256
EPOCHS = 100
TASKS = {'primary': Classification(num_classes=2), 'energy': Regression(), 'axis': VectorRegression("angular")}
hess_config = "hybrid"  # "hess_1u_stereo "mono"
make_sparse = False

path_proton = "/home/wecapstor1/caph/mppi111h/phase2d3_proton_hybrid_postselect_20deg_0deg_noZBDT_noLocDist.h5"
path_gamma = "/home/wecapstor1/caph/mppi111h/phase2d3_gamma_diffuse_hybrid_postselect_20deg_0deg_noZBDT_noLocDist.h5"

hdf_loader = HESSLoader([path_proton, path_gamma])  # , prepro_fn=lambda x: x)
train_data, val_data, test_data = hdf_loader.make_graph_datasets(sparse=make_sparse)

train_data = train_data.__getattribute__(hess_config)()
train_data.torch_geometric(graph_transform=T.KNNGraph(k=6))
val_data = val_data.__getattribute__(hess_config)()
val_data.torch_geometric(graph_transform=T.KNNGraph(k=6))

gcn_model = TagGCNMonoDeep(tasks=TASKS)  # DummyGCN

my_aiact = training.Trainer(model=gcn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS, batch_size=BATCHSIZE)
my_aiact.train(train_data, val_data)  # , val_data=val_aug, ad_val_data=ad_val_data)

test_datasets = []
for i, fn in enumerate([hess_config]):
    test_datasets.append(test_data.__getattribute__(fn)())
    test_datasets[i].torch_geometric(graph_transform=T.KNNGraph(k=6))

evaluation = evaluate.Evaluator(my_aiact.model, test_datasets, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
