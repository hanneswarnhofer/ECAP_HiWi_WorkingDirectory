#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # noqa
from hess.dataset import HESSLoader
from models import evaluate
from tools.utils import config
from models.tasks import Classification
from plotting.style import mplrc
import glob
mplrc(False)
CONFIG = config()
TASKS = {'primary': Classification(num_classes=2)}


path_offset = glob.glob("/home/wecapstor1/caph/mppi111h/new_sims/bdt/proton_noZBDT_stereo_*")
# path_gamma_new = "/home/wecapstor1/caph/mppi111h/new_sims/bdt/gamma_diffuse_noZBDT_stereo.h5"
path_gamma_new = "/home/wecapstor1/caph/mppi111h/new_sims/bdt/gamma_pointsource_noZBDT_stereo.h5"

hdf_loader = HESSLoader(path_offset + [path_gamma_new])
_, _, test_data = hdf_loader.make_graph_datasets(sparse=True, val_split=0.0, test_split=1)

# test_data.cut("energy_reco", 0.1)
test_data.name = "BDT"

# set proton showers from [1,0] to [0, 1]
mask = (test_data.labels["true_h_first_int"] == 0).flatten()
test_data.labels["primary"][mask] = np.abs(1 - test_data.labels["primary"][mask])


def to_2d(arr):
    return np.concatenate([1 - arr, arr], axis=-1)


test_data.hess_1u_stereo(max_loc_dist=0.525)
test_data.predictions = {"primary": to_2d(test_data.labels["zeta_bdt"])}

# TASKS = {'primary': Classification(test_data.labels["primary"])}
TASKS = {'primary': Classification(num_classes=2)}

evaluation = evaluate.Evaluator(None, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()
evaluation.observable_dep("energy_reco", (0.1, 300, 21), log_x=True)

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir, "with_cuts")

# evaluation.observable_dep("energy_reco", (0.1, 300, 21), log_x=True)
