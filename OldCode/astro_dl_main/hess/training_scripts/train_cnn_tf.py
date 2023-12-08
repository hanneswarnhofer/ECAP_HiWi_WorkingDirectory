#!/usr/bin/env python
# -*- coding: utf-8 -*-
from plotting.label import LabelPlotter
import numpy as np  # noqa
from hess.dataset import HESSLoader
from hess.hess_mappings import default_mapping
from models import training, evaluate
from hess.models import tf_cnn
from tools.utils import config
from models.tasks import Classification

CONFIG = config()
TASKS = {'primary': Classification(num_classes=2)}  # , 'axis': VectorRegression("angular")} # 'primary': Classification(num_classes=2)}  #
EPOCHS = 100

path_proton_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5"
path_gamma_new = "/home/wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5"


def prepro_fn(x):
    x = np.clip(x, 0, None)
    x = np.log10(1 + x)
    return x / 0.5


hdf_loader = HESSLoader([path_gamma_new, path_proton_new], np_prepro_fn=prepro_fn)
train_data, val_data, test_data = hdf_loader.make_image_datasets(test_split=0.0, val_split=0.0)
train_data, test_data = train_data.hess_1u_stereo(), test_data.hess_1u_stereo()

lp = LabelPlotter(train_data, "primary", log_dir=CONFIG.log_dir, class_names=["photon", "proton"], formats=["png"])
lp.plot_distribution("energy", log_x=True, bins=(0.01, 100, 300))
lp.plot_distribution("primary")


# pip.apply_cuts("true_az", ">150")
# pip.apply_cuts("true_az", "<200")
# pip.plot_test_summary_figures(CONFIG.log_dir)
# pip.plot_train_summary_figures(CONFIG.log_dir)


cnn_model = tf_cnn.get_model(train_data.feat, tasks=TASKS, stats=train_data.get_stats(), bn=True, share_ct14=True)
train_data.tf(transform=default_mapping)
# ad_val_data = [copy.deepcopy(val_data).__getattribute__("hess_1u_stereo")]
val_data.hess_1u_stereo()
val_data.tf(transform=default_mapping)

# ad_val_data = [val_data.__getattribute__(fn)() for fn in ["mono", "stereo", "hess_1u_stereo", "hybrid"]]
# val_data.tf(transform=default_mapping)

my_aiact = training.Trainer(model=cnn_model, log_dir=CONFIG.log_dir, tasks=TASKS, epochs=EPOCHS)
my_aiact.train(train_data, val_data=val_data)  # , ad_val_data=ad_val_data)

# test_datasets = []
# for i, fn in enumerate(["mono", "stereo", "hess_1u_stereo", "hybrid"]):
#     test_datasets.append(test_data.__getattribute__(fn)())
#     test_datasets[i].tf()
test_data.tf()
evaluation = evaluate.Evaluator(my_aiact.model, test_data, TASKS, log_dir=CONFIG.log_dir)
evaluation.evaluate()

# test predictions for later future studies
test_data.save_labels(CONFIG.log_dir)
