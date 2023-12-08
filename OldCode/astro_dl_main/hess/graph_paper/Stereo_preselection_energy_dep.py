from os.path import join
from models.tasks import Classification
from models.evaluate import Evaluator
from data.data import load_dataset_from_npz
from plotting.style import mplrc
from tools.utils import config
import numpy as np  # noqa

mplrc(True)
CONFIG = config()
saved_dir = "/home/atuin/b129dc/b129dc11/share/graph_paper/stereo"
min_val = 0.1
max_val = 300

# tag = load_dataset_from_npz(join(saved_dir, "BDT_diffuse/with_cuts.npz"), name="BDT diffuse")
bdt = load_dataset_from_npz(join(saved_dir, "BDT_point_source/with_cuts.npz"), name="BDT")
tag = load_dataset_from_npz(join(saved_dir, "BS_96_NF_150_NResNets_2_NHops_2_LR5E-3_job_1/with_cuts.npz"), name="TAGConv")
edge = load_dataset_from_npz(join(saved_dir, "train_graph_stereo_BS96_nodes96_LR1e3_delta03_drop05_resnet2_job_2/with_cuts.npz"), name="EdgeConv")

TASKS = {'primary': Classification(num_classes=2)}

# MC cut for shilon similarity
edge.cut("energy", 0.1)
tag.cut("energy", 0.1)

# Reco cut for BDT similarity
bdt.cut("energy_reco", min_val, max_val)
edge.cut("energy_reco", min_val, max_val)
tag.cut("energy_reco", min_val, max_val)

evaluation_loaded = Evaluator(None, [tag, edge, bdt], TASKS, log_dir=CONFIG.log_dir, figsize=(8, 6.5), formats=["png"])  # , "pgf"])

# #####
# Energy-dependent plot
# #####
evaluation_loaded.observable_dep("energy_reco", (0.1, 200, 15), log_x=True)
ax = evaluation_loaded.plotter["primary"]["auroc"]["all"].ax
ax.set_title("Stereo - preselection applied")
ax.set_ylim(0.85, 1.0)
ax.set_xlabel(r"$E_{\mathrm{rec}}\;/\;\mathrm{TeV}$")
ax.set_ylabel("AUROC")
ax.set_xlim(0.095, 200)
evaluation_loaded.plotter["primary"]["auroc"]["all"].save(obs="en_dep_stereo_cuts_all")
