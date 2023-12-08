import numpy as np
from os.path import join
from models.tasks import Classification
from models.evaluate import Evaluator
from data.data import load_dataset_from_npz
from plotting.style import mplrc
from tools.utils import config

mplrc(True)
CONFIG = config()
saved_dir = "/home/atuin/b129dc/b129dc11/share/graph_paper/mono"
min_val = 0.05
max_val = np.inf


tag = load_dataset_from_npz(join(saved_dir, "BS_96_NF_130_NResNets_2_NHops_2_LR5E-3_job_2/test_mono.npz"), name="TAGConv")
edge = load_dataset_from_npz(join(saved_dir, "train_graph_mono_DELTA_=0.33_lr1e3_BS128_node256_job_1/no_cuts.npz"), name="EdgeConv")

TASKS = {'primary': Classification(num_classes=2)}

edge.cut("energy_reco", min_val, max_val)
tag.cut("energy_reco", min_val, max_val)

evaluation_loaded = Evaluator(None, [tag, edge], TASKS, log_dir=CONFIG.log_dir, figsize=(8, 6.5), formats=["png"])  # , "pgf"])

# #####
# Energy-dependent plot
# #####
evaluation_loaded.observable_dep("energy_reco", (0.05, 200, 18), log_x=True)
ax = evaluation_loaded.plotter["primary"]["auroc"]["all"].ax
ax.set_title("Mono - no cuts")
ax.set_ylim(0.92, 1.005)
ax.set_xlabel(r"$E_{\mathrm{rec}}\;/\;\mathrm{TeV}$")
ax.set_ylabel("AUROC")
evaluation_loaded.plotter["primary"]["auroc"]["all"].save(obs="mono_no_cuts_all")
