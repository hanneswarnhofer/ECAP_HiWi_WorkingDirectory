import numpy as np  # noqa
from os.path import join
from models.tasks import Classification
from models.evaluate import Evaluator
from data.data import load_dataset_from_npz
from plotting.style import mplrc
from tools.utils import config

mplrc(True)
CONFIG = config()
saved_dir = "/home/atuin/b129dc/b129dc11/share/graph_paper/mono"


def make_plot(tag, edge, min_val, max_val):

    TASKS = {'primary': Classification(num_classes=2)}

    edge.cut("energy_reco", min_val, max_val)
    tag.cut("energy_reco", min_val, max_val)

    evaluation_loaded = Evaluator(None, [tag, edge], TASKS, log_dir=CONFIG.log_dir, figsize=(8, 6.5), formats=["png"])  # , "pgf"])
    evaluation_loaded.evaluate()
    evaluation_loaded.plotter["primary"]["auroc"]["all"].set_xscale("log")
    evaluation_loaded.plotter["primary"]["auroc"]["all"].set_yscale("log")
    ax = evaluation_loaded.plotter["primary"]["auroc"]["all"].ax
    ax.set_title("Mono - no cuts (%s TeV - %s TeV)" % (str(min_val), str(max_val)))
    ax.set_ylim(0.01, 1.2)

    evaluation_loaded.plotter["primary"]["auroc"]["all"].save(obs="mono_no_cuts_modified_%sTeV_%sTeV" % (str(min_val), str(max_val)))


tag_low = load_dataset_from_npz(join(saved_dir, "BS_96_NF_130_NResNets_2_NHops_2_LR5E-3_job_2/test_mono.npz"), name="TAGConv")
edge_low = load_dataset_from_npz(join(saved_dir, "train_graph_mono_DELTA_=0.33_lr1e3_BS128_node256_job_1/no_cuts.npz"), name="EdgeConv")
make_plot(tag_low, edge_low, 0.05, 1.)

tag_high = load_dataset_from_npz(join(saved_dir, "BS_96_NF_130_NResNets_2_NHops_2_LR5E-3_job_2/test_mono.npz"), name="TAGConv")
edge_high = load_dataset_from_npz(join(saved_dir, "train_graph_mono_DELTA_=0.33_lr1e3_BS128_node256_job_1/no_cuts.npz"), name="EdgeConv")
make_plot(tag_high, edge_high, 1, 300)

tag_all = load_dataset_from_npz(join(saved_dir, "BS_96_NF_130_NResNets_2_NHops_2_LR5E-3_job_2/test_mono.npz"), name="TAGConv")
edge_all = load_dataset_from_npz(join(saved_dir, "train_graph_mono_DELTA_=0.33_lr1e3_BS128_node256_job_1/no_cuts.npz"), name="EdgeConv")
make_plot(tag_all, edge_all, 0.05, 300)
