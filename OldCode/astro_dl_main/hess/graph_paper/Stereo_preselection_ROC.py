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


def make_plot(bdt, tag, edge, min_val, max_val):
    TASKS = {'primary': Classification(num_classes=2)}

    # MC cut for shilon similarity
    edge.cut("energy", 0.1)
    tag.cut("energy", 0.1)

    # Reco cut for BDT similarity
    bdt.cut("energy_reco", min_val, max_val)
    edge.cut("energy_reco", min_val, max_val)
    tag.cut("energy_reco", min_val, max_val)

    evaluation_loaded = Evaluator(None, [tag, edge, bdt], TASKS, log_dir=CONFIG.log_dir, figsize=(8, 6.5), formats=["png"])  # , "pgf"])
    evaluation_loaded.evaluate()
    ax = evaluation_loaded.plotter["primary"]["auroc"]["all"].ax
    ax.set_title("Stereo - preselection applied (%0.1f TeV - %0.0f TeV)" % (min_val, max_val))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.2)

    evaluation_loaded.plotter["primary"]["auroc"]["all"].save(obs="stereo_cuts_modified_%0.1fTeV_%0.0fTeV" % (min_val, max_val))


bdt_low = load_dataset_from_npz(join(saved_dir, "BDT_point_source/with_cuts.npz"), name="BDT")
tag_low = load_dataset_from_npz(join(saved_dir, "BS_96_NF_150_NResNets_2_NHops_2_LR5E-3_job_1/with_cuts.npz"), name="TAGConv")
edge_low = load_dataset_from_npz(join(saved_dir, "train_graph_stereo_BS96_nodes96_LR1e3_delta03_drop05_resnet2_job_2/with_cuts.npz"), name="EdgeConv")
make_plot(bdt_low, tag_low, edge_low, 0.1, 1.)


bdt_high = load_dataset_from_npz(join(saved_dir, "BDT_point_source/with_cuts.npz"), name="BDT")
tag_high = load_dataset_from_npz(join(saved_dir, "BS_96_NF_150_NResNets_2_NHops_2_LR5E-3_job_1/with_cuts.npz"), name="TAGConv")
edge_high = load_dataset_from_npz(join(saved_dir, "train_graph_stereo_BS96_nodes96_LR1e3_delta03_drop05_resnet2_job_2/with_cuts.npz"), name="EdgeConv")
make_plot(bdt_high, tag_high, edge_high, 1, 300)


bdt_all = load_dataset_from_npz(join(saved_dir, "BDT_point_source/with_cuts.npz"), name="BDT")
tag_all = load_dataset_from_npz(join(saved_dir, "BS_96_NF_150_NResNets_2_NHops_2_LR5E-3_job_1/with_cuts.npz"), name="TAGConv")
edge_all = load_dataset_from_npz(join(saved_dir, "train_graph_stereo_BS96_nodes96_LR1e3_delta03_drop05_resnet2_job_2/with_cuts.npz"), name="EdgeConv")
make_plot(bdt_all, tag_all, edge_all, 0.1, 300)
