#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.image_mapper import ImageMapper
from hess.config.config import make_hess_geometry
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  # noqa
from hess.dataset import HESSLoader
import torch_geometric.transforms as T
from tools.utils import config
from plotting.style import mplrc


path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"
path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"
# path_proton = "/home/wecapstor1/caph/mppi111h/dnn/proton/hybrid/phase2d3_proton_hybrid_postselect_20deg_0deg_noZBDT_noLocDist.h5"
# path_gamma = "/home/wecapstor1/caph/mppi111h/dnn/gamma_diffuse/hybrid/phase2d3_gamma_diffuse_hybrid_postselect_20deg_0deg_noZBDT_noLocDist.h5"


mplrc(True)
CONFIG = config()
idx = 124


def prepro_fn(x):
    mask = x < 0
    x = np.log10(1 + np.abs(x))
    x[mask] = -x[mask]
    return x / x.std()


hdf_loader = HESSLoader([path_proton, path_gamma], np_prepro_fn=prepro_fn)
np.random.seed(1)
train_data, val_data, test_data = hdf_loader.make_graph_datasets(sparse=False)
val_data = val_data.mono()
val_data.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=False))

idx = 44
# idx = 124

g = val_data()[idx]
G = nx.Graph()
G.add_nodes_from(range(g["ct5"].shape[0]))

for i, (u, v) in enumerate(g["ei_pos_ct5"].T.tolist()):
    G.add_edge(u, v)

node_size = np.array(120 + 100 * g["ct5"][:, 0].numpy())
node_color = node_size + 100
mask = g["ct5"][:, 0].numpy() == 0

# pos = np.tile(g["pos_ct5"].numpy(), (6, 1))
pos = g["pos_ct5"].numpy()

fig, ax = plt.subplots(1, figsize=(24, 24))
nx.draw_networkx(G, node_size=node_size, node_color=node_color, pos=pos, with_labels=False, cmap="Reds", width=0.25, vmin=0)
nx.draw_networkx(G, node_size=node_size * mask, node_color="grey", pos=pos, with_labels=False, edgelist=[])
ax.axis('off')

plt.tight_layout()
fig.savefig(CONFIG.log_dir + "/graph_fixed.png", dpi=240)
fig.savefig(CONFIG.log_dir + "/graph_fixed.pgf")

fig, ax = plt.subplots(1, figsize=(24, 24))
# ax.scatter(pos[:, 0], pos[:, 1], s=node_size, cmap="Reds")
# ax.scatter(pos[:, 0]*mask, pos[:, 1]*mask, s=node_size*mask, cmap="Reds")
nx.draw_networkx(G, node_size=node_size, edgelist=[], node_color=node_color, pos=pos, with_labels=False, cmap="Reds", vmin=0)
nx.draw_networkx(G, node_size=node_size * mask, edgelist=[], node_color="grey", pos=pos, with_labels=False)
ax.axis('off')
plt.tight_layout()
fig.savefig(CONFIG.log_dir + "/triggered_pixels.png", dpi=240)
fig.savefig(CONFIG.log_dir + "/triggered_pixels.pgf")

# #####################
# VAL DATA Sparse
# #####################
np.random.seed(1)
train_data_sparse, val_data_sparse, test_data_sparse = hdf_loader.make_graph_datasets(sparse=True)
val_data_sparse = val_data_sparse.mono()
val_data_sparse.torch_geometric(graph_transform=T.KNNGraph(k=6, loop=False))


g_sparse = val_data_sparse()[idx]
G_sparse = nx.Graph()
G_sparse.add_nodes_from(range(g_sparse["ct5"].shape[0]))


for i, (u, v) in enumerate(g_sparse["ei_pos_ct5"].T.tolist()):
    G_sparse.add_edge(u, v)

node_size_sparse = np.array(100 + 100 * g_sparse["ct5"][:, 0].numpy())
node_color_sparse = node_size_sparse + 100

# pos_sparse = np.tile(g_sparse["pos_ct5"].numpy(), (6, 1))
pos_sparse = g_sparse["pos_ct5"].numpy()

fig, ax = plt.subplots(1, figsize=(24, 24))
nx.draw_networkx(G_sparse, node_size=3 * node_size_sparse, node_color=node_color_sparse, pos=pos_sparse, with_labels=False, cmap="Reds", width=0.75)
# ax.set_aspect(1)
ax.axis('off')
plt.tight_layout()
fig.savefig(CONFIG.log_dir + "/graph_dynamic.png", dpi=240)
fig.savefig(CONFIG.log_dir + "/graph_dynamic.pgf")


m = (val_data()[0]["ct5"] > 0).squeeze()
x = val_data()[0]["pos_ct5"][m]
y = val_data_sparse()[0]["pos_ct5"]


# #####################
# VAL DATA Image
# #####################
np.random.seed(1)
train_data_image, val_data_image, test_data_image = hdf_loader.make_image_datasets()
val_data_image = val_data_image.mono()

geo_ct14, geo_ct5 = make_hess_geometry()
f = hdf_loader.dms[0].get_h5_file()
ct_5_mapper = ImageMapper(camera_types=["FlashCam"], pixel_positions={"FlashCam": geo_ct5.get_pix_pos()}, mapping_method={"FlashCam": "axial_addressing"})


test_img_ct5 = f["dl1/event/telescope/images/tel_005"][1][3][:, np.newaxis]
test_img_ct5 = 5 * np.ones(geo_ct5.pix_id.shape[0])[:, np.newaxis]
image_ct5 = ct_5_mapper.map_image(test_img_ct5, "FlashCam") - 5


img = np.copy(val_data_image()[0]["ct5"][idx])
img[img > 0] = 2.5 * img[img > 0] + 1

image = img.squeeze() + image_ct5[:, :, 0]
fig, ax = plt.subplots(1, figsize=(24, 24))
ax.imshow(image, cmap='RdGy_r', vmin=-15, vmax=15)
plt.xticks([])
plt.yticks([])
ax.axis('off')
plt.tight_layout()

fig.savefig(CONFIG.log_dir + "/image.png", dpi=240)
fig.savefig(CONFIG.log_dir + "/image.pgf")
