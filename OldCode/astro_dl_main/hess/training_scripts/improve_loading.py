#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data import loading_helpers
from torch_geometric import transforms as T
from data.data import DataContainer
from hess.config.config import make_hess_geometry
import numpy as np  # noqa
from data import loader

# hdf_loader = HESSLoader([path_proton])
# hdf_loader.dms[0].walk_tree()
# f = hdf_loader.dms[0].get_h5_file()

# # h5file = open_file(fn, mode="a")
# data = pd.read_hdf(path_proton, key='/configuration/instrument/telescope/camera/geometry_1')
# data = pd.read_hdf(path_proton, key='/dl1/event/telescope/parameters/tel_001')
# data = pd.read_hdf(path_proton, key='/dl1/event/telescope/images/tel_001')


path_proton = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
path_gamma = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"


# path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"
# path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"


# h5file = tables.open_file(path_proton, "r")
# data = h5file('/dl1/event/telescope/parameters/tel_001')


# pd_f = pd.HDFStore(path_proton, mode="r")
# d = pd.read_hdf(pd_f, "/dl1/event/telescope/images/tel_001")
# d = pd.read_hdf(pd_f, "/dl1/event/telescope/parameters/tel_001")


def prepro_fn(x):
    mask = x < 0
    x = np.log10(1 + np.abs(x))
    x[mask] = -x[mask]
    return x / 0.5


f = loader.File(path_gamma)
f.walk_tree()

features_to_load = []


def rename_to_ct(x):
    return str.replace(x, "tel_00", "ct")


def rename_to_time(x):
    return str.replace(x, "peak_time", "time")


def rename_to_mc(x):
    return str.replace(x, "true", "mc")


images = loading_helpers.hdf5_group_to_structs(f, "dl1/event/telescope/images", table_keys="image", key_name_replace_fn=rename_to_ct)
features_to_load += images
times = loading_helpers.hdf5_group_to_structs(f, "dl1/event/telescope/images", table_keys="peak_time", key_name_replace_fn=rename_to_ct, table_key_replace_fn=rename_to_time)
features_to_load += times

table_keys = ['true_energy', 'true_alt', 'true_az', 'true_core_x', 'true_core_y', 'true_shower_primary_id']
labels_to_load = loading_helpers.hdf5_tables_to_structs(f, "simulation/event/subarray/shower", table_keys, transform_fns=None, table_name_replace_fn=rename_to_mc)

load_ = loader.Loader([path_gamma, path_proton])
# load.create_arr(ct1)
energy = loading_helpers.StructuredArray("simulation/event/subarray/shower", transform_fn=None, name="mc_energy", unit="", table_key="true_energy")
load_.add_features(features_to_load)
load_.add_labels(labels_to_load)
data = load_.load_data()
geo_ct14, geo_ct5 = make_hess_geometry()
ct14_pos = geo_ct14.get_pix_pos().T
ct5_pos = geo_ct5.get_pix_pos().T
config = {"pos_ct1": ct14_pos, "pos_ct2": ct14_pos, "pos_ct3": ct14_pos, "pos_ct4": ct14_pos, "pos_ct5": ct5_pos, "pos_ct14": ct14_pos}

labels, feats = data
data = feats, labels, config

dc = DataContainer(data)


def filter_fn(x):
    return (x != 0) * (x > -100)  # remove empty pixels and non-working telescopes


# Create Point Clouds
dc.make_point_cloud_data(["ct1_image", "ct1_time"], "pos_ct14", True, filter_fn)
dc.make_point_cloud_data(["ct2_image", "ct2_time"], "pos_ct14", True, filter_fn)
dc.make_point_cloud_data(["ct3_image", "ct3_time"], "pos_ct14", True, filter_fn)
dc.make_point_cloud_data(["ct4_image", "ct4_time"], "pos_ct14", True, filter_fn)
dc.make_point_cloud_data(["ct5_image", "ct5_time"], "pos_ct5", True, filter_fn)

graph_transform = T.KNNGraph(k=6, loop=True)

# dc.point_clouds_to_pyg_graphs(graph_transform, point_clouds=["ct1_pc"], exclude_keys=[], add_positions=True, empty_graphs_as_single_node=True)
dc.pyg2np()
# feat_keys = "ct1_image"
# positions = "pos_ct14"
# sparse = True
# remove_old_data = True
# name = None

# def append(self, key, data):
#     if key not in self.dict.keys():
#         data = self[key]


# """ DataContainer--> {Observable: Loader} --> {"ct1": np.array([1,2,3,4])} """
# DataContainer.add(obs)
# dc["ct1"].to_sparse()
# dc["ct1"].to_image()
# dc.apply_transform(["ct1", "ct2"], np.fn)
# dc.apply_cut("")


# class Vector(Observable):
#     def __init__(self, pre_transform, data_path, name="vector", label="", unit=""):
#         super().__init__(pre_transform, name, label, unit)
#         self.data_path = data_path


# def add_feat(data):
#     for file in file_list:
