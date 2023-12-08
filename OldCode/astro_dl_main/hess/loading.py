#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.data import Array
from hess.dataset import HessDataContainer
from hess.config.config import make_hess_geometry
import numpy as np  # noqa
from data import loader, loading_helpers
from data.prepro import to_one_hot, ang2vec
path_proton = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"
path_gamma = "/home/wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"

paths = [path_proton, path_gamma]


def load_hess_data(paths, time=False, mode="hybrid", prepro_fn=None):


    example_file = loader.File(paths[0])
    example_file.walk_tree()

    features_to_load = []

    def rename_to_ct(x):
        return str.replace(x, "tel_00", "ct")

    def rename_to_time(x):
        return str.replace(x, "peak_time", "time")

    def rename_to_locdist(x):
        return str.replace(x, "r", "loc_dist")

    def rename_to_mc(x):
        return str.replace(x, "true", "mc")

    images = loading_helpers.hdf5_group_to_structs("dl1/event/telescope/images", file=example_file, table_keys="image", key_name_replace_fn=rename_to_ct, transform_fns=prepro_fn)
    features_to_load += images
    # times = loading_helpers.hdf5_group_to_structs("dl1/event/telescope/images", file=example_file, table_keys="peak_time", key_name_replace_fn=rename_to_ct, table_key_replace_fn=rename_to_time)
    # features_to_load += times

    table_keys = ['true_energy', 'true_alt', 'true_az', 'true_core_x', 'true_core_y', 'true_shower_primary_id']
    labels_to_load = loading_helpers.hdf5_tables_to_structs("simulation/event/subarray/shower", table_keys, transform_fns=None, table_name_replace_fn=rename_to_mc)
    locdists = loading_helpers.hdf5_group_to_structs("dl1/event/telescope/parameters", file=example_file, table_keys="r", key_name_replace_fn=rename_to_ct, table_key_replace_fn=rename_to_locdist)
    labels_to_load += locdists
    labels_to_load += loading_helpers.hdf5_tables_to_structs("dl1/event/telescope/other_parameters", table_keys="energy_reco", transform_fns=None)
    # labels_to_load += loading_helpers.hdf5_tables_to_structs("dl1/event/telescope/other_parameters", file=example_file, table_keys="event_id", key_name_replace_fn=rename_to_ct, transform_fns=prepro_fn)


    load_ = loader.Loader(paths)
    load_.add_features(features_to_load)
    load_.add_labels(labels_to_load)
    data = load_.load_data()

    def get_hess_config():
        geo_ct14, geo_ct5 = make_hess_geometry()
        ct14_pos = geo_ct14.get_pix_pos().T
        ct5_pos = geo_ct5.get_pix_pos().T
        return {"pos_ct1": ct14_pos, "pos_ct2": ct14_pos, "pos_ct3": ct14_pos, "pos_ct4": ct14_pos, "pos_ct5": ct5_pos, "pos_ct14": ct14_pos}

    labels, feats = data

    def get_primary(labels):
        labels['primary'] = labels.pop('mc_shower_primary_id')
        labels['primary'].name = 'primary'
        prims = labels['primary'].arr
        prims[prims == 101] = 1.
        labels['primary'].arr = prims
        labels['primary'].arr = to_one_hot(labels['primary'](), num_classes=2)
        return labels

    def get_shower_axis(labels):
        shower_axis = Array(ang2vec(labels["mc_az"](), labels["mc_alt"](), deg=True).squeeze(), name="shower_axis")
        labels[shower_axis.name] = shower_axis
        del labels["mc_az"], labels["mc_alt"]
        return labels

    def get_core(labels): 
        core = Array(np.stack([labels["mc_core_x"](), labels["mc_core_y"]()], axis=-1).squeeze(), name="core")
        labels[core.name] = core
        del labels['mc_core_x'], labels['mc_core_y']
        return labels

    def get_energy(labels):
        labels['energy'] = labels.pop('mc_energy')
        labels['energy'].name = 'energy'
        return labels

    # convert the lables to the form we want in trainings
    get_primary(labels), get_shower_axis(labels), get_core(labels), get_energy(labels)

    data = feats, labels, get_hess_config()
    dc = HessDataContainer(data)

    if mode == "stereo":
        print("make stereo data")
        dc.hess_1u_stereo()
    elif mode == "hybrid":
        print("make hybrid data")
        dc.hybrid()
    else:
        assert("mode: %s not valid" % mode)

    # Make PointCloud

    def filter_fn(x):
        return (x != 0) * (x > -100)  # remove empty pixels and non-working telescopes

    # Create Point Clouds
    if time:
        dc.make_point_cloud_data(["ct1_image", "ct1_time"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct2_image", "ct2_time"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct3_image", "ct3_time"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct4_image", "ct4_time"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct5_image", "ct5_time"], "pos_ct5", True, filter_fn)
    else:
        dc.make_point_cloud_data(["ct1_image"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct2_image"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct3_image"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct4_image"], "pos_ct14", True, filter_fn)
        dc.make_point_cloud_data(["ct5_image"], "pos_ct5", True, filter_fn)

    return dc
