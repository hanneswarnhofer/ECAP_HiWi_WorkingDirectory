import numpy as np
from data.image_mapper import ImageMapper
from hess.config.config import make_hess_geometry
from data.data import DataContainer
from data.generators import HDFLoader


class HessDataContainer(DataContainer):
    def __init__(self, data, name=""):
        super().__init__(data, name)

    def mono_triggered_only_mask(self):
        ''' Return mono events, No ct1-ct14 trigger'''
        x = [self.sum(t) for t in ["ct1", "ct2", "ct3", "ct4", "ct5"]]
        ct1, ct2, ct3, ct4, ct5 = [c > 0 for c in x]
        ct14 = ct1 + ct2 + ct3 + ct4
        return ct5 * ~ct14

    def loc_dist(self, tel):
        return self.labels["%s_loc_dist" % tel]()

    def sum(self, tel):
        if self.feat[tel].dtype == object:
            return np.array([arr.sum() for arr in self.feat[tel]])
        else:
            return self.feat[tel].sum(tuple(range(1, self.feat[tel].ndim)))

    def mono_mask(self, min_loc_dist, max_loc_dist):
        '''Return mask if ct5 was triggered'''
        return (self.labels["ct5_loc_dist"]() > min_loc_dist) * (self.labels["ct5_loc_dist"]() < max_loc_dist)

    def get_ct14_loc_dist_mask(self, min_loc_dist, max_loc_dist):
        return [((self.loc_dist(t) > min_loc_dist) * (self.loc_dist(t) < max_loc_dist)) for t in ["ct1", "ct2", "ct3", "ct4"]]

    def stereo_mask(self, min_loc_dist, max_loc_dist):
        return ~self.mono_mask() * self.hess_1u_stereo_mask(min_loc_dist, max_loc_dist)

    def hybrid_mask(self, min_loc_dist_ct14, max_loc_dist_ct14, min_loc_dist_ct5, max_loc_dist_ct5):

        x = self.get_ct14_loc_dist_mask(min_loc_dist_ct14, max_loc_dist_ct14)
        ct5 = self.mono_mask(min_loc_dist_ct5, max_loc_dist_ct5).flatten()
        ct14 = np.concatenate(x, axis=-1).sum(axis=-1).astype(bool)
        return ct5 * ct14  # ct5 trigger and at least a single ct1-ct4 trigger

    def hess_1u_stereo_mask(self, min_loc_dist, max_loc_dist):
        x = self.get_ct14_loc_dist_mask(min_loc_dist, max_loc_dist)
        res = np.concatenate(x, axis=-1).sum(axis=-1)
        return res >= 2  # at least 2 single ct1-ct4 trigger (multiplicity cut used)

    def all_mask(self):
        return np.ones(self.labels["primary"].shape[0], dtype=np.bool)

    def mono(self, min_loc_dist=0, max_loc_dist=100):
        return make_mono(self, min_loc_dist, max_loc_dist)

    def stereo(self, min_loc_dist=0, max_loc_dist=100):
        return make_stereo(self, min_loc_dist, max_loc_dist)

    def hybrid(self, min_loc_dist_ct14=0, max_loc_dist_ct14=100, min_loc_dist_ct5=0, max_loc_dist_ct5=100):
        return make_hybrid(self, min_loc_dist_ct14, max_loc_dist_ct14, min_loc_dist_ct5, max_loc_dist_ct5)

    def hess_1u_stereo(self, min_loc_dist=0, max_loc_dist=100):
        return make_hess1u_stereo(self, min_loc_dist, max_loc_dist)


def select_telescopes(feat_data, ct1s_to_keep=True, ct2s_to_keep=True, ct3s_to_keep=True, ct4s_to_keep=True, ct5s_to_keep=False):
    """
    Select the telecopes that should be kept (True).

    Args:
        feat_data (DataContainer): DataContainer
        ct1s_to_keep (bool, or np.array): Mask for the CT1 telescope. Which data should be kept? Use False for masking telescope in all events . Defaults to True.
        ct2s_to_keep (bool, or np.array): Mask for the CT2 telescope. Which data should be kept? Use False for masking telescope in all events . Defaults to True.
        ct3s_to_keep (bool, or np.array): Mask for the CT3 telescope. Which data should be kept? Use False for masking telescope in all events . Defaults to True.
        ct4s_to_keep (bool, or np.array): Mask for the CT4 telescope. Which data should be kept? Use False for masking telescope in all events . Defaults to True.
        ct5s_to_keep (bool, or np.array): Mask for the CT5 telescope. Which data should be kept? Use False for masking telescope in all events . Defaults to False.
    Returns:
        _type_: _description_
    """

    for i, mask_tel in enumerate([ct1s_to_keep, ct2s_to_keep, ct3s_to_keep, ct4s_to_keep, ct5s_to_keep]):

        # mask should be the same for image and time data
        if type(mask_tel) is bool:
            mask = np.ones(feat_data["ct%i_image" % (i + 1)].shape[0], dtype=bool) * mask_tel
        else:
            mask = mask_tel
        mask = mask.squeeze()
        feat_data["ct%i_image" % (i + 1)][~mask] = np.zeros((feat_data["ct%i_image" % (i + 1)][~mask].shape))
        try:
            feat_data["ct%i_time" % (i + 1)][~mask] = np.zeros((feat_data["ct%i_time" % (i + 1)][~mask].shape))
        except KeyError:
            pass
    return feat_data


def select_data(data, events2keep, ct1s_to_keep=False, ct2s_to_keep=False, ct3s_to_keep=False, ct4s_to_keep=False, ct5s_to_keep=False, name=""):
    """
    Create data set using masking of events (e.g., multiplicity selection) and masking of single telescopes (e.g. mask CT5 for stereo events).

    Args:
        data (DataContainer): Data to process
        events2keep (np.array(bools)): Event mask to mask events that should pass the selection
        ct1s_to_keep (bool, optional np.array(bools)): Telescope mask, indicating which telescopes should be selected.
                                                       For an event-by-event basis use np.array. For masking always use bool.
                                                       Defaults to False.
        ct2s_to_keep (bool, np.array(bools)): Telescope mask, indicating which telescopes should be selected.
                                                       For an event-by-event basis use np.array. For masking always use bool.
                                                       Defaults to False.
        ct3s_to_keep (bool, np.array(bools)): Telescope mask, indicating which telescopes should be selected.
                                                       For an event-by-event basis use np.array. For masking always use bool.
                                                       Defaults to False.
        ct4s_to_keep (bool, np.array(bools)): Telescope mask, indicating which telescopes should be selected.
                                                       For an event-by-event basis use np.array. For masking always use bool.
                                                       Defaults to False.
        ct5s_to_keep (bool, np.array(bools)): Telescope mask, indicating which telescopes should be selected.
                                                       For an event-by-event basis use np.array. For masking always use bool.
                                                       Defaults to False.
        name (str, optional): Name of the new DataContainer

    Returns:
        DataContainer: Returns DataContainer with selected events and selected telescope measurements.
    """
    data.name = "%s_%s" % (data.name, name)
    # select_telescopes(data.feat, ct1s_to_keep, ct2s_to_keep, ct3s_to_keep, ct4s_to_keep, ct5s_to_keep, sparse=data.sparse)
    select_telescopes(data.feat, ct1s_to_keep, ct2s_to_keep, ct3s_to_keep, ct4s_to_keep, ct5s_to_keep)
    for k, val in data.feat.items():  # overwrite features
        if "pos" in k:
            if data.sparse is True:
                data.feat[k] = val[events2keep]
            else:
                data.feat[k] = val
        else:
            data.feat[k] = val[events2keep]

    for k, val in data.labels.items():  # overwrite labels
        data.labels[k].arr = val[events2keep]

    return data


"""
    make event types (mono, hess-1u-stereo, hybrid, stereo) using the given data:
    mono: only CT5
    hess-1u-stereo: only CT1 - CT4 (mask CT5)
    stereo: only CT1 - CT4 (CT5 on but NOT triggered)
    hybrid: CT5 and at least one CT1 - CT4
"""


def make_hybrid(data, min_loc_dist_ct14, max_loc_dist_ct14, min_loc_dist_ct5, max_loc_dist_ct5):
    data.get_ct14_loc_dist_mask(min_loc_dist_ct14, max_loc_dist_ct14)
    data.mono_mask(min_loc_dist_ct5, max_loc_dist_ct5)
    event_mask = data.hybrid_mask(min_loc_dist_ct14, max_loc_dist_ct14, min_loc_dist_ct5, max_loc_dist_ct5)
    return select_data(data, event_mask, True, True, True, True, True, name="hybrid")


def make_hess1u_stereo(data, min_loc_dist, max_loc_dist):
    data.get_ct14_loc_dist_mask(min_loc_dist, max_loc_dist)
    event_mask = data.hess_1u_stereo_mask(min_loc_dist, max_loc_dist)
    return select_data(data, event_mask, True, True, True, True, ct5s_to_keep=False, name="hess_1u_stereo")


def make_stereo(data, min_loc_dist, max_loc_dist):
    data.get_ct14_loc_dist_mask(min_loc_dist, max_loc_dist)
    event_mask = data.stereo_mask(min_loc_dist, max_loc_dist)
    return select_data(data, event_mask.stereo_mask(), True, True, True, True, name="stereo")


def make_mono(data, min_loc_dist, max_loc_dist):
    ct5 = data.mono_mask(min_loc_dist, max_loc_dist)
    return select_data(data, ct5.squeeze(), ct1s_to_keep=False, ct2s_to_keep=False, ct3s_to_keep=False, ct4s_to_keep=False, ct5s_to_keep=True, name="mono")


class Mapper():

    def __init__(self, make_test_image=False):
        print("initalize image mappers")

        geo_ct14, geo_ct5 = make_hess_geometry()

        camera_types = ["HESS-I", "FlashCam"]
        pixel_positions = {"HESS-I": geo_ct14.get_pix_pos(), "FlashCam": geo_ct5.get_pix_pos()}
        mapping_method = {k: "axial_addressing" for k in camera_types}

        self.mapper = ImageMapper(camera_types=camera_types, pixel_positions=pixel_positions,
                                  mapping_method=mapping_method)

        if make_test_image is True:
            self.plot_test_mapping()

    def map_image(self, data, camera_types):
        return self.mapper.map_image(data, camera_types)

    def map_image_back(self, data, camera_type):
        return self.mapper.map_image_back(data, camera_type)

    def plot_test_mapping(self):
        from matplotlib import pyplot as plt

        def plot_image(image, name=None):
            fig, ax = plt.subplots(1)
            ax.set_aspect(1)
            # ax.pcolor(np.flip(image[:, :, 0], axis=(0)), cmap='viridis', vmin=-5)
            ax.imshow(np.flip(image[:, :, 0], axis=(0)), cmap='viridis', vmin=-5)
            # plt.show()
            fig.savefig("./binned_image%s.png" % name)

        test_img_ct14 = np.ones(960)[:, np.newaxis]

        for i in range(1, 5):
            plot_image(self.mapper(test_img_ct14, "HESS-I")[5:, ...], name="CT%i" % i)

        test_img_ct5 = np.ones(1764)[:, np.newaxis]
        plot_image(self.mapper(test_img_ct5, "FlashCam"), name="CT5")


class HESSLoader(HDFLoader):

    def __init__(self, h5_paths, np_prepro_fn=None):
        mc_subpath = "simulation/event/subarray/shower"  # in HDF5 array
        super().__init__(h5_paths, mc_subpath, np_prepro_fn=np_prepro_fn)
        self.mapper = Mapper()

    def preprocess_labels(self, mc_dict):
        from data.prepro import ang2vec, to_one_hot
        m = mc_dict["true_shower_primary_id"] == 101.
        mc_dict["true_shower_primary_id"][m] = 1.  # proton == 1
        mc_dict["primary"] = to_one_hot(mc_dict["true_shower_primary_id"])  # photon = [1,0} / proton = [0,1]
        mc_dict["impact"] = np.concatenate([mc_dict["true_core_x"], mc_dict["true_core_y"]], axis=-1)
        mc_dict["energy"] = mc_dict["true_energy"]
        mc_dict["axis"] = ang2vec(mc_dict["true_az"][:, 0], mc_dict["true_alt"][:, 0], deg=True)
        return mc_dict

    def preprocess_feat(self, feat_dict):
        # preprocess data
        # To be replace by tf mapping function

        for k, val in feat_dict.items():
            if k not in ["ct1", "ct2", "ct3", "ct4", "ct5"]:
                continue

            if val.dtype == np.object:
                feats = np.array([self.np_prepro_fn(val_) for val_ in val], dtype=np.object)
            else:
                feats = self.np_prepro_fn(val)

            feat_dict[k] = feats

        return feat_dict

    def make_flattened_h5_images(self, telescope_images, camera_type):
        """
        Creation of hexagonal telescope images from grid images as input;
        reverts the make_image function; needed for usage of processed
        images in ctapipe.

        Inputs:
            np.array: array of indexed camera images (2D grid image)
            str: camera geometry

        Output:
            np.array: flattened array of images
        """
        return self.mapper.map_image_back(telescope_images, camera_type)

    def make_rectangular_images(self, hexagonal_images, camera_type):
        image = self.mapper.map_image(hexagonal_images.T, camera_type).T
        if image.ndim != 4:
            image = image[..., np.newaxis]
        return image

    def make_cartesian_images(self, data_manager, img_dict, make_test_image=False):
        # load IACT image data from h5 file of DataManager into RAM
        #  ct5: 44 x 44 # ct14: 36 x 36
        self.sparse = None

        if img_dict == {}:
            img_dict = {"ct%i" % k: [] for k in range(1, 6)}

        with data_manager.get_h5_file() as f:
            print("map images of file:\n", f.filename)

            for i in range(1, 6):
                x = f["dl1/event/telescope/images/tel_00%i" % i][:]
                x = np.stack(np.stack(x.tolist(), axis=0)[:, 3].tolist(), axis=0)

                if i == 5:
                    x = self.mapper.map_image(x.T, "FlashCam").T
                else:
                    x = self.mapper.map_image(x.T, "HESS-I")[5:, ...].T  # remove empty pixels

                if x.ndim != 4:
                    x = x[..., np.newaxis]
                try:
                    y = f["dl1/event/telescope/images/tel_00%i" % i][:]
                    y = np.stack(np.stack(y.tolist(), axis=0)[:, 4].tolist(), axis=0)

                    if i == 5:
                        y = self.mapper.map_image(y.T, "FlashCam").T
                    else:
                        y = self.mapper.map_image(y.T, "HESS-I")[5:, ...].T  # remove empty pixels

                    if y.ndim != 4:
                        y = y[..., np.newaxis]

                    x = np.stack([x, y], axis=-1)

                except KeyError:
                    pass

                img_dict["ct%i" % i].append(x)

            del x

        return img_dict

    def make_point_cloud_data(self, data_manager, feat_dict):
        sparse = self.sparse

        if feat_dict == {}:
            if sparse is False:
                feat_dict = {**{"ct%i" % k: [] for k in range(1, 6)},
                             # **{"ct%i_time" % k: [] for k in range(1, 6)},
                             **{"pos_ct14": [], "pos_ct5": []}}
            else:
                feat_dict = {**{"ct%i" % k: [] for k in range(1, 6)},
                             # **{"ct%i_time" % k: [] for k in range(1, 6)},
                             **{"pos_ct%i" % i: [] for i in range(1, 6)}}

        geo_ct14, geo_ct5 = make_hess_geometry()
        ct14_pos = geo_ct14.get_pix_pos()
        ct5_pos = geo_ct5.get_pix_pos()

        if sparse is False:
            # for i in range(1, 5):
            #     feat_dict["pos_ct%i" % i] = [ct14_pos.T]
            feat_dict["pos_ct14"] = [ct14_pos.T]
            feat_dict["pos_ct5"] = [ct5_pos.T]

        with data_manager.get_h5_file() as f:
            print("load point cloud data:\n", f.filename)

            for i in range(1, 6):
                x = f["dl1/event/telescope/images/tel_00%i" % i][:]

                if sparse is True:
                    for idx in range(len(x)):
                        ev = x[idx][3]
                        mask = (ev != 0) * (ev > -100)  # not triggered pixels = -999

                        feat_dict["ct%i" % i].append(ev[mask][..., np.newaxis])
                        # try:
                        #     time = x[idx][4]
                        #     feat_dict["ct%i_time" % i].append(time[mask][..., np.newaxis])
                        # except IndexError:
                        #     if "ct%i_time" % i in feat_dict.keys():
                        #         feat_dict.pop("ct%i_time" % i)

                        if i == 5:
                            feat_dict["pos_ct5"].append(ct5_pos.T[mask])
                        else:
                            feat_dict["pos_ct%i" % i].append(ct14_pos.T[mask])
                else:
                    x_ = np.stack(np.stack(x.tolist(), axis=0)[:, 3].tolist(), axis=0)
                    feat_dict["ct%i" % i].append(x_[..., np.newaxis])
                    try:
                        x_ = np.stack(np.stack(x.tolist(), axis=0)[:, 4].tolist(), axis=0)
                        feat_dict["ct%i_time" % i].append(x_[..., np.newaxis])
                    except IndexError:
                        if "ct%i_time" % i in feat_dict.keys():
                            feat_dict.pop("ct%i_time" % i)
                        pass

        return feat_dict

    def make_graph_datasets(self, sparse, test_split=0.15, val_split=0.05):
        self.sparse = sparse

        def feat_fn(x, feat_dict):
            return self.make_point_cloud_data(x, feat_dict)
        return self.make_training_data(feat_fn, test_split, val_split)

    def make_image_datasets(self, test_split=0.15, val_split=0.05):
        feat_fn = self.make_cartesian_images
        return self.make_training_data(feat_fn, test_split, val_split)

    def make_training_data(self, feat_fn, test_split=0.15, val_split=0.05):
        feat_dict, mc_dict = {}, {}
        assert test_split + val_split <= 1, "splits have to be smaller one"

        for dm in self.dms:
            feat_dict = feat_fn(dm, feat_dict)
            mc_dict = self.make_mc_data(dm, mc_dict)

        # {k: np.array(val, dtype=object) for k, val in feat_dict.items()}
        print("stacking data")

        if self.sparse is True:
            feat_dict = {k: np.array(val, dtype=object) for k, val in feat_dict.items()}  # for sparse graphs (all different shapes)
        else:
            feat_dict = {k: np.concatenate(val) for k, val in feat_dict.items() if len(val) > 0}  # for images (all same shapes)

        for k, val in mc_dict.items():
            val = np.concatenate(val, axis=0)

            # add dummy axis for len(1) arrs to ensure np.arrays when indexing arrs
            if len(val.shape) == 1:
                val = val[:, np.newaxis]

            mc_dict[k] = val

        feat_dict = self.preprocess_feat(feat_dict)
        mc_dict = self.preprocess_labels(mc_dict)
        nsamples = mc_dict["primary"].shape[0]

        idx_train, idx_test, idx_val = self.get_train_val_test_indices(nsamples, test_split, val_split)
        feat_dict_train, feat_dict_val, feat_dict_test = self.split(feat_dict, idx_train, idx_val, idx_test)
        mc_dict_train, mc_dict_val, mc_dict_test = self.split(mc_dict, idx_train, idx_val, idx_test)
        feat_dict, mc_dict = {}, {}  # maybe memory overhead!
        self.train, self.valid, self.test = HessDataContainer((feat_dict_train, mc_dict_train), name="train", sparse=self.sparse), HessDataContainer((feat_dict_val, mc_dict_val), name="valid", sparse=self.sparse), HessDataContainer((feat_dict_test, mc_dict_test), name="test", sparse=self.sparse)

        return self.train, self.valid, self.test
