import numpy as np
from functools import partial
from tools.progress import Progbar
from tools.utils import is_interactive
from models.evaluate import Predictor
from plotting.base import PlottingFactory
from tools.utils import to_list


class DataObject():

    def __init__(self, name='') -> None:
        self.name = name

    def __len__(self):
        return np.unique([val for val in self.dict.values()])

    def __getitem__(self, key):
        return self.dict[key]

    def get_empty_copy(self):
        """Make empty (light_weight) copy of DataObject

        Returns:
            _type_: _description_
        """
        class_kwargs = self.__dict__
        empty_dicts = {}
        for k, val in class_kwargs.items():
            if type(val) == dict:
                empty_dicts[k] = {k_: [] for k_ in val.keys()}
            elif type(val) == np.ndarray:
                empty_dicts[k] = np.array([])
            elif type(val) == list:
                empty_dicts[k] = []

        class_kwargs = {**class_kwargs, **empty_dicts}  # overwrite dict
        return self.__class__(**class_kwargs)

    @property
    def dict(self):
        """Get dictionary of data.

        Returns:
            dict: Dictionary of the stored data
        """
        d = {k: val for k, val in self.__dict__.items() if k not in ["name", "unit", "label"]}
        return d


class Array(DataObject):

    def __init__(self, arr, name='', unit=None, label="") -> None:
        super().__init__(name)
        self.arr = arr
        self.unit = unit
        self.label = label

    def __repr__(self):
        return "DataObject: %s\n - shape: %s\n - inital unit (might changed after transformation)=%s" % (self.name, self.shape, self.unit)

    def __call__(self):
        if self.arr.ndim == 1:
            return self.arr[:, np.newaxis]
        return self.arr

    @property
    def dict(self):
        return {self.name: self.arr}

    @property
    def dtype(self):
        return self().dtype

    @property
    def features(self):
        return {"name": self.arr}

    def append(self, arr):
        self.arr = np.append(self.arr, arr)

    @property
    def shape(self):
        return self().shape

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self()[idx]

    def __setitem__(self, idx, value):
        if len(self.arr) == 0:
            self.arr = value
        else:
            self.arr[idx] = value


class Image(Array):

    def __init__(self, image, name="", label_x=None, label_y=None) -> None:
        super().__init__(name)
        self.label_x = label_x
        self.label_y = label_y
        self.arr = self.stack_channels()

        # if type(image) == dict:
        #     self.arr = image
        # elif type(image) == tuple:
        #     self.arr = {"channel_%i" % i: val for i, val in enumerate(image)}
        # else:

    @property
    def channels(self):
        c = self.arr.shape[-1]

        if c == len(self):
            c = 1

        return c

    def stack_channels(self, images):
        return np.stack([c for c in images.values()], axis=-1)


class PointCloud(DataObject):
    def __init__(self, positions=None, features=None, name="", **kwargs):
        """Container for managing pointcloud-like data

        Args:
            positions (arr, optional): positions for defining the positions of the points.
            features (dict, optional): features. Defaults to None.
            name (str, optional): Name of the point cloud. Defaults to "".
        """
        super().__init__(name)
        positions = {} if positions is None else positions
        features = {} if features is None else features

        if positions is not None:
            self.positions = positions

        self.features = features
        self.pos = self.positions
        self.feats = self.features

    @property
    def feat_shape(self):
        return self.calc_feat_shape(self.features)

    def __repr__(self) -> str:
        if self.is_sparse:
            pos_str = "list len = %s" % len(self.positions)
        else:
            pos_str = "shape = %s" % list(self.positions.shape)

        if self.is_sparse:
            feat_str = "%s" % ["%s: list len = %s" % (key, len(val)) for key, val in self.features.items()]
        else:
            feat_str = "%s" % ["%s: shape = %s" % (key, val.shape) for key, val in self.features.items()]

        return "Pointcloud: %s\n - positions: %s\n - features: %s" % (self.name, pos_str, feat_str)

    def __call__(self):
        return self.dict

    def calc_feat_shape(self, features):
        def get_shape(feat):
            return list(feat[0].shape[1:])

        return {k: get_shape(val) for k, val in features.items()}

    def __getitem__(self, idx):
        if self.is_sparse is False:  # then type(arr) = np.arr -> use np fancy indexing
            return {k: val[idx] for k, val in self.dict.items()}
        else:
            if type(idx) == int or type(idx) == slice:
                return {k: val[idx] for k, val in self.dict.items()}
            else:
                return {k: [val[i] for i in idx] for k, val in self.dict.items()}

    def __setitem__(self, idx, values):
        if self.is_sparse is False:  # then type(arr) = np.arr -> use np fancy indexing
            for k, val in values.items():
                self.dict[k][val] = values[k]
        else:
            if type(idx) == int or type(idx) == slice:
                for k, val in values.items():  #
                    self.dict[k][idx] = val
            else:
                for k, val in values.items():  # fir indexing using index /list, e.g., array([6,22,0,3])
                    for i, idx_ in enumerate(to_list(idx)):
                        self.dict[k][idx_] = val[i]

    def __len__(self):
        len_pos = len(self.pos)

        if self.features is not None:
            len_feat = np.unique([len(val) for val in self.features.values()])
            assert len(len_feat) == 1, ("At least one feature or label feature different length than the others. See %s" % self.__repr__())
            assert len_pos == len_feat, "Positions (len:%i) and features (len:%i) have to be of same length." % (len_pos, len_feat)

        return int(len_pos)

    def append(self, pos, feature=None):
        self.pos.append(pos)
        if feature is not None:
            for k, val in feature.items():
                self.features[k].append(val)

    def add_feature(self, data):
        assert type(data) == dict, "added data has to be of type 'dict' "
        self.features = {**data, **self.features}

    def add_position(self, data):
        assert type(data) == dict, "added data has to be of type 'dict' "
        self.positions = {**data, **self.positions}

    @property
    def dict(self):
        pos_name = "%s_pos" % self.name.split("_pc")[0]
        d = {"%s_%s" % (self.name.split("_pc")[0], k): val for k, val in self.features.items()}
        return {pos_name: self.positions, **d}

    @property
    def fixed(self):
        return not self.is_sparse

    @property
    def is_dynamic(self):
        return self.is_sparse

    @property
    def is_sparse(self):
        return not type(self.positions) == np.array

    def to_graph(self):
        name = self.name.replace("_pc", "_graph")
        return Graph(None, features=self(), name=name)


class Graph(PointCloud):

    def __init__(self, edge_index, features=None, name=""):
        super().__init__(positions=None, features=features, name=name)

        self.edge_index = edge_index if edge_index is not None else list()
        self.features = features if features is not None else list()

    def __repr__(self) -> str:

        if self.is_sparse:
            feat_str = "%s" % ["%s: list len = %s" % (key, len(val)) for key, val in self.features.items()]
        else:
            feat_str = "%s" % ["%s: shape = %s" % (key, val.shape) for key, val in self.features.items()]

        return "Graph:\n%s\n - features: %s" % (self.name, feat_str)

    @property
    def dict(self):
        ei_name = "%s_edge_index" % self.name.split("_graph")[0]
        return {ei_name: self.edge_index, **self.features}


class DataContainer():
    def __init__(self, data, name=""):
        """_summary_

        Args:
            data (_type_): _description_
            name (str, optional): _description_. Defaults to "".
            sparse (_type_, optional): _description_. Defaults to None.
        """
        try:
            self.feat_, self.labels_, self.config = data
        except ValueError:
            self.feat_, self.labels_ = data
            self.config = {}

        self.name = name
        self.predictions = None
        self.dtype = "np"
        self.type = self.dtype
        self.dataset = None
        self.plt_kwargs = {"linestyle": "-"}
        self.transformed = False
        self.predictor = Predictor()
        self.plot = PlottingFactory(self)
        self.features = self.feats

    def __repr__(self):
        details = "Features:\n%s\nLabels:\n%s\nConfig:%s" % (self.feat.__repr__(), self.labels.__repr__(), self.config.keys())
        dc_props = "DataContainer: %s\nType: %s\nTransformed: %s\n - - - - - DATA - - - - - \n" % (self.name, self.dtype, self.transformed)
        return "%s\n%s" % (dc_props, details)

    def __getitem__(self, key):
        return {**self.feats, **self.labels}[key]

    def keys(self):
        """Get all keys in the dataset and in the underlying data structures.
        This includes keys of arrays usually hidden in the data structures. E.g., a pointcloud has a feature "pc_x" and positions pc_pos.
        Calling self.features.keys() will, however, only show "pc". Use self.keys() to see all keys.


        Returns:
            list: list of all keys in the stored data structure.
        """
        return list(self.dict.keys())

    def to_dict(self, obj):
        """ Get dict of an dictionary that cotains DataObjects (that include dicts / and arrays) and arrays"""
        data_dict = {}
        obj_dict = {}
        for k, val in obj.items():
            if k in list({**obj_dict, **data_dict}.keys()):
                raise KeyError("Found two keys with the same name")

            if isinstance(val, DataObject):  # Data Objects
                obj_dict = {**val.dict, **obj_dict}
            else:  # array
                data_dict[k] = val

        return {**obj_dict, **data_dict}

    def feat_dict(self):
        """Return dictionary containing all arrays stored in the feature containers.

        Returns:
            dictionary: Dictionary of feature arrays
        """
        return self.to_dict(self.feat)

    def label_dict(self):
        """Return dictionary containing all arrays stored in the label containers.

        Returns:
            dictionary: Dictionary of label arrays
        """
        return self.to_dict(self.labels)

    def check_for_double_keys(self, dict1, dict2):
        for k in dict1.keys():
            if k in list(dict2.keys()):
                raise KeyError("Found two keys with the same name in dict:\n%s\nand dict:\n%s" % (dict1.keys(), dict2.keys()))

    @property
    def dict(self):
        """ Return dict containing all arrays in with respective keys stored in the data container, e.g., in features, labels,
        and config, including all substructures.

        Returns:
            dictionary: Dictionary of all arrays stored in the instance of DataContainer.
        """
        self.check_for_double_keys(self.feat, self.labels)
        self.check_for_double_keys(self.feat, self.config)
        self.check_for_double_keys(self.labels, self.config)

        d = {**self.feat, **self.labels, **self.config}
        return self.to_dict(d)

    def predict(self, model, data, batch_size=None):
        """Function to infer predictions on a dataset using a trained TF/Keras/Torch/PyG model.

        Parameters
        ----------
        model : DNN model
            Trained model that should be evaluated
        dataset : type
            DataContainer that for the evaluation
        batch_size : type
            Batchsize for inference (optional). Default=64.

        Returns
        -------
        predictions : np.array
            Predictions for input dataset

        """
        preds = self.predictor(model, data, batch_size)
        self.add_predictions(preds)
        return preds

    @property
    def feat(self):
        self.feat_, self.labels_ = self.data2np()
        return self.feat_

    @property
    def labels(self):
        self.feat_, self.labels_ = self.data2np()
        return self.labels_

    def to_np(self):
        return self.data2np()

    def data2np(self):
        if self.transformed is False:
            return self.feat_, self.labels_

        if self.dtype == "tf":
            feat, labels = self.tf2np()
        elif self.dtype == "torch":
            feat, labels = self.torch2np()
        elif self.dtype == "pyg":
            feat, labels = self.pyg2np()
        elif self.dtype == "np":
            feat, labels = self.feat_, self.labels_
        else:
            raise AttributeError("Conversion not supported so far")

        self.feat_, self.labels_ = feat, labels
        return feat, labels

    def set_dataset(self, dataset, dtype):
        assert self.dataset is None, "Dataset was already build as %s" % self.dtype
        self.dataset = dataset
        self.dtype = dtype
        return True

    def __call__(self):
        if self.dataset is None:
            return self.to_np()

        return self.dataset

    @property
    def lengths(self):
        len_dict = {}
        for k, val in self.dict.items():
            len_dict[k] = len(val)
        return len_dict

    def __len__(self):
        length = np.unique([len(val) for val in self.labels.values()])
        assert len(length) == 1, "At least one feature or label feature different length than the others. See %s" % self.lengths
        return int(length)

    @property
    def n_samples(self):
        return self.__len__()

    def get_stats(self, tasks=["energy", "impact"]):
        stats = {}
        mc = self.labels

        for task in tasks:
            stats[task] = {"mean": mc[task].mean(), "std": mc[task].std()}

        return stats

    def cut_with_label(self, key, min_val=-np.inf, max_val=np.inf):
        """_summary_

        Args:
            key (_type_): key to cut
            min_val (float, optional): Min value for the cut. Defaults to None.
            max_val (float, optional): Max value for the cut. Defaults to None.
        """
        m1 = self.labels[key]() > min_val
        m2 = self.labels[key]() < max_val
        self.mask((m1 * m2).squeeze())

    def mask(self, mask):
        print("DataSet ", self.name, "mask %i/%i events\n" % (mask.sum(), mask.shape[0]))
        for k, vals in self.feat.items():
            self.feat[k] = vals[mask]

        for k, vals in self.labels.items():
            self.labels[k] = vals[mask]

        if self.predictions is not None:
            for k, vals in self.ypred.items():
                self.ypred[k] = vals[mask]

        return self.feat, self.labels

    def cut(self, mask_or_cut_fn):
        """Apply cuts to the data set. Input can be a mask or a cutting function outputting a mask indicating which events to keep.
            Note that the cut is applied to all values in the feature and the value dict but not the config dict.

        Args:
            mask_or_cut_fn (np.ndarray or pyfunc): Input a mask indicating which events to keep or a python function outputting a mask.
                                                   The input for the calles function is (labels, features)

        Raises:
            TypeError: if output of pyfunc is not a boolean np.ndarray

        Returns:
            tuple(dict, dict): Returns label and feature dict with applied cuts. However, underlying data container is overwritten.
        """
        if type(mask_or_cut_fn) == np.ndarray:
            return self.mask(mask_or_cut_fn)
        elif callable(mask_or_cut_fn) is True:
            mask = mask_or_cut_fn(self.labels, self.feats)
            assert type(mask) == np.ndarray, "Result of cut function has to be of type np.ndarray but is %s" % type(mask)
            assert mask.dtype == np.bool, "Result mask of cut function has to be od dtype np.bool but is %s" % mask.dtype
            return self.mask(mask)
        else:
            raise TypeError("mask_or_cut_fn has to be np.ndarray mask or py_func but is %s" % type(mask_or_cut_fn))

    def add_label(self, label_dict, overwrite=False):
        """Add label(s) to the DataCotainer.

        Args:
            label_dict (dict): Dictionary of feature to add.
            overwrite (bool, optional): Overwrite feature if it is already existent. Defaults to False.

        Raises:
            KeyError: if the labek already exists
        """
        self.add_data(label_dict, self.labels)

    def add_feature(self, feat_dict, overwrite=False):
        """Add feature(s) to the DataCotainer.

            Args:
                feat_dict (dict): Dictionary of feature to add.
                overwrite (bool, optional): Overwrite feature if it is already existent. Defaults to False.

            Raises:
                KeyError: if the feature already exists
            """
        self.add_data(feat_dict, self.feat)

    def add_data(self, data2add, old_dict, overwrite=False):
        """Merge two dictionaries.

        Args:
            data2add (dict): data to add to the new dictionary
            old_dict (dict): destitnation of the merging operation
            overwrite (bool, optional): Overwrite values if key is already existent. Defaults to False.

        Raises:
            KeyError: if the key already exists
        """
        for k in data2add.keys():
            for k_ in old_dict.keys():
                if overwrite is True:
                    old_dict[k] = data2add[k]
                else:
                    if k == k_:
                        raise KeyError("key %s is already part of the feature dict")
                    else:
                        old_dict[k] = data2add[k]

    @property
    def y_pred(self):
        return self.ypred

    @property
    def ypred(self):
        if self.predictions is None:
            raise AttributeError("No predicted data found, add predictions using '.add_predictions()' first.")

        return self.predictions

    def add_predictions(self, predictions):
        for pred in predictions.values():
            assert len(pred) == self.n_samples, "size of predicted samples (%i) has to be of dataset size (%i)" % (len(pred), self.n_samples)

        self.predictions = predictions

    @property
    def y_true(self):
        return {k: val() for k, val in self.labels.items()}

    @property
    def ytrue(self):
        return self.y_true

    # @property
    # def true(self):
    #     return self.labels

    @property
    def tuple(self):
        return (self.feat, self.labels)

    @property
    def x(self):
        return self.feat

    @property
    def y(self):
        return self.labels

    @property
    def feats(self):
        return self.feat

    def make_point_cloud_data(self, feat_keys, positions, sparse, sparse_fn=None, remove_old_data=True, name=None):
        """Function to generate PointCloud data out of a given data

        Args:
            feat_keys ([str]): features of the point cloud
            positions (str): positions of the point cloud
            sparse (bool): should the point cloud should be sparse? I.e., should points without measurements be rejected?
            sparse_fn (py_func, optional): Function for the sparsity condition, i.e., when a point should be rejected. Not, vectorized. Defaults to None.
            remove_old_data (bool, optional): Remove the data in the old feature dictionary to save RAM. Defaults to True.
            name (str, optional): Name of the gerenated point cloud. Defaults to None.

        Returns:
            PointCloud: Instance of PointCloud class
        """

        feat_dict_new = {}
        feat_keys = to_list(feat_keys)

        print("Create point cloud using features %s and positions %s" % (feat_keys, positions))

        if sparse is True:
            point_positions = []

            for f_name in feat_keys:
                assert f_name in self.feats.keys(), "Given feature %s has to be loaded into feature dict" % f_name
                feat_dict_new["%s" % (f_name)] = []

            x = self.feats[feat_keys[0]]

            for idx in range(len(x)):
                ev = x[idx]

                if type(ev) == np.ndarray:
                    ev = np.array(ev, dtype=np.float32)

                if sparse_fn is None:
                    mask = np.bool_(np.ones_like(ev))
                else:
                    mask = sparse_fn(ev)  # mask = (ev != 0) * (ev != sparse_filter)  # empty & add filter

                d_ = (self.config[positions][mask]).squeeze()

                if d_.ndim == 1:
                    d_ = d_[..., np.newaxis]

                point_positions.append(d_)

                for f_name in feat_keys:
                    ev = self.feats[f_name][idx]

                    if type(ev) == np.ndarray:
                        ev = np.array(ev, dtype=np.float32)

                    d_ = ev[mask].squeeze()

                    if d_.ndim == 1:
                        d_ = d_[..., np.newaxis]

                    feat_dict_new[f_name].append(d_)

            # for k, val in feat_dict_new.items():
            #     feat_dict_new[k] = ak.Array(val)
            # point_positions = ak.Array(point_positions)

            pc = PointCloud(point_positions, feat_dict_new)

        else:  # fixed

            for f_name in feat_keys:
                assert f_name in self.feats.keys(), "Given feature %s has to be loaded into feature dict" % f_name
                feat_dict_new[f_name] = self.feats[f_name].squeeze()[..., np.newaxis]

            point_positions = self.config[positions]
            pc = PointCloud(point_positions, feat_dict_new)

        if remove_old_data is True:  # delete old data
            for f_name in feat_keys:
                del self.feats[f_name]

        if name is None:
            leading_key = to_list(feat_keys)[0].split("_")[0]
            name = "%s_pc" % leading_key  # has to have pc in name --> for PyG dataset
            pc.name = name
            self.feats[name] = pc
        else:
            pc.name = name
            self.feats["%s_pc" % name] = pc  # has to have pc in name --> for PyG dataset

        return pc

    def tf(self, exclude_keys=None, transform=None, batch_size=None):
        """ Create high-performance dataset using tf.dataset.

        Parameters
        ----------
        exclude_keys : list
            keys that be removed from the dataset (not loaded)
        transform : fn
            transformation as applied to the features using tf.dataset.map
        batch_size : int
            set number if dataset should be batched

        Returns
        -------
        type
            tf.Dataset

        """
        from tensorflow.data import Dataset
        if transform is not None:
            self.transformed = True

        try:
            tf_dataset = Dataset.from_tensor_slices(self.tuple, name=self.name)
        except ValueError:
            raise ValueError("Failed to convert a NumPy array to a Tensor (Unsupported object type numpy.ndarray). Are you trying to convert a graph collection of np.objects sampkes into a tf dataset?")

        if transform is not None:
            tf_dataset = tf_dataset.map(partial(transform))

        if exclude_keys is not None:
            from data.tf.mappings import rm_unused_labels
            tf_dataset = tf_dataset.map(partial(rm_unused_labels, exclude_keys=exclude_keys))

        if batch_size is not None:
            tf_dataset = tf_dataset.batch(batch_size)

        self.set_dataset(tf_dataset, "tf")
        return tf_dataset

    @property
    def pt(self):
        return self.torch()

    def pyg(self):
        return self.torch_geometric()

    def point_clouds_to_pyg_graphs(self, graph_transforms, point_clouds=None, exclude_keys=[], add_positions=True, empty_graphs_as_single_node=True, to_np=False):
        """Transform data set to graph (PyG) datasets using PointClouds.

        Args:
            graph_transform (torch_geometric.transforms or dictionary of transforms): Definition of torch_geometric transform to form a Graph using the inserted PointClouds. Insert dictionary for applying different transform to differen PointClouds. If only one transform is given it is applied for all PointClouds.
            point_clouds (str, optional): PointClouds to be transformed into Graphs PyG datasets. Defaults to all PointClouds found in the features of the DataContainer.
            exclude_keys (list(str), optional): List of key that should not be exported into torch_geometric Dataset. Defaults to [].
            add_positions (bool, optional): Should be the position of the point cloud used as feature?. Defaults to True.
            empty_graphs_as_single_node (bool, optional): Setting to True, creates for an empty graph (e.g., no detection in specific instrument) a graph with a single node and a self connection. Note, this is also a meaningful choice if the PointCloud is non-fixed (variable), e.g., can change from sample to sample. Defaults to True.

        Returns:
            list: list of torch geometric graph data using the loaded PointClouds

        """
        import torch
        from data.torch.pyg_data import MultiGraphData
        from torch_geometric.data import Data

        # if transform is not None:
        #     self.transformed = True

        print("********************************\ncreate PyTorch_Geometric Dataset\n********************************")

        if point_clouds is None:
            point_clouds = {k: val for k, val in self.feats.items() if isinstance(val, PointCloud)}
        else:
            point_clouds = {k: self.feats[k] for k in to_list(point_clouds)}

        if type(graph_transforms) != dict:
            graph_transforms = {k: graph_transforms for k in point_clouds.keys()}

        fixed_graphs = {}
        for pc in point_clouds.values():
            assert isinstance(pc, PointCloud), "Input feature for building graph has to be of type 'PointCloud'."

            if pc.fixed is True:  # create single graph for fixed postition to avoid performing same graph transform for reach sample
                fixed_graphs[pc.name] = graph_transforms[pc.name](Data(pos=torch.Tensor(pc.positions)))
                print("Create fixed graph for:", pc.name)

        list_graph_data = []
        prog = Progbar(self.n_samples, verbose=1 if is_interactive() else 2)
        features2load = {k: val for k, val in self.feats.items() if type(val) != PointCloud and k not in exclude_keys}
        labels2load = {k: val for k, val in self.labels.items() if k not in exclude_keys}

        for idx in range(self.n_samples):  # loop over all samples in dataset and create Graph / Multi-Graph
            data = {}

            for k, pc in point_clouds.items():  # write features to data_dict

                split_name = k.split("_pc")[0]
                if len(point_clouds) > 1:  # naming of Graphs
                    ei_name = "%s_edge_index" % split_name
                else:
                    ei_name = "edge_index"  # For single graph. To support largest PyG comaptibility

                # FIXED GRAPHS --> simply use position and edge index from the single built graph
                if pc.fixed is True:
                    data[ei_name] = fixed_graphs[k].edge_index  # Add edge index // Adjacency

                    if add_positions is True:
                        data["%s_pos" % split_name] = fixed_graphs[k].pos

                    # add PointCloud features to graph dataset
                    for pc_feat_key, pc_feat_val in pc.features.items():
                        if pc_feat_val.ndim == 1:
                            data[pc_feat_key] = torch.Tensor([pc_feat_val[idx]])[None, :]
                        else:
                            data[pc_feat_key] = torch.Tensor(pc_feat_val[idx])

                else:
                    # NON-FIXED GRAPHS // Graphs of variable size --> loop over list, transform, and check if empty
                    pos = pc.positions[idx]

                    if len(pos) == 0:  # empty graph
                        if empty_graphs_as_single_node is True:  # -> create single node with feature 0 and self-loop
                            data["%s_pos" % split_name] = torch.zeros(pos.shape[1:], dtype=torch.float)[None, :]
                            data[ei_name] = torch.zeros((2, 2), dtype=torch.long)

                    else:
                        data[ei_name] = graph_transforms[pc.name](Data(pos=torch.Tensor(pos))).edge_index  # add edge index

                        if add_positions is True:
                            data["%s_pos" % split_name] = torch.Tensor(pos)  # add pos

                    for pc_feat_key, pc_feat_val in pc.features.items():
                        val = pc_feat_val[idx]

                        if len(val) == 0:  # empty graph
                            if empty_graphs_as_single_node is True:  # -> add to single node zeroed features
                                data[pc_feat_key] = torch.tensor(val.shape[1:], dtype=torch.float)[None, :]
                        else:
                            data[pc_feat_key] = torch.tensor(val, dtype=torch.float)  # add edge index

            # add additional features to graph dataset
            for feat, feat_val in features2load.items():
                # if len(val[idx]) == 0:
                #     data[feat] = torch.tensor([[0]], dtype=torch.float)  # add zero for empty graphs / features
                # else:
                data[feat] = torch.Tensor(feat_val[idx])[None, :] if feat_val[idx].ndim == 1 else torch.Tensor(feat_val[idx])

            # add labels to graph dataset
            for lab, lab_val in labels2load.items():
                arr_val = lab_val()
                if arr_val.size != len(arr_val):
                    data[lab] = torch.Tensor([lab_val[idx]])
                else:
                    data[lab] = torch.Tensor(lab_val[idx])

            # Build PyG Dataset
            if len(point_clouds) > 1:
                list_graph_data.append(MultiGraphData(**data))
            else:
                list_graph_data.append(Data(**data))  # single graph

            if idx % 1000 == 0:
                prog.add(1000)

        self.set_dataset(list_graph_data, "pyg")
        return list_graph_data

    def torch_geometric(self, graph_transform, transform=None, exclude_keys=[]):
        """ Make list of torch geometric graph data using the loaded numpy arrays."""
        import torch
        from data.torch.pyg_data import MultiGraphData
        from torch_geometric.data import Data

        if transform is not None:
            self.transformed = True

        print("********************************\ncreate PyTorch_Geometric Dataset\n********************************")
        assert True in ["pos" in k for k in self.feat.keys()], "No position feature (containing 'pos') was found in the feature data dict. Is point cloud data loaded? Are the positions of the points defined?"

        pos_dict = {k: val for k, val in self.dict.items() if "pos" in k}
        data_dict = {k: val for k, val in self.dict.items() if "pos" not in k}
        fixed_graph = {}

        for k, val in pos_dict.items():
            if len(val) != self.n_samples:
                fixed_graph[k] = graph_transform(Data(pos=torch.Tensor(val)))

        print("Found %i position keywords in the dataset." % len(pos_dict.keys()))
        print("Is the graph fixed?:\n", {k: k in fixed_graph.keys() for k in pos_dict.keys()})

        prog = Progbar(self.n_samples, verbose=1 if is_interactive() else 2)
        data_list = []

        for idx in range(self.n_samples):
            data = {}

            for k, val in data_dict.items():

                if len(val[idx]) == 0:
                    data[k] = torch.tensor([[0]], dtype=torch.float)  # add zero for empty graphs / features
                else:
                    data[k] = torch.Tensor(val[idx])[None, :] if val[idx].ndim == 1 else torch.Tensor(val[idx])

            for k, val in pos_dict.items():
                ei_name = "ei_" + k  # edge_index named after clustering feature -> Meaningful!

                if k in fixed_graph:  # fixed graphs
                    data[k] = fixed_graph[k].pos  # add pos
                    data[ei_name] = fixed_graph[k].edge_index  # add edge index
                else:  # sparse graphs
                    if len(val[idx]) == 0:  # empty graph -> create single node with feature 0 and self-loop
                        data[k] = torch.tensor([[0, 0]], dtype=torch.float)
                        data[ei_name] = torch.tensor([[0, 0],
                                                     [0, 0]], dtype=torch.long)
                    else:
                        data[k] = torch.Tensor(val[idx])  # add pos
                        data[ei_name] = graph_transform(Data(pos=torch.Tensor(val[idx]))).edge_index  # add edge index

            data_list.append(MultiGraphData(**data))

            if idx % 1000 == 0:
                prog.add(1000)

        self.set_dataset(data_list, "pyg")
        return data_list

    def torch(self, transform=None, target_transform=None):
        from data.torch.torch_data import PytorchImageDataset

        if transform is not None:
            self.transformed = True

        if target_transform is not None:
            self.transformed = True

        dset = PytorchImageDataset(self, transform=transform, target_transform=target_transform)
        self.set_dataset(dset, "torch")
        return dset

    def tf_loader(self, batch_size, exclude_keys=None, shuffle=False):
        from data.tf.mappings import rm_unused_labels
        from tensorflow.data import Dataset

        assert isinstance(self(), Dataset) is True, "Data has to be of type tf.data.Dataset"

        tf_mapped = self()

        if shuffle is True:
            nsamples = self().cardinality().numpy()
            tf_mapped = self().shuffle(nsamples, reshuffle_each_iteration=shuffle)

        if exclude_keys is not None:
            ex_k_fn = partial(rm_unused_labels, exclude_keys=exclude_keys)
            tf_mapped = tf_mapped.map(ex_k_fn)

        return tf_mapped.batch(batch_size)

    def pyg_loader(self, batch_size, shuffle=False, follow_batch=None, exclude_keys=None, drop_last=False, **kwargs):
        from torch_geometric.loader import DataLoader
        return DataLoader(self(), batch_size, shuffle, follow_batch, exclude_keys, drop_last=drop_last, **kwargs)

    def torch_loader(self, batch_size=1, exclude_keys=None, shuffle=False, follow_batch=None, drop_last=False, **kwargs):
        """ Create torch and PyG dataloader for the training pipeline

        Parameters
        ----------
        keys : list
            keys that should form the labels in the torch DataLoader
        transform : fn
            transformation as applied to the features using torch transform
        batch_size : int
            set number if dataset should be batched

        Returns
        -------
        type
            torch.utils.data.DataLoader

        """
        from torch.utils.data import Dataset

        if isinstance(self(), Dataset) is True:  # pytorch dataset
            from torch.utils.data import DataLoader

            if follow_batch is not None:
                print("setting 'follow_batch' for loading a torch data set is meaningless")

            if exclude_keys is not None:
                from data.torch.mappings import RmUnusedLabels
                target_transform = RmUnusedLabels(exclude_keys)
                dset = self().map(target_transform=target_transform)
            else:
                dset = self()
            return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)

        else:
            from torch_geometric.data import Data
            if isinstance(self(), list) and isinstance(self()[0], Data) is True:  # pyg dataset
                return self.pyg_loader(batch_size, shuffle=shuffle, follow_batch=follow_batch, exclude_keys=exclude_keys, drop_last=drop_last, **kwargs)
            else:
                raise TypeError("Dataset has wrong type to build torch loader. Dataset has to be of type torch.utils.data.Dataset or list of torch_geometric.data.Data objects")

    def pyg2np(self):
        """Converts pytorch geometric data set into numpy data set.

        """
        value_dict = {}
        to_graph = {}

        for k, val in self.feat.items():
            if isinstance(val, PointCloud) is True:
                graph = self.feat[k].to_graph()
                self.feat[k] = graph
                to_graph[k] = graph.name
                value_dict = {**value_dict, **graph()}  # Reference to values
            else:
                value_dict = {**value_dict, k: val}

        # rename to graph
        for old_name, new_name in to_graph.items():
            self.feat[new_name] = self.feat.pop(old_name)

        prog = Progbar(self.n_samples, verbose=1 if is_interactive() else 2)
        print("Converting pyg data set into np data")

        for i, event in enumerate(self()):
            for k, val in event.to_dict().items():
                if k in value_dict.keys():
                    value_dict[k].append(val.cpu().detach().numpy().squeeze())
                elif k in self.labels.keys():
                    self.labels[k][i] = val.cpu().detach().numpy().squeeze()

            if i % 1000 == 0:
                prog.add(1000)

        for k, val in value_dict.items():  # to be fixed
            try:
                value_dict[k] = np.array(val, dtype=np.object)
            except ValueError:
                pass

        for k, val in self.labels.items():
            arr = np.stack(val(), axis=0)
            # if arr.ndim == 1:
            #     arr = arr[:, np.newaxis]
            self.labels[k].arr = arr

        self.transformed = False
        self.dtype = "np"
        return True

    def torch2np(self):
        import torch
        from models.torch.base import to_device
        loader = self.torch_loader(batch_size=64, shuffle=False)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feat = {k: [] for k in self.feat.keys()}
        lab = {k: [] for k in self.labels.keys()}

        for i, data in enumerate(loader):
            val_feat, val_lab = to_device(data, device)

            for k in feat.keys():
                feat[k].append(val_feat[k].cpu().detach().squeeze())

            for k in lab.keys():
                lab[k].append(val_lab[k].cpu().detach().squeeze())

            for k, val in feat.items():
                feat[k] = np.concatenate(val, axis=0)

            for k, val in lab.items():
                feat[k] = np.concatenate(val, axis=0)

        self.transformed = False
        return feat, lab

    def tf2np(self, batch_size=64):
        from tools.progress import Progbar

        feat, lab = self().element_spec
        n_samples = self().cardinality().numpy()
        feat = {k: np.zeros((n_samples,) + tuple(val.shape), dtype=val.dtype.as_numpy_dtype) for k, val in feat.items()}
        lab = {k: np.zeros((n_samples,) + tuple(val.shape), dtype=val.dtype.as_numpy_dtype) for k, val in lab.items()}

        it, iter_max = 0, n_samples // batch_size
        progbar = Progbar(iter_max + 1)
        print("perform mapping tf --> np")
        for batch in self().batch(batch_size):
            feat_batch, lab_batch = batch
            if it != iter_max:
                low = it * batch_size
                up = (it + 1) * batch_size
            else:
                low = it * batch_size
                up = low + n_samples % batch_size

            for k, val in feat_batch.items():
                feat[k][low:up] = val

            for k, val in lab_batch.items():
                lab[k][low:up] = val

            progbar.add(1)
            it += 1

        self.transformed = False
        return feat, lab

    def save(self, log_dir=None, name=None):
        from os.path import join
        import pickle
        log_dir = self.log_dir if log_dir is None else log_dir
        name = self.name if name is None else name
        dir = join(log_dir, "%s.pickle" % name)

        with open(dir, 'wb') as file:
            pickle.dump(self, file)

    def save_labels(self, log_dir, name=""):
        from os.path import join
        name = self.name if name == "" else name
        preds = {"%s_pred" % k: val for k, val in self.y_pred.items()}
        np.savez(join(log_dir, name), **{**self.labels, **preds})

    def resample_after_classes(self, var_name, class_mask, bins=None):
        """Make a disitrbution class balanced. I.e. sample the distribution of the input variable so that it is similar for all classes
        Args:
            var_name (key): Name of the distribution to be re-sampled
            bins (arr): Binning over which resampling is done on var_name
            class_mask (arr): Boolean mask, defining which event belongs to which class
        """
        if bins is None:
            print("No binning is provided, no resampling is done!")
            return
        dist_data = self.dict[var_name]
        mask_arr1, mask_arr2 = self.down_sample(dist_data[class_mask], dist_data[~class_mask], bins)
        mask = np.ones(self.n_samples, dtype=np.bool)
        class_mask_temp = ~class_mask
        # now apply the masks keeping in mind the double indexing
        class_mask[class_mask] = mask_arr1
        mask[class_mask] = False
        class_mask_temp[class_mask_temp] = mask_arr2
        mask[class_mask_temp] = False
        self.mask(mask)

    def down_sample(self, arr1, arr2, bins):
        """Sample the distribution of 2 variables to follow the same distribution

        Args:
            arr1 (arr): Array of Variable 1
            arr2 (arr): Array of Variable 2

        Returns:
            mask_arr1 (arr): Boolean mask for Variable 1
            mask_arr2 (arr): Boolean mask for Variable 2

        """
        arr1_binned, _ = np.histogram(arr1, bins=bins)
        arr2_binned, _ = np.histogram(arr2, bins=bins)
        min_arr1_arr2 = np.minimum(arr1_binned, arr2_binned)
        sel_arr1 = list()
        sel_arr2 = list()

        for i, j, min in zip(bins[:-1], bins[1:], min_arr1_arr2):
            arr1_indices = np.where((arr1 > i) & (arr1 <= j))[0]
            sel_arr1.append(np.random.choice(arr1_indices, min, replace=False))
            arr2_indices = np.where((arr2 > i) & (arr2 <= j))[0]
            sel_arr2.append(np.random.choice(arr2_indices, min, replace=False))
        sel_arr1 = np.concatenate(sel_arr1).ravel()
        sel_arr2 = np.concatenate(sel_arr2).ravel()
        mask_arr1 = np.ones(arr1.shape[0], dtype=np.bool)
        mask_arr2 = np.ones(arr2.shape[0], dtype=np.bool)

        mask_arr1[sel_arr1] = ~mask_arr1[sel_arr1]
        mask_arr2[sel_arr2] = ~mask_arr2[sel_arr2]
        return mask_arr1.ravel(), mask_arr2.ravel()

    def split2train_val_test(self, val_split=0.05, test_split=0.15):
        """Split data into training, validation, and test dataset.

        Args:
            val_split (float, optional): validation split. Defaults to 0.1.
            test_split (float, optional): test_split. Defaults to 0.2.

        Returns:
            _type_: _description_
        """
        idx_train, idx_val, idx_test = self.get_train_val_test_indices(val_split, test_split)
        valid_tuple, test_tuple = self.split(idx_train, idx_val, idx_test)
        test_dc = DataContainer(test_tuple + (self.config,))
        valid_dc = DataContainer(valid_tuple + (self.config,))
        return self, valid_dc, test_dc

    def get_train_val_test_indices(self, val_split, test_split):
        np.random.seed(1)
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        n_val = int(self.n_samples * val_split)
        n_test = int(self.n_samples * test_split)
        n_train = self.n_samples - n_val - n_test
        return idx[: n_train], idx[n_test + n_train:], idx[n_train: n_test + n_train]

    def split(self, idx_train, idx_val, idx_test):
        from data.data import Array

        train = self.feats, self.labels
        valid, test = ({}, {},), ({}, {},)
        # first re-index whole array
        # than make simple slice?!
        # doesn't matter should be similar fast
        # finialize __getitem__ and __setitem__

        def make_copy(obj, idx):
            new_obj = obj.get_empty_copy()
            new_obj[:] = obj[idx]
            return new_obj

        for k, val in self.feats.items():
            test[0][k] = make_copy(val, idx_test)
            valid[0][k] = make_copy(val, idx_val)

            # overwrite current dataset
            if isinstance(val, Array):
                train[0][k].arr = val[idx_train]
            else:
                train[0][k][:] = val[idx_train]

        for k, val in self.labels.items():
            test[1][k] = make_copy(val, idx_test)
            valid[1][k] = make_copy(val, idx_val)

            # overwrite current dataset
            if isinstance(val, Array):
                train[1][k].arr = val[idx_train]
            else:
                train[1][k][:] = val[idx_train]

        return valid, test


def to_device(self, data, device=None):
    if device is None:
        device = self.device

    feat, labels = data

    for k, val in feat.items():
        feat[k] = val.to(device)

    for k, val in labels.items():
        labels[k] = val.to(device)

    return feat, labels


def load_dataset_from_npz(path, name, feat_path=""):
    """Load data_set.

    Args:
        path (_type_): path of the saved predictions (and labels)
        name (_type_): name of the dataset
        feat_path (str, optional): path of the saved features

    Returns:
        DataContainer: loaded DataContainer
    """
    f = np.load(path)
    labels = {name: f[name] for name in f.files}

    if feat_path != "":
        f_feat = np.load(feat_path)
        feat = {name: f_feat[name] for name in f_feat.files}
    else:
        feat = {}

    predictions = {}
    for k in [k_ for k_ in labels.keys() if "_pred" in k_]:
        predictions.update({k.replace("_pred", ""): labels.pop(k)})

    dc = DataContainer((feat, labels), name=name)
    if predictions is not None:
        dc.add_predictions(predictions)

    return dc
