#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tools.utils import to_list
import numpy as np
from data.data import Array
import h5py
import numpy as np  # noqa
import hdf5plugin  # noqa


def identity(x):
    return x


def struct_array_from_HDF5_group(path2group, keys="all"):
    pass


def struct_array_from_saved_pytables(path2group, keys="all"):
    pass


class StructuredArray():
    def __init__(self, key_path, transform_fn=None, name="", unit="", table_key=None, label="") -> None:
        """Fundamental for the definition of data arrays to be loaded from saved hdf5 / numpy files.

        Args:
            key_path (str): path in the HDF5 to the values
            transform_fn (fn, optional): Function applied during loading (e.g. basic pre-processing or reasonable re-shaping). Defaults to None.
            name (str, optional): Name of the loaded data. Defaults to last part (.../NAME) of the key_path.
            table_key (str, optional): If HDF5 dataset is pytable, the 'table_key' has to be used to index array.
            If the whole array should be loaded, make use of the function 'struct_array_from_saved_pytables'. Defaults to None.
        """
        self.key_path = key_path
        self.transform_fn = identity if transform_fn is None else transform_fn
        if name == "":
            self.name = key_path.split("/")[-1] if table_key is None else table_key
        else:
            self.name = name

        self.table_key = table_key
        self.unit = unit
        self.label = label

    @property
    def info(self):
        print(self)
        print("  Key path: '%s'" % self.key_path)
        print("  Transform_fn: %s" % self.transform_fn.__name__)
        print("  Unit: '%s'" % self.unit)
        print("  Table_key: '%s'" % self.table_key)
        return

    def __repr__(self):
        return "StructuredArray: %s" % self.name

    def __call__(self, file):
        """Load data into RAM by providing a file (instance of data.File).
        The array is constructed by reading from the given file and applying the transformation function.

        Args:
            file (data.File): File to open.

        Returns:
            arr: Loaded and transformed array.
        """

        if self.table_key is not None:
            return self.transform_fn(file[self.key_path][self.table_key])
        else:
            return self.transform_fn(file[self.key_path])

    def get_len(self, file):
        return len(file[self.key_path])

    def get_shape_before_transform(self, file):
        """Get shape of arr in the HDF5 file before applying the transformation.
        Note function is not computationally nor memory efficient for table datasets!

        Args:
            file (File): file to read

        Returns:
            tuple: Shape of the array after transform
        """
        if self.table_key is not None:
            return file[self.key_path][self.table_key].shape  # this is stupid!
        else:
            return file[self.key_path].shape

    def get_shape(self, file):
        """Get shape of arr after applying the transformation.

        Args:
            file (File): file to read

        Returns:
            tuple: Shape of the array after transform
        """
        shape = self.get_shape_before_transform(file)

        out_shape = self.get_shape_after_transform((2,) + shape[1:])
        if out_shape[0] == 2:
            shape = (shape[0],) + out_shape[1:]  # first axis remains unchanged
        else:
            shape = out_shape  # first axis does not remain unchanged
        return shape

    def get_shape_after_transform(self, arr_shape):
        """Calculate the output shape of an array after performing a transformation using self.transform_fn.

        Args:
            arr_shape (tuple): Shape of the input array

        Returns:
            tuple: Shape after the transformation
        """
        err_state = np.geterr()
        np.seterr(all="ignore")
        arr = np.ones(shape=arr_shape)
        np.seterr(**err_state)
        return self.transform_fn(arr).shape


class Image(StructuredArray):  # structured array
    def __init__(self, name, data_path, load=True) -> None:
        super().__init__()
        self.name = name
        self.data_path = data_path
        self.load = True

    def to_graph(self, pos=None):
        return True


class File():
    def __init__(self, path, slice_op_or_mask=None, dtype="hdf5"):
        """Basic class for managing numpy and HDF5 files. Currently only hdf5 supported!
            By calling the an instance of file with an StructuredArray the data is loaded into the RAM.

        Args:
            path (str): Path to the data file.
            slice_op_or_mask (_type_, optional): Masking or slicing to be applied to the dataset when loading the data. Defaults to None.
            dtype (str, optional): format of the dataset. Defaults to "hdf5".
        """
        assert type(path) == str, "insert path has to be of type(str), but type: %s was given" % type(path)
        self.path = path
        assert type(slice_op_or_mask) == np.ndarray or type(slice_op_or_mask) == slice or slice_op_or_mask is None, "slice_op_or_mask has to be of type slice or np.ndarray"
        self.slice_op_or_mask = slice_op_or_mask
        self.dtype = dtype
        self.file_ = None

    @property
    def file(self):
        return self.open()

    def get_loaded_len(self, key_obs):
        return self.get_loaded_shape(key_obs)[0]

    def get_len_shape(self, key_obs):
        """Get shape when

        Args:
            key_obs (_type_): _description_

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """

        if isinstance(key_obs, StructuredArray):
            key_obs = key_obs.key_path

        shape = key_obs.get_shape(self.open)

        if self.slice_op_or_mask is not None:
            if type(self.slice_op_or_mask) == slice:
                samples2load = len(range(*slice(0, 2).indices(shape[0])))
            elif type(self.slice_op_or_mask) == np.ndarray:
                samples2load = self.slice_op_or_mask.astype(bool).sum()
            else:
                raise TypeError("slice_op_or_mask %s has wrong type" % self.slice_op_or_mask)

        assert samples2load < shape[0], "%s\n Samples2load %i have to be smaller than number of samples in the file %s" % (self.f, samples2load, shape[0])
        if samples2load < shape[0]:
            shape[0] = samples2load

        return shape

    def __call__(self, obs=None):
        if obs is None:
            return self.open()
        else:
            return obs(self)

    def __getitem__(self, key, table_key=None, slicing=None):
        """Load data from file into disk

        Args:
            key (str): _description_

        Returns:
            arr: data array
        """

        # Next lines need to be reworked for on-disk datasets
        opened_data = self.open()[key]

        if table_key is not None:  # use for table indexing
            opened_data = opened_data[table_key]  # this already loads everything into RAM -> to be improved

        if slicing is not None:  # use for array slicing
            opened_data = opened_data[slicing]  # slice always if explicitly asked for

        elif type(opened_data) == h5py.Dataset:  # only slice for Dataset
            if self.slice_op_or_mask is not None:
                return opened_data[self.slice_op_or_mask]
            else:
                res = opened_data[:]

                if res.dtype == np.float64:
                    res = res.astype(np.float32)

                return res  # load full data into RAM
        else:
            return opened_data

    def __repr__(self):
        repr_str = "File: %s" % self.path
        return repr_str

    def is_open(self):
        try:
            return self.file_.mode
        except ValueError:  # file is closed
            return False
        except AttributeError:  # file is closed
            return False  # file was not opened

    def is_closed(self):
        x = self.is_open()

        if type(x) is not bool:
            x = True

        return x

    def open(self):
        if self.is_open() is False:
            try:
                self.file_ = h5py.File(self.path, "r")
            except FileNotFoundError:
                try:
                    self.file_ = np.load(self.path)
                except FileNotFoundError:
                    raise FileNotFoundError("File %s was not found" % self.path)

        return self.file_

    def close(self):
        self.file_.close()
        self.file_ = None

    def walk_tree(self, details=True):
        """ Draw the tree of yout HDF5 file to see the hierachy of your dataset
            params: detail(activate details to see shapes and used compression ops, Default: True)
        """

        def walk(file, iter_str=''):
            try:
                keys = file.keys()
            except AttributeError:
                keys = []

            for key in keys:
                try:
                    if details:
                        d = file[key].dtype
                        if len(d) > 1:
                            print(iter_str + str(file[key]), "PyTable")
                            for desc in d.descr:
                                print("   %s> %s " % (iter_str, desc))
                        else:
                            print(iter_str + str(file[key]))
                    else:
                        print(iter_str + key)
                except (AttributeError, TypeError):
                    print(iter_str + key)
                    walk(file[key], "   " + iter_str)

        with self.open() as file:
            print("filename:", file.filename)
            for key in file.keys():
                print(' - ' + key)
                walk(file[key], iter_str='   - ')

    def extract_info(self, path):
        with self.get_h5_file() as f:
            data = f[path]
            y = np.stack(data[:].tolist())

        return {k: y[:, i] for i, k in enumerate(data.dtype.names)}, dict(data.dtype.descr)


class Loader():
    def __init__(self, file_list, recipe=None) -> None:
        """Basic class for computationally efficient data loading.
            Given a list of HDF5 / numpy files the loader construct a tuple of dictionaries as input for the DataContainer
            and loads the data from the files into the memory (to be changed for on-disk datasets).
            For loading the data, the loader is based on StructuredArrays which define how the data is loaded and its property.
            For convenience the StructuredArrays can be grouped into recipes.

        Args:
            file_list (_type_): _description_
            recipe (_type_, optional): _description_. Defaults to None.
        """
        self.files = [f if isinstance(f, File) else File(f) for f in file_list]
        self.recipe = recipe
        self.features = []
        self.labels = []

    def __repr__(self) -> str:
        list_files = "\n".join([f.path for f in self.files])
        return "Loader: \n Files to load\n %s" % list_files

    def add_file(self, file_or_file_name, **file_kwargs):
        if isinstance(file_or_file_name, File):
            self.files.append(file_or_file_name)
        else:
            self.files.append(File(file_or_file_name))

    def add_features(self, features):
        self.features += to_list(features)

    def add_labels(self, labels):
        self.labels += to_list(labels)

    def append():
        pass

    def check4doubles(self):
        """Check for doubles in the filelist. Remove files that occur more often than once.
        """
        files = np.array([file.path for file in self.files])
        files, idx, counts = np.unique(files, return_index=True, return_counts=True)

        if counts.max() > 1:
            mask = counts > 1
            for file, count in zip(files[mask], counts[mask]):
                print("Found %i times the file: '%s'" % (count, file))

            print("\n --> REMOVE FILES WHICH ARE REPEATED IN FILELIST")
            self.files = np.array(self.files)[idx].tolist()

    def check_files(self):
        bad_files = []

        self.check4doubles()

        for file in self.files:
            try:
                file.open()
                file.close()
            except FileNotFoundError:
                bad_files.append(file)

        if bad_files != []:
            raise FileNotFoundError("Files %s were not Found" % bad_files)

        return True

    def create_arr(self, obs):
        """Generate empty arrays for loading obs into RAM

        Args:
            obs (StructuredArray): Array to load from file

        Returns:
            np.array: empty array (filled with zeros)
        """
        shape = []

        for file in self.files:
            if shape == []:
                shape = list(obs.get_shape(file))
            else:
                file_len = obs.get_len(file)
                shape[0] += file_len

        shape = tuple(shape)
        return np.zeros(shape=shape)

    def load_data(self):
        print("\n******** Loading data from Disk ********\n")
        return self.load_labels(), self.load_features()

    def load_labels(self, label_dict=None):
        label_dict = {} if label_dict is None else label_dict
        return self.load(self.labels, label_dict)

    def load_features(self, feat_dict=None):
        feat_dict = {} if feat_dict is None else feat_dict
        return self.load(self.features, feat_dict)

    def load(self, observables, data_dict):
        """load list of observables into the given data dictionary

        Args:
            observables (list): _description_
            data_dict (dict): _description_
        """
        # Check files
        self.check_files()

        # create empty (zeros) arrs for all events in the files
        for obs in observables:
            assert obs.name not in data_dict.keys(), "Data %s are already part of the dataset. Raise Error to prevent overwriting" % obs.name
            data_dict[obs.name] = Array(self.create_arr(obs), name=obs.name, unit=obs.unit, label=obs.label)

        # sequentially fill the data
        curr_buff = 0
        for file in self.files:
            assert isinstance(file, File), "Given file %s has to be instance of the File class." % file
            with file.open() as f:
                arr_size = 0
                for i, obs in enumerate(observables):
                    len_ = obs.get_len(f)
                    if i == 0:
                        arr_size = len_
                    if i > 0:
                        if len_ != arr_size:
                            print("Warning: obs %s does have length %i but %i was expected" % (obs.name, len_, arr_size))

                    data_dict[obs.name].arr[curr_buff:curr_buff + arr_size] = obs(f)

            curr_buff += arr_size

        return data_dict
