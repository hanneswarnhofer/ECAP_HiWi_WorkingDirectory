import h5py
from tools.utils import to_list
from data.loader import StructuredArray


def hdf5_tables_to_structs(table_path, table_keys=None, file=None, transform_fns={}, table_name_replace_fn=None):
    """ Defined full tables to data structures to prepare for data loading.

    Args:
        table_path (str): Path to the pytable in HDF5 file
        table_keys (str, optional): Keys of data you want to load. Defaults to None. None: all tables will be loaded (insert example file)
        file (str, optional): Path to h5 file. Defaults to None. Needed when no table_keys are defined.
        transform_fns (dict, optional): dictionary of transformation functions to preprocess data during loading. Defaults to {}.
        table_name_replace_fn (pyfunc, optional): Function defining how the name shoukd be modified. Defaults to None.

    Returns:
        list: list of build data structures
    """

    if file is not None:
        assert isinstance(file.file[table_path], h5py.Dataset) is True, "given path does not link to a HDF group"
        if table_keys is None:
            table_keys = list(file[table_path].dtype.names)
    else:
        assert table_keys is not None, "Give table keys to load or provide a file (via the file argument) to use all existent table keys"

    table_keys = to_list(table_keys)

    obs = []

    if callable(transform_fns):
        transform_fns = {k: transform_fns for k in table_keys}
    if transform_fns is None:
        transform_fns = {k: None for k in table_keys}
        
    for t_key in table_keys:
        t_fn = transform_fns[t_key] if transform_fns is not None else None
        table_name = t_key if table_name_replace_fn is None else table_name_replace_fn(t_key)
        obs.append(StructuredArray(table_path, table_key=t_key, transform_fn=t_fn, name=table_name))

    return obs


def hdf5_group_to_structs(group_path, group_keys=None, file=None, table_keys=None, key_name_replace_fn=None, table_key_replace_fn=None, transform_fns=None):
    """Construct list of observables from a HDF5 group, which can also hold various tables

    Args:
        group_path (str): path to the hf5 group
        group_keys (str): path to the hf5 group
        file (File): example file which features the structure to be used later
        table_key (_type_): Only to set if hdf5 Datasets within the group are tables
        transform_fns (dict, optional): dictionary of transform functions to be applied for the respective Dataset. Defaults to {}.

    Returns:
        _type_: _description_
    """

    if file is not None:
        assert isinstance(file.file[group_path], h5py.Group) is True, "given path does not link to a HDF group"
        dset_keys = file[group_path].keys()
        group_keys = dset_keys if group_keys is None else to_list(group_keys)
    else:
        assert group_keys is not None, "Give group keys to load or provide a file (via the file argument) to load all existent data in the group"
        group_keys = to_list(group_keys)

    assert table_keys is not None, "Give table keys to load or provide a file (via the file argument) to load all existent data in the group"
    table_keys = to_list(table_keys)
    obs = []
    
    if callable(transform_fns):
        transform_fns = {k: transform_fns for k in group_keys}
    if transform_fns is None:
        transform_fns = {k: None for k in group_keys}

    for key in group_keys:
        # if key not in group_keys:
        #     continue
        key_name = key if key_name_replace_fn is None else key_name_replace_fn(key)

        if table_keys is not None:
            dset_path = "%s/%s" % (group_path, key)
            if table_key_replace_fn is not None:
                def replace_fn(x):
                    return key_name + "_" + table_key_replace_fn(x)
            else:
                def replace_fn(x):
                    return key_name + "_" + x
            t_fn = transform_fns[key]
            obs += hdf5_tables_to_structs(dset_path, table_keys=table_keys, file=file, transform_fns=t_fn, table_name_replace_fn=replace_fn)
        else:
            obs.append(StructuredArray(group_path, table_key=None, transform_fn=t_fn, name=key_name))

    return obs
