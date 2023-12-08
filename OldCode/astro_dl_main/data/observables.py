import numpy as np


def observable_from_label_key(label_key):
    def obs_fn(feats, labels):
        return labels[label_key]

    return Observable(obs_fn=obs_fn, name=label_key, label=label_key)


class Observable():
    def __init__(self, obs_fn, name="", label="", unit=""):
        """Class for definig variables to study performance or for performing visualizations.
           Each observables is defined using an observable function 'obs_fn' that defines
           how the final observable is formed using the available feat / label dict.

        Args:
            obs_fn (_type_): Function for defining the observable. Note that the args of the "obs_fn" are the
                             feat, label dicts, which manage the label and feature data of a dataset.
            name (str, optional): Name of the observable. Defaults to "".
            label (str, optional): Label of the observable (for plotting). Defaults to "".
            unit (str, optional): Unit of the observable (for plotting). Defaults to "".
        """
        self.name = name
        self.obs_fn = obs_fn
        self.label = name if label == "" else label
        self.unit = unit
        self.obs1 = None
        self.obs2 = None
        self.str = self.name

    def __repr__(self) -> str:
        return "Observable: " + self.name

    # @property
    # def str(self):
        # str_ = self.label
        # str_ = str_.replace("/", "over")
        # str_ = str_.replace(r"\langle", "")
        # str_ = str_.replace(r"\rangle", "")
        # str_ = str_.replace(r"\mathrm", "")
        # str_ = str_.replace(r"\\", "")
        # str_ = str_.replace(r"{", "")
        # str_ = str_.replace(r"}", "")
        # str_ = str_.replace(r" ", "")
        # str_ = str_.replace(r"_", "")

    def __call__(self, data_container):
        ''' Estimate Observable values using a DataContainer or a feat, label dictionary (type: tuple(dict, dict)).
            Args:
                data (DataContainer, tuple(dict, dict), dict): Data object to be visualized.
                returns:
                    values (arr): Estimate values for the observable given the DataContainer.    
        '''
        from data.data import DataContainer
        if isinstance(data_container, DataContainer):
            feats, labels = data_container()
            data = data_container.dict
        elif type(data_container) == tuple:
            feats, labels = data_container
        elif type(data_container) == dict:
            data = feats = labels = data_container
        else:
            raise TypeError("Given data object has to be of type: DataContainer, tuple(dict, dict), or dict")

        try:
            return self.obs_fn(feats, labels).squeeze()
        except TypeError:
            return self.obs_fn(data).squeeze()


# Useful observables
freq = Observable(lambda x, y: np.ones(x[list(x.keys())[0]].shape[0]), r"freq", label='Events')
freq = Observable(lambda x: np.ones(x[list(x.keys())[0]].shape[0]), r"freq", label='Events')
