from tools.utils import to_list
from plotting.base import Plotter
from data.observables import Observable
import numpy as np


class LabelPlotter():

    def __init__(self, data_container, class_key, log_dir="./", class_names=None, formats=["png"]):
        """Class to study the label distribution. Since for machine learning it is crucial that the label
            distribution for various classes is similar, comparison between both label distributions can be directly performed,
            using the "class_key" argument.

        Args:
            data_container (DataContainer): The dataset to investigate
            class_key (str): The key defining your classes in the dataset.
                             The values are expected to be one-hot enoced or integers with the respective class index.
            log_dir (str, optional): . Defaults to "./".
            class_names (list(str), optional): Names for the classes defined via the "class_key" argument. Defaults to None.
            formats (list, optional): Types of the saved figures. Defaults to ["png"].
        """
        self.dc = data_container
        self.class_key = class_key
        self.log_dir = log_dir
        self.plotter = {}
        self.formats = formats
        self.labels = self.dc.labels
        self.class_masks, self.class_names = self.get_class_masks(class_names)

    def get_class_masks(self, class_names=None):
        """_summary_

        Args:
            class_names (list(str), optional): Names for the classes defined via the "class_key" argument. Defaults to None.

        Returns:
            list(arr(bool)), list(str): Masks for the respecrive class, corresponding class names. If class_names is not
                                        defined, the classes are numerated starting from zero.
        """
        class_data = np.squeeze(self.labels[self.class_key].astype(bool))
        try:
            n_classes = class_data.shape[1]
        except IndexError:
            n_classes = 1

        if n_classes == 1:
            c_keys = np.unique(class_data)
            n_classes = len(c_keys)
            cmasks = [el == class_data for el in c_keys]

            if class_names is None:
                class_names = c_keys
        else:
            cmasks = [split for split in np.split(class_data, n_classes, axis=-1)]
            c_keys = np.arange(n_classes)

        print("found %i classes" % n_classes)
        if class_names is None:
            class_names = ["%s" % c for c in c_keys]
        else:
            assert n_classes == len(class_names), "class_names have to be of same shape then n_classes"

        return to_list(cmasks), class_names

    def plot(self, keys=[], log_dir=None):
        """Handle the plotting of distribution for various labels given as list using the "keys" argument.

        Args:
            keys (list(str), optional): Labels to investigate. Defaults to [].
            log_dir (str, optional): Directory for the plots. Defaults to None.
        """
        log_dir = self.log_dir if log_dir is None else log_dir
        keys = to_list(keys)
        keys = [k for k in keys if k in self.dc.labels.keys()]

        for k in keys:
            self.plot_distribution(k)

    def plot_distribution(self, label_key, log_x=False, log_y=False, bins=None, name=""):
        """Plot distribution for a inputted single label x".

        Args:
            label_key (str): key of labels to plot as distribution.
            log_x (bool, optional): Log scale x-axis?. Defaults to False.
            log_y (bool, optional): Log scale y-axis?. Defaults to False.
            bins (int / tuple(int,int,int), optional): Number of bins or (x_low, x_up, n_bins). Defaults to None.
            name (str, optional): Name of the figure. Defaults to "".

        Returns:
            _type_: _description_
        """
        # from IPython import embed
        # embed()
        if bins is not None:
            if type(bins) == tuple and len(bins) == 3:
                if log_x is True:
                    bins = np.logspace(np.log10(bins[0]), np.log10(bins[1]), bins[2])
                else:
                    bins = np.linspace(*bins)

        plb = Plotter(figsize=(8, 5), log_dir=self.log_dir, name=name, formats=self.formats)
        self.plotter["%s" % label_key] = plb

        if isinstance(label_key, Observable):  # observable
            x = label_key(self.dc)
            plb.ax.set_xlabel(label_key.label)
            name = label_key.name if name == "" else name
        else:  # is string of label key
            x = self.labels[label_key]
            plb.ax.set_xlabel(label_key)
            name = label_key if name == "" else name
        # from IPython import embed
        # embed()
        for i, cmask in enumerate(self.class_masks):
            x_ = x[cmask.squeeze()].squeeze()
            plb.ax.hist(x_, bins=bins, label=self.class_names[i], histtype="step")

        plb.set_log(log_x, log_y)
        plb.ax.set_ylabel("frequency")
        plb.ax.legend()
        plb.save(name=name)

        return plb


def size_fn(feat, labels):
    return np.sum(feat["ct5"], axis=tuple(-np.arange(1, feat["ct5"].ndim)))[:, np.newaxis]


class Feature(dict):

    def __init__(self):
        pass

    def __repr__(self):
        return "lalala"


size = Observable("size", size_fn)

# lp = LabelPlotter(train_data, "primary", log_dir=CONFIG.log_dir, class_names=["photon", "proton"], formats=["png"])
# # lp.plot_distribution("energy", log_x=True, bins=(0.01, 100, 100))
# lp.plot_distribution(size, log_x=False, bins=(1, 1500, 100))
