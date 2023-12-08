import inspect
import numpy as np
from plotting import utils as pu
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from tools.utils import SearchAttribute


def tf_not_implemented(*args):
    raise NotImplementedError("tf loss not set. Please check your training and task configuration")


def torch_not_implemented(*args):
    raise NotImplementedError("torch loss not set. Please check your training and task configuration")


class Metric:
    def __init__(self, metric_fn, vis_fn=plt.hist, xlabel="x", ylabel="n", num_classes=None, unit=None, name=''):
        """ Underlying metrics class for designing metrics for the evaluation.
        A astro_dl metric is defined by two properties (function):
        - metric_fn: The function to evaluate a scalar value for the metric (e.g. to estimate the accuracy or the resoultion)
        - vis_fn: The function to visualize the metric (e.g., a scatter plot y_pred vs. y_true) for correlation or a 1D hist of y_pred - y_true for a resolution.) Default is: plt.hist(y_pred - y_true).
        For more advanced metric visualizations the plot funciton has to be overwritten.

        For estimating the scalar simply call the metric:
        '
         resolution = Metric(resoultion_fn)
         resolution(y_true, y_pred)
        '

        Parameters
        ----------
        metric_fn : fn
            Metric fn to estimate performance value of model, applied to (y_true, y_pred). E.g., for bias
        vis_fn : fn
            Plotting function to visulaite the metric performance.
        tf_fn : fn
            Set tf/keras metric that should be used with the validation data during training. If None, will not evaluate and not log metric during training. (Check Keras page for simple implementation).
        torch_fn : fn
            Set torch metric that should be used with the validation data during training. If None, will not evaluate and not log metric during training. (Check torchmetrics page for simple implementation).

        Returns
        -------
        type
            Description of returned object.

        """
        self.metric_fn = metric_fn  # numpy based (after training)
        self.vis_fn = vis_fn
        self.name = name if name != '' else metric_fn.__name__.split("_fn")[0]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_classes = num_classes
        self.unit = unit
        self.train_fn_kwargs = {}

    def set_tf_torch_fn(self, search_name):
        search_tf = SearchAttribute(search_name)
        search_tf.search("from models.tf import tf_metrics", not_found_error=False)
        search_tf.search("from tensorflow.keras import metrics as keras_metrics")
        met = search_tf.end()

        if met is not None:  # num_classes at the moment not needed
            self.tf_fn = met  # Not build Metirc here
        else:
            self.tf_fn = tf_not_implemented

        search_torch = SearchAttribute(search_name)
        search_torch.search("from models.torch import torch_metrics", not_found_error=False)
        search_torch.search("from models.torch import torch_metrics")
        met = search_torch.end()

        if met is not None:  # num_classes at the moment not needed
            self.torch_fn = met
        else:
            self.torch_fn = torch_not_implemented

    def __call__(self, y_true, y_pred):
        return self.metric_fn(y_true, y_pred)

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        results = self(y_true, y_pred)
        label = "%s: %.3f" % (name, results)
        self.vis_fn(results, axes=ax, label=label, **plt_kwargs)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("frequency")

    def set_xlabel(self, task_name, ax):
        ax.set_xlabel("%s / %s" % (task_name, self.unit))

    def set_labels(self, task_name, ax):
        self.set_xlabel(task_name, ax)
        self.set_ylabel(task_name, ax)

    def plot_obs_dep(self, y_true, y_pred, obs_vals, bins, ax, name='', show_bin_edges=False, **plt_kwargs):
        results = self(y_true, y_pred)
        label = "%s: %.4f" % (name, results)
        com_x, btrp_y, btrp_y_err, edges = pu.bin_and_btrp(obs_vals, bins, y_true, y_pred, self)

        if show_bin_edges:
            edges_as_err = np.abs(edges - com_x[:, np.newaxis])
            ax.errorbar(com_x, btrp_y, xerr=edges_as_err.T, yerr=0, ecolor="grey", fmt="none", label=name)

        kwargs = {k: val for k, val in plt_kwargs.items() if k != "linestyle"}
        ax.errorbar(com_x, btrp_y, xerr=None, yerr=np.abs(btrp_y_err), axes=ax, capsize=5, elinewidth=1,
                    label=label, **{'fmt': 'o', **kwargs})

    def train_fn(self, dtype):

        if dtype == "tf":
            metric_fn = self.tf_fn
        elif dtype == "torch":
            metric_fn = self.torch_fn
        else:
            raise TypeError("training has to be of type 'torch' or 'tf'.")

        return metric_fn(**self.train_fn_kwargs) if inspect.isclass(metric_fn) is True else metric_fn


class RegressionMetric(Metric):

    def __init__(self, metric_fn, vis_fn=plt.hist, xlabel="x", ylabel="n", unit=None, name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, num_classes=None, unit=unit, name=name)
        search_name = metric_fn.__name__.split("_fn")[0].capitalize()
        self.set_tf_torch_fn(search_name)

    def statistics(self, y_true, y_pred):
        def mse(x):
            return np.mean(x**2)

        stats = {}

        for met in [np.mean, np.std, mse]:
            stats[met.__name__] = met(y_pred - y_true)

        return "\n".join(["%s = %.2f" % (k, val) for k, val in stats.items()])

    def set_xlabel(self, task_name, ax):
        label = "%s_{pred} - %s_{true}" % (task_name, task_name)
        ax.set_xlabel("%s / %s" % (label, self.unit))

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        plt_kwargs = {**plt_kwargs, "bins": 100}
        results = self(y_true, y_pred)
        label = "%s: %.2f" % (name, results)
        self.vis_fn(y_pred - y_true, axes=ax, label=label, **plt_kwargs)


class ClassificationMetric(Metric):

    def __init__(self, metric_fn, num_classes, vis_fn=plt.hist, xlabel="x", ylabel="n", class_labels=[], name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, num_classes=num_classes, unit="prob.", name=name)
        search_name = metric_fn.__name__.split("_fn")[0].capitalize()
        self.set_tf_torch_fn(search_name)
        assert type(class_labels) == list, "'class_labels' has to be of type list and contain the class labels"
        self.class_labels = class_labels
        self.train_fn_kwargs = {"num_classes": self.num_classes}

    def statistics(self, y_true, y_pred):
        from models.metrics import accuracy_fn

        def acc(y_true, y_pred):
            return accuracy_fn(y_true, y_pred)

        stats = {}

        for met in [acc]:
            stats[met.__name__] = met(y_true, y_pred)

        return "\n".join(["%s = %.3f" % (k, val) for k, val in stats.items()])

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        results = self(y_true, y_pred)
        y_true == 1  # divide into num_classes!
        label = "%s: %.3f" % (name, results)

        self.vis_fn(results, axes=ax, label=label, **plt_kwargs)
        return self.statistics(y_true, y_pred)


class ScatterMetric(RegressionMetric):
    def __init__(self, metric_fn, vis_fn=plt.scatter, xlabel="y_true", ylabel="y_pred", unit=None, name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, unit=None, name=name)

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):

        results = self(y_true, y_pred)
        label = "%s: %.2f" % (name, results)
        self.vis_fn(y_true, y_pred, axes=ax, label=label, **plt_kwargs)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("%s_{pred} / %s" % (task_name, self.unit))

    def set_xlabel(self, task_name, ax):
        ax.set_xlabel("%s_{true} / %s" % (task_name, self.unit))


class DistanceMetric(RegressionMetric):
    def __init__(self, metric_fn, agg_fn, vis_fn=plt.hist, xlabel="y_true", ylabel="y_pred", unit=None, name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, unit=unit, name=name)
        self.agg_fn = agg_fn

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        plt_kwargs = {**plt_kwargs, "bins": 100}
        result = self(y_true, y_pred)
        ymin, ymax = ax.get_ylim()
        ax.vlines(result, ymin, ymax)
        label = "%s: %.2f" % (name, result)

        if np.isnan(result).sum() == 0:
            self.vis_fn(self.agg_fn(y_true, y_pred), axes=ax, label=label, **plt_kwargs)


class ROCMetric(ClassificationMetric):
    def __init__(self, metric_fn, num_classes, vis_fn=plt.plot, xlabel="y_true", ylabel="y_pred", name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, num_classes=num_classes, name=name)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("True positive rate")

    def set_xlabel(self, task_name, ax):
        ax.set_xlabel("False positive rate")

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        results = self(y_true, y_pred)
        label = "%s: %.4f" % (name, results)
        fpr, tpr, _ = roc_curve(y_true[:, 0], y_pred[:, 0])  # [1,0] used as signal (True positive)
        self.vis_fn(fpr, tpr, axes=ax, label=label, **plt_kwargs)


class ScoreMetric(ClassificationMetric):
    def __init__(self, metric_fn, num_classes, vis_fn=plt.hist, xlabel="y_true", ylabel="y_pred", name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, num_classes=num_classes, name=name)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("Density")

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        plt_kwargs = {**plt_kwargs, "bins": 100, "density": True, "histtype": "step"}
        results = self(y_true, y_pred)
        label = "%s: %.4f" % (name, results)
        class_0, class_1 = y_true[:, 0] == 1, y_true[:, 1] == 1
        self.vis_fn(y_pred[:, 0][class_0], axes=ax, label=label + " class: 1", **plt_kwargs)
        self.vis_fn(y_pred[:, 0][class_1], axes=ax, label=label + " class: 2", **plt_kwargs)
        ax.set_xlim(0, 1)
        ax.set_yscale("log")


class ConfusionMetric(ClassificationMetric):
    def __init__(self, metric_fn, num_classes, vis_fn=plt.imshow, xlabel="y_true", ylabel="y_pred", name=''):
        super().__init__(metric_fn, vis_fn=vis_fn, xlabel=xlabel, ylabel=ylabel, num_classes=num_classes, name=name)

    def set_ylabel(self, task_name, ax):
        ax.set_ylabel("True Label")

    def set_xlabel(self, task_name, ax):
        ax.set_xlabel("Predicted Label")

    def plot(self, ax, y_true, y_pred, name="", **plt_kwargs):
        n_bins = np.linspace(-0.5, self.num_classes - 0.5, self.num_classes + 1)
        C = np.histogram2d(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), bins=n_bins)[0]
        Cn = C / np.sum(C, axis=1)

        self.vis_fn(Cn, interpolation="nearest", vmin=0, vmax=1, cmap=plt.cm.summer_r, axes=ax)
        plt.colorbar()

        ax.set_xticks(range(self.num_classes), range(self.num_classes), rotation="vertical")
        ax.set_yticks(range(self.num_classes), range(self.num_classes))

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                ax.annotate("%i" % C[x, y], xy=(y, x), ha="center", va="center")

        result = self(y_true, y_pred)
        label = "%s: %.4f" % (name, result)
        ax.plot(0, 0, c="k", marker="", label=label, linestyle="None")
