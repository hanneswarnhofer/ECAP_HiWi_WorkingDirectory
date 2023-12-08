from os import makedirs, path as osp
import numpy as np
from tools.progress import Progbar
from tools.utils import to_list
import matplotlib as mpl
from matplotlib import pyplot as plt
from plotting.utils import bin_to
from plotting.base import Plotter


class Predictor:
    """ Underlying class for performing predictions using data container and trained models.

    Parameters
    ----------
    batch_size : int
        batch size used for inference. Note, small sizes can result in long evaluation times, too large in huge RAM or rather GRAM consumption.

    Attributes
    ----------
    batch_size

    """

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def predict_torch(self, model, dataset, batch_size=None):
        from tools.utils import is_interactive
        batch_size = self.batch_size if batch_size is None else batch_size

        exclude_keys = model.get_exclude_label_keys(dataset.labels.keys())
        loader = dataset.torch_loader(batch_size, exclude_keys, shuffle=False, follow_batch=model.inputs if dataset.dtype == "pyg" else None)

        prog = Progbar(loader.__len__(), verbose=1 if is_interactive() else 2)
        outputs = {k: [] for k in model.outputs}
        model.eval()  # set eval flag for BN and Dropout

        for i, data in enumerate(loader):
            inputs, _ = model.batch2dict2device(data)
            out = model.inference(inputs)
            prog.add(1)

            for k, val in outputs.items():
                val.append(out[k].cpu().detach().numpy())

        for k, val in outputs.items():
            outputs[k] = np.concatenate(val).squeeze()

        return outputs

    def predict_tf(self, model, dataset, batch_size=None):
        from tools.utils import to_list
        from tools.utils import is_interactive
        batch_size = self.batch_size if batch_size is None else batch_size

        exclude_keys = model.get_exclude_label_keys(dataset.labels.keys())
        mapped_dataset = dataset.tf_loader(batch_size, exclude_keys, shuffle=False)

        y_pred = model.predict(mapped_dataset, verbose=1 if is_interactive() else 2)
        return {k: val.squeeze() for k, val in zip(model.output_names, to_list(y_pred))}

    def predict(self, model, dataset, batch_size=None):
        import models
        batch_size = self.batch_size if batch_size is None else batch_size
        print("Perform inference on the Dataset %s using model %s" % (dataset.name, model.name))
        if isinstance(model, models.torch.base.BaseModel):
            return self.predict_torch(model, dataset)
        else:
            from tensorflow import keras
            if isinstance(model, keras.models.Model):
                return self.predict_tf(model, dataset)
            else:
                raise TypeError("Given model has strange type", model)

    def __call__(self, model, dataset, batch_size=None):
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
        return self.predict(model, dataset, batch_size)


class ExperimentSettings():

    def __init__(self, **kwargs):
        pass


# SWGO = ExperimentSettings({"Energy": [u.TeV], "Zenith": [], "Azimuth": []})


class EvalPlotter(Plotter):
    def __init__(self, metric, figsize=(8, 5), log_dir="./", name="", formats=["png"], **plt_kwargs):
        super().__init__(figsize=figsize, log_dir=log_dir, name=name, formats=formats, **plt_kwargs)
        self.metric = metric

    def set_labels(self, task_name):
        self.set_labels(self.task_name, self.ax)

    def finalize_figure(self, task_name, stats=None, obs=None, log_x=False, log_y=False, path_suffix=""):
        if stats is not None:
            self.plotter.add_statbox(stats, self.metric)

        if obs is None:
            self.metric.set_labels(task_name, self.ax)
            self.ax.set_title(self.metric.name)
        else:
            self.ax.set_ylabel(self.metric.name)
            self.ax.set_xlabel(obs)
            self.ax.set_title(task_name)

        self.ax.legend()

        if log_x is True:
            self.set_xscale("log", nonpositive="clip")

        if log_y is True:
            self.set_yscale("log", nonpositive="clip")

        if path_suffix == '':
            self.save(obs=obs)
        else:
            new_path = osp.join(self.log_dir, path_suffix)
            makedirs(new_path, exist_ok=True)
            self.save(obs=obs, log_dir=new_path)

    def set_xscale(self, value, **kwargs):
        self.ax.set_xscale(value, **kwargs)
        if value == "log":
            x_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
            self.ax.xaxis.set_major_locator(x_major)
            x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
            self.ax.xaxis.set_minor_locator(x_minor)
            self.ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    def set_yscale(self, value, **kwargs):
        self.ax.set_yscale(value, **kwargs)
        if value == "log":
            y_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
            self.ax.yaxis.set_major_locator(y_major)
            y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
            self.ax.yaxis.set_minor_locator(y_minor)
            self.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    def save(self, log_dir=None, obs=False):
        if obs is None:
            name = self.name + ""
        else:
            name = self.name + "_%s_dep" % obs

        return super().save(log_dir=log_dir, name=name)

    def add_statbox(self, stats, metric):
        from matplotlib.offsetbox import AnchoredText

        if metric.vis_fn == plt.hist:
            loc = "upper right"
        elif metric.vis_fn == plt.scatter or metric.vis_fn == plt.hexbin:
            loc = "lower right"
        else:
            loc = "upper left"

        anc_text = AnchoredText(stats, loc=loc)
        self.ax.add_artist(anc_text)


class Evaluator():
    """ Basic class of the evaluation of supervised neural networks.

            Parameters
            ----------
            model : trained DNN model
                trained model (Keras / Torch / or PyG model)
            data : list
                list of DataContainers
            tasks : dict
                Dictionary of tasks specifying the reconstruction {"primary": "classification", "energy": "regression"}
            experiment : str
                Should special

            Returns
            -------
            type
                Description of returned object.

            """

    def __init__(self, model, data, tasks, log_dir="./", class_kwarg=None, experiment=None, stats=True, figsize=(8, 5), formats=["png"]):

        self.model = model
        self.data = to_list(data)
        self.tasks = tasks
        self.experiment = experiment
        self.figsize = figsize
        self.class_kwarg = class_kwarg
        self.plotter = {}
        self.log_dir = log_dir
        self.stats = stats
        self.formats = self.check_tex_install(formats)
        print("Plot using the formats: %s" % self.formats)

    def check_tex_install(self, formats):
        import matplotlib as mpl
        from plotting.style import test_tex_install
        supported_formats = test_tex_install(formats=formats)

        if len(supported_formats) == 0:
            if mpl.rcParams["text.usetex"] is True:
                print("WARNING: no tex installation for the selected formats was found. Undo tex-formatting of mplrc.")
                mpl.rc('text', usetex=False)
                supported_formats = test_tex_install(formats=formats)

        if len(supported_formats) == 0:
            print("!!!!!!\nMatplotlib was not able to generate figures with the given formats %s\n!!!!!!" % formats)
            print("Try to use .png as backup")
            supported_formats = test_tex_install(["png"])

            if len(supported_formats) == 0:
                print("Non working format was found")
        else:
            formats = supported_formats

        return formats

    def evaluate(self, task_names=None, metric_names=None, log_x=False, log_y=False):
        return self.evaluate_(task_names, metric_names, obs=None, log_x=log_x, log_y=log_y)

    def evaluate_(self, task_names, metric_names=None, obs=None, obs_bins=None, log_x=False, log_y=False):
        # tasks = to_list(tasks) if tasks is not None else self.tasks.keys()
        # tasks = {task: t_type for task, t_type in self.tasks.items() if task in tasks}
        task_names = [t.name for t in self.tasks] if task_names is None else to_list(task_names)

        for learning_task in self.tasks:
            task_name = learning_task.name
            if task_name not in to_list(task_names):
                continue

            print("* * * * * * * * * * * * *\n Evaluate %s performance for %s" % (learning_task.name, task_name))
            print("* * * * * * * * * * * * *")
            self.plotter[task_name] = {}

            if metric_names is not None:
                metric_names = [m.lower() for m in to_list(metric_names)]
                metric_coll = []

                for m in to_list(metric_names):
                    try:
                        metric_coll.append(learning_task.metric_dict[m])
                    except KeyError:
                        print("metric %s was not found in metrics of the %s task" % (m, learning_task.name))
            else:
                metric_coll = learning_task.metrics

            for metric in metric_coll:
                print("  --  Create figures for %s" % metric.name)
                log_dir = osp.join(self.log_dir, task_name + "_" + metric.name)
                makedirs(log_dir, exist_ok=True)

                plotter_all = EvalPlotter(metric, log_dir=log_dir, name="%s_all" % metric.name, formats=self.formats, figsize=self.figsize)
                self.plotter[task_name][metric.name] = {"all": plotter_all}

                for dset in self.data:
                    print("      -- Plotting %s" % dset.name)
                    if dset.predictions is None:
                        dset.predict(self.model, dset)

                    y_pred, y_true = dset.y_pred[task_name].squeeze(), dset.y_true[task_name].squeeze()
                    plotter = EvalPlotter(metric, log_dir=log_dir, name="%s_%s" % (metric.name, dset.name), formats=self.formats, figsize=self.figsize)
                    self.plotter[task_name][metric.name][dset.name] = plotter

                    # plot for each data set
                    if obs is None:
                        stats = metric.plot(plotter.ax, y_true, y_pred, dset.name, **dset.plt_kwargs)
                    else:  # for observable-dependent plotting
                        obs_vals = dset.y_true[obs].squeeze()
                        stats = metric.plot_obs_dep(y_true, y_pred, obs_vals, obs_bins, plotter.ax, dset.name,
                                                    **dset.plt_kwargs)

                    plotter.finalize_figure(task_name, stats, obs, log_x=log_x, log_y=log_y)

                    # combined plot (over data sets)
                    if obs is None:
                        metric.plot(plotter_all.ax, y_true, y_pred, dset.name)  # , **dset.plt_kwargs)  # * converts np array into tuple
                    else:  # for obs dependent plotting
                        metric.plot_obs_dep(y_true, y_pred, obs_vals, obs_bins, plotter_all.ax, dset.name,
                                            **dset.plt_kwargs)

                        _, y_true_binned, edges = bin_to(obs_vals, y_true, obs_bins)
                        _, y_pred_binned, _ = bin_to(obs_vals, y_pred, obs_bins)

                        # make metric plot per energy bin
                        for i, (y_p_bin, y_t_bin, edge) in enumerate(zip(y_pred_binned, y_true_binned, edges)):

                            low, up = edge
                            print("         -- %s bin %.3f --> %.3f " % (obs, low, up))

                            bin_plotter = EvalPlotter(metric, log_dir=log_dir, name="%s_%s_%s_%.1f_to_%.1f" % (metric.name, dset.name, obs, low, up), formats=self.formats, figsize=self.figsize)
                            stats = metric.plot(bin_plotter.ax, y_t_bin, y_p_bin, dset.name)
                            bin_plotter.finalize_figure(task_name, stats, log_x=log_x, log_y=log_y,
                                                        path_suffix="%s_dependence" % obs)

                if obs is None:
                    plotter_all.finalize_figure(task_name, obs=obs, log_x=log_x, log_y=log_y)
                else:
                    plotter_all.finalize_figure(task_name, obs=obs, log_x=log_x, log_y=log_y, path_suffix=metric.name)

        return self.plotter

    def observable_dep(self, obs, bin_range, task_names=None, metric_names=None, log_x=False, log_y=False,
                       classes=None):
        assert bin_range[2] > 1, "at least a single bin with two bin edges has to be created. Thus, the last number of bin_range has to be > 1."
        if log_x is True:
            bins = np.logspace(np.log10(bin_range[0]), np.log10(bin_range[1]), bin_range[2])
        else:
            bins = np.linspace(*bin_range)

        print("Start %s-dependent plotting ...." % obs)
        return self.evaluate_(task_names, metric_names, obs, bins, log_x=log_x, log_y=log_y)

    def plot(plt_fn, xdata, ydata, xerr=None, yerr=None, **plt_kwargs):
        pass

    def plot_class_perf(self, data):
        for dset in to_list(data):
            pass

    def estimate_performance(self, data, metric, task):
        for dset in data:
            data
            metric

    def plot_regression_perf(self, task):
        self.energy_dep_bias_and_resolution(task)
        self.scatter_perfomance(task)

    def energy_dep_bias_and_resolution(self, task):
        pass

    def scatter_perfomance(self, task):
        pass
