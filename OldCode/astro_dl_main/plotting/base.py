import matplotlib.pyplot as plt
from os import makedirs, path as osp
import numpy as np
from plotting import style, utils as pu


class PlottingFactory():

    def __init__(self, data, log_dir="./", ci=68, formats=["png"]):
        """Class for the visualization of datasets. Use default functions for easy plotting of relationships.


        Args:
            data (DataContainer, tuple(dict, dict), dict): Data object to be visualized.
            log_dir (str, optional): Logging directory to produce plots. Defaults to "./".
            ci (int, optional): Confidence Intervall for uncertainty estimation. Defaults to 68.
            formats (list, optional): file format. Defaults to ["png"].
        """
        self.data = data
        self.formats = formats
        self.log_dir = log_dir
        self.ci = ci

    def create_plt(self, fig, width, height):
        """Create Figures

        Args:
            fig (plt.Figure): Hand figure to function to use same plot, else pass None
            width (float): width of the figure
            height (float): height of the figure

        Returns:
            fig, axes: plt.fig and plt.axes of the defined object
        """
        if fig is None:
            return plt.subplots(1, figsize=(width, height))
        else:
            return fig, fig.get_axes()[0]

    def set_lims(self, ax, xlim, ylim):
        if ylim is not None:
            ax.set_ylim(*ylim)

        if xlim is not None:
            ax.set_xlim(*xlim)

    def bin_and_btrp(self, x, vals, x_bins=None, fn=np.mean, ci=None, n_events=False):
        ''' Assume you have events which have property x and vals.
            Bin x in "x_bins" and apply function "fn" for the values "vals" in the obtained bins '''

        ci = self.ci if ci is None else ci
        return pu.bin_and_btrp(x, x_bins, vals, fn=fn, ci=ci, n_events=n_events)

    def apply_legend(self, ax):
        ax.legend()

    def plot_label_distributions(self):
        labels = self.data.labels
        from data.observables import Observable, observable_from_label_key
        fig_dict = {}

        for label_key, values in labels.items():
            if type(label_key) != Observable:
                obs = observable_from_label_key(label_key)
                fig, ax, _ = self.x_binned_1d(obs, name=label_key)
            else:
                fig, ax, _ = self.x_binned_1d(values, name=obs.name)

            fig_dict[obs.name] = fig

        return fig_dict

    def apply_labels(self, ax, x, y, log_x=False, log_y=False, btrp_fn=np.mean):

        if len(ax.get_xaxis().get_label().get_text()) == 0:
            if x.unit != '':
                unit = (r"/\;" + x.unit) if x.unit != "" else ""
                ax.set_xlabel(style.to_tex(r"$\mathit{%s}\;\mathrm{%s}$" % (x.label, unit)))
            else:
                ax.set_xlabel(style.to_tex(r"$\mathit{%s}$" % x.label))

        if len(ax.get_yaxis().get_label().get_text()) == 0:
            if y == "":
                ax.set_ylabel(style.to_tex("Events"))
            else:
                unit = r"/\;" + y.unit if y.unit else ""
                if btrp_fn == np.std:
                    ax.set_ylabel(style.to_tex(r"\sigma(\mathit{%s})\;%s" % (y.label, unit)))
                else:
                    ax.set_ylabel(style.to_tex(r"$\mathit{%s}\;\mathrm{%s}$" % (y.label, unit)))

    def _get_data(self, x):
        ''' Function to obtain data from HDF file.
            Input can be tuple of strings that point to HDF5 datasets ("sd_energy"), to observable ("n_stations")
            or self defined observables (using data.observables.Observable)

            return tuple of data
        '''

        return x.call_data_name_suffix(self)

    def rm_latex_chars(self, str):
        str_ = str.replace("/", "over")
        str_ = str_.replace(r"\langle", "")
        str_ = str_.replace(r"\rangle", "")
        str_ = str_.replace(r"\mathrm", "")
        str_ = str_.replace(r"\mathit", "")
        str_ = str_.replace(r"\\", "")

        return str_

    def save_figure(self, fig, name, add, log_dir, *strings):
        log_dir = self.log_dir if log_dir is None else log_dir

        if name == "":
            name = "_".join([str_ for str_ in strings])
            name = name.replace(r"\mathrm", "")
            name = name.replace(r"\mathit", "")
            name = name.replace("}", "")
            name = name.replace("{", "")

        name += "_%s" % add
        name_ = name.replace("/", "over")

        for file_type in self.formats:
            fig.savefig(osp.join(log_dir, "%s.%s" % (name_, file_type)))

    def finalize_axes_and_fig(self, fig, ax, x, y, log_x, log_y, btrp_fn=None):

        if log_x:
            ax.set_xscale("log", nonpositive="clip")

        if log_y:
            ax.set_yscale("log", nonpositive="clip")

        self.apply_labels(ax, x, y, log_x, log_y, btrp_fn)
        fig.tight_layout()

    def x_binned_1d(self, x, x_bins=None, log_x=False, log_y=False, width=10, height=7.5, save_fig=True, fig=None,
                    name="", show_bin_edges=True, log_dir=None, xlim=None, ylim=None, **kwargs):
        """Plot histogramm of x binned using x_bins.

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axiss
        bins_x : tuple, bins to bin data x
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        show_bin_edges : bool
            should x bin edges be shown as (vertical) errorbar?

        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        from data.observables import freq
        return self.x_vs_y_binned_1d(x, freq, x_bins, btrp_fn=np.sum, log_x=log_x, log_y=log_y,
                                     width=width, height=height, save_fig=save_fig, fig=fig, name=name, xlim=xlim, ylim=ylim, show_bin_edges=show_bin_edges, log_dir=log_dir, **kwargs)

    def x_vs_y(self, x, y, plt_fn, log_x=False, log_y=False, save_fig=True, fig=None, name="", width=9.5,
               height=7, statbox="small", log_dir=None, xlim=None, ylim=None, **kwargs):
        """Short summary.

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        fig, ax = self.create_plt(fig, width, height)
        x_vals = x(self.data)
        y_vals = y(self.data)

        plt_fn(x_vals, y_vals, axes=ax, **kwargs)

        if statbox:
            unit = x.unit if x.unit == y.unit else ""
            stat_text = pu.get_statbox(x_vals, y_vals, unit=unit, name="", complete=(statbox == "full"))
            ax.text(0.95, 0.3, stat_text, verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes, backgroundcolor="none")

        self.set_lims(ax, xlim, ylim)
        self.finalize_axes_and_fig(fig, ax, x, y, log_x, log_y)

        if save_fig:
            self.save_figure(fig, name, plt_fn.__name__, log_dir, x.str, y.str)

        return fig, ax

    def x_vs_y_binned_1d(self, x, y, bins_x, btrp_fn=np.mean, plt_fn=plt.errorbar, log_x=False, log_y=False,
                         width=10, height=7.5, save_fig=True, fig=None, name="", show_bin_edges=True, log_dir=None, xlim=None, ylim=None, **kwargs):
        """Plot x vs. y in bins of x .

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        bins_x : tuple, bins to bin data x
        btrp_fn : pyfunc, function to apply bootstrapping (default: np.mean)
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        show_bin_edges : bool
            should x bin edges be shown as (vertical) errorbar?

        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        fig, ax = self.create_plt(fig, width, height)
        x_vals = x(self.data)
        y_vals = y(self.data)

        com_x, btrp_y, btrp_y_err, edges = self.bin_and_btrp(x_vals, y_vals, bins_x, fn=btrp_fn)

        if show_bin_edges:
            edges_as_err = np.abs(edges - com_x[:, np.newaxis])
            ax.errorbar(com_x, btrp_y, xerr=edges_as_err.T, yerr=None, ecolor="grey", fmt="none")

        plt_fn(com_x, btrp_y, xerr=None, yerr=np.abs(btrp_y_err), axes=ax, capsize=5, elinewidth=1, **{'fmt': 'o',
                                                                                                       **kwargs})

        self.set_lims(ax, xlim, ylim)
        self.finalize_axes_and_fig(fig, ax, x, y, log_x, log_y, btrp_fn)

        if save_fig:
            self.save_figure(fig, name, "binned", log_dir, x.str, y.str)

        return fig, ax, (com_x, btrp_y, btrp_y_err)

    def xyz_hexbin(self, x, y, z, bins_x, bins_y, btrp_fn=np.mean, plt_fn=plt.hexbin, log_x=False, log_y=False, width=10, height=7.5, save_fig=True, fig=None, name="", log_dir=None, xlim=None, ylim=None, **kwargs):
        """Plot x vs. y in bins of x .

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        bins_x : tuple, bins to bin data x
        btrp_fn : pyfunc, function to apply bootstrapping (default: np.mean)
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        show_bin_edges : bool
            should x bin edges be shown as (vertical) errorbar?

        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = self.create_plt(fig, width, height)
        x_vals = x(self.data)
        y_vals = y(self.data)
        z_vals = z(self.data)

        xscale = "linear" if log_x is False else "log"
        yscale = "linear" if log_y is False else "log"

        plot = plt.hexbin(x_vals, y_vals, z_vals, xscale=xscale, yscale=yscale, axes=ax, reduce_C_function=btrp_fn, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)

        if btrp_fn == np.std:
            label_z = style.to_tex(r"\sigma(%s) / %s" % (z.label, z.unit))
        else:
            label_z = style.to_tex(r"%s / %s" % (z.label, z.unit))

        fig.colorbar(plot, extend="both", cax=cax, orientation="vertical", label=label_z)
        self.set_lims(ax, xlim, ylim)
        self.finalize_axes_and_fig(fig, ax, x, y, log_x, log_y)

        if save_fig:
            self.save_figure(fig, name, "binned", log_dir, x.str, y.str)

        return fig, ax

    def x_vs_y_binned_in_z(self, x, y, z, bins_z, plt_fn=plt.hexbin, log_x=False, log_y=False, width=9.5,
                           height=7, save_fig=True, figs=None, name="", statbox="small", log_dir=None, xlim=None, ylim=None, **kwargs):
        """Short summary.

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        x_vals = x(self.data)
        y_vals = y(self.data)
        z_vals = z(self.data)

        com_zs, x_vals_in_zs, edges = pu.bin_to(z_vals, x_vals, bins_z)
        com_zs, y_vals_in_zs, edges = pu.bin_to(z_vals, y_vals, bins_z)
        raw_figs = [None for i in enumerate(com_zs)] if figs is None else figs
        assert len(raw_figs) == len(com_zs), "expect %i figures but got %i figures" % (len(com_zs), len(raw_figs))
        figs = []

        for com_z, x_vals_in_z, y_vals_in_z, edges_, fig in zip(com_zs, x_vals_in_zs, y_vals_in_zs, edges, raw_figs):

            fig, ax = self.create_plt(fig, width, height)
            plt_fn(x_vals_in_z, y_vals_in_z, axes=ax, **kwargs)

            if statbox:
                unit = x.unit if x.unit == y.unit else ""
                header = r"%.2f \leq\;%s\leq %.2f" % (edges_[0], z.unit, edges_[1])
                stat_text = pu.get_statbox(x_vals_in_z, y_vals_in_z, complete=(statbox == "full"), unit=unit,
                                           name=header)
                ax.text(0.95, 0.28, stat_text, verticalalignment='top', horizontalalignment='right',
                        transform=ax.transAxes, backgroundcolor="none")

            self.set_lims(ax, xlim, ylim)
            self.finalize_axes_and_fig(fig, ax, x, y, log_x, log_y)
            figs.append(fig)

            if save_fig:
                if log_dir is None:
                    log_dir = self.make_folder("%s_scatter_%s_binned_in_%s" % (x.str, y.str, z.str))

                self.save_figure(fig, name, "_%.2f_to_%.2f_" % (edges_[0], edges_[1]), log_dir, x.str, y.str)

        return figs, [f.get_axes()[0] for f in figs]

    def x_vs_y_binned_1d_binned_in_z(self, x, y, z, bins_x, bins_z, plt_fn=plt.errorbar, btrp_fn=np.mean,
                                     name="", log_x=False, log_y=False, width=9.5, height=7, save_fig=True, xlim=None, ylim=None, figs=None, statbox="small", show_bin_edges=True, log_dir=None, **kwargs):
        """Bin data in z using z_bins. In each of this bins a x vs y is plotted using a 1D representation.
           The transformation to the 1D representation is to be defined using 'btrp_fn' (Default: np.mean).

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        z : str, data.observables.Observable
            data to plot on z axis
        bins_x : tuple, bins to bin data x
        bins_z : tuple, bins to bin data z
        btrp_fn : pyfunc, function to apply bootstrapping (default: np.mean)
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        log_y : bool
            should y axis be on log scale
        save_fig : bool
            should figure be saved
        show_bin_edges : bool
            should x bin edges be shown as (vertical) errorbar?
        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        x_vals = x(self.data)
        y_vals = y(self.data)
        z_vals = z(self.data)

        # bin data in z
        com_zs, x_vals_in_zs, edges = pu.bin_to(z_vals, x_vals, bins_z)
        com_zs, y_vals_in_zs, edges = pu.bin_to(z_vals, y_vals, bins_z)
        raw_figs = [None for i in enumerate(com_zs)] if figs is None else figs
        assert len(raw_figs) == len(com_zs), "expect %i figures but got %i figures" % (len(com_zs), len(raw_figs))
        figs, stats_list = [], []

        for com_z, x_vals_in_z, y_vals_in_z, edges_, fig in zip(com_zs, x_vals_in_zs, y_vals_in_zs, edges, raw_figs):

            fig, ax = self.create_plt(fig, width, height)

            com_x, btrp_y, btrp_y_err, edges = self.bin_and_btrp(x_vals_in_z, y_vals_in_z, bins_x, fn=btrp_fn)

            if show_bin_edges:
                edges_as_err = np.abs(edges - com_x[:, np.newaxis])
                ax.errorbar(com_x, btrp_y, xerr=edges_as_err.T, yerr=None, ecolor="grey", fmt="none")

            plt_fn(com_x, btrp_y, xerr=None, yerr=np.abs(btrp_y_err), axes=ax, capsize=5, elinewidth=1,
                   **{'fmt': 'o', **kwargs})

            stats_list.append((com_x, btrp_y, btrp_y_err))

            if statbox:
                header = r"%.2f \leq\;%s\leq %.2f" % (edges_[0], z.unit, edges_[1])

                if statbox == "tiny":
                    ax.text(0.1, 0.1, r"$%s$" % header, verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, backgroundcolor="none")
                else:
                    unit = x.unit if x.unit == y.unit else ""
                    stat_text = pu.get_statbox(y_vals_in_z, np.zeros_like(y_vals_in_z), complete=(statbox == "full"), unit=unit, name=header)
                    ax.text(0.95, 0.28, stat_text, verticalalignment='top', horizontalalignment='right',
                            transform=ax.transAxes, backgroundcolor="none")

            self.set_lims(ax, xlim, ylim)
            self.finalize_axes_and_fig(fig, ax, x, y, log_x, log_y, btrp_fn)

            if save_fig:
                if log_dir is None:
                    if name == "":
                        log_dir = self.make_folder("%s_vs_%s_binned_in_%s" % (x.str, y.str, z.str))
                    else:
                        log_dir = self.make_folder(name)

                self.save_figure(fig, name, "binned_%.2f_to_%.2f_" % (edges_[0], edges_[1]), log_dir, x.str, y.str)

            figs.append(fig)

        return figs, [f.get_axes()[0] for f in figs], stats_list

    def x_binned_binned_in_y(self, x, y, bins_x, bins_y, plt_fn=plt.hist, name="", log_x=False, width=9.5,
                             height=7, save_fig=True, figs=None, statbox="small", show_bin_edges=True, log_dir=None, merged=False, ncols=4, nrows=4, **kwargs):
        """ Plot x as function of y.

        Parameters
        ----------
        x : str, data.observables.Observable
            data to plot on x axis
        y : str, data.observables.Observable
            data to plot on y axis
        bins_x : tuple, bins to bin data x
        plt_fn : plt.plotting function
            pyplot plotting function (plt.scatter, plt.hexbin, ...)
        log_x : bool
            should x axis be on log scale
        save_fig : bool
            should figure be saved
        show_bin_edges : bool
            should x bin edges be shown as (vertical) errorbar?
        Returns
        -------
        plt.fig, plt.ax
            return pyplot figure and axis

        """
        x_vals = x(self.data)
        y_vals = y(self.data)

        # bin data in z
        com_ys, x_vals_in_ys, edges = pu.bin_to(y_vals, x_vals, bins_y)
        raw_figs = [None for i in enumerate(com_ys)] if figs is None else figs
        assert len(raw_figs) == len(com_ys), "expect %i figures but got %i figures" % (len(com_ys), len(raw_figs))
        figs = []

        for i, (com_y, x_vals_in_y, edges_, fig) in enumerate(zip(com_ys, x_vals_in_ys, edges, raw_figs)):

            if merged:
                if i == 0 or i == len(raw_figs) - 1:  # don't show over and underlow bin
                    continue
                if i == 1:
                    fig_, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(ncols * (width - 1), (height - 1) * nrows))
                    axes = axes.flatten()

                ax = axes[i - 1]
                method = getattr(ax, plt_fn.__name__)
                method(x_vals_in_y, bins=bins_x, linewidth=2, **kwargs)
            else:
                fig, ax = self.create_plt(fig, width, height)
                plt_fn(x_vals_in_y, bins=bins_x, linewidth=2, axes=ax, **kwargs)

            if statbox:
                ax.set_ylim(0, ax.get_ybound()[1] * 1.25)
                header = r"%.1f\;%s \leq\;\mathit{%s}\leq %.1f\;%s" % (edges_[0], y.unit, y.label, edges_[1], y.unit)
                stat_text = pu.get_statbox(x_vals_in_y, np.zeros_like(x_vals_in_y), x.unit, header, False,
                                           "mean", "std", "samples")
                ax.text(0.95, 0.95, stat_text, verticalalignment='top', horizontalalignment='right',
                        transform=ax.transAxes)

            if log_x:
                ax.set_xscale("log", nonpositive="clip")

            if not merged:
                self.apply_labels(ax, x, "", log_x, False)
                figs.append(fig)
                fig.tight_layout()

                if save_fig:
                    if log_dir is None:
                        log_dir = self.make_folder("distributions_of_%s_binned_in_%s" % (x.str, y.str))

                    self.save_figure(fig, name, "%.2f_to_%.2f" % (edges_[0], edges_[1]), log_dir, x.str)

        if merged:
            ax_invis = pu.get_invisible_common_axis(fig_)
            self.apply_labels(ax_invis, x, "", log_x, False)
            fig_.tight_layout()
            self.save_figure(fig_, name, "merged", ".")
            return fig_, fig_.get_axes(), (com_ys, x_vals_in_ys, edges)

        return figs, [f.get_axes()[0] for f in figs], (com_ys, x_vals_in_ys, edges)

    def add_plot_to_fig(self, *args, plt_fn, fig, **kwargs):
        # self.update_legend
        # self.
        self.plt_fn(*args, **kwargs, ax=fig.get_axes()[0])
        fig.tight_layout()
        return fig


class Plotter():
    def __init__(self, figsize=(8, 5), log_dir="./", name="", formats=["png"], **plt_kwargs):
        self.figsize = figsize
        self.log_dir = log_dir
        self.fig, self.ax = plt.subplots(1, figsize=figsize)
        self.name = name
        self.formats = formats

    def save(self, log_dir=None, name=False):
        log_dir = log_dir if log_dir is not None else self.log_dir

        self.tight_layout()
        for f in self.formats:
            if f == "pgf":
                tex_path = osp.join(log_dir, "tex")
                makedirs(tex_path, exist_ok=True)
                self.fig.savefig(osp.join(tex_path, "%s.pgf" % name))
            elif f == "png":
                self.fig.savefig(osp.join(log_dir, "%s.%s" % (name, f)), dpi=200)
            else:
                self.fig.savefig(osp.join(log_dir, "%s.%s" % (name, f)), dpi=200)

        plt.close(self.fig)

    def set_log(self, log_x=False, log_y=False):
        if log_x is True:
            self.ax.set_xscale("log", nonpositive="clip")

        if log_y is True:
            self.ax.set_yscale("log", nonpositive="clip")

    def tight_layout(self):
        try:
            self.fig.tight_layout()
        except ValueError:
            self.ax.set_yscale("linear")
            try:
                self.fig.tight_layout()
            except ValueError:
                self.ax.set_xscale("linear")
                try:
                    self.fig.tight_layout()
                except ValueError:
                    pass
