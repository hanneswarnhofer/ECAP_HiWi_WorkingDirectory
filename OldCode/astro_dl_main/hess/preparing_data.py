from hess.event import EventBrowser
import matplotlib.pyplot as plt
from data import manage
import numpy as np


def plot_intensity(file, name=""):
    for i in range(1, 6):
        x = file["dl1/event/telescope/images/tel_00%i" % i][:]
        x = np.stack(np.stack(x.tolist(), axis=0)[:, 3].tolist(), axis=0)
        m = np.mean(x, axis=-1) > -998
        images = x[m]

        fig, ax = plt.subplots(1, figsize=(11.5, 9))
        ax.hist(images.flatten(), bins=100)
        ax.set_yscale("log")
        ax.set_ylabel(r"$entries$")
        ax.set_xlabel(r"$pixel\;signal CT%i$" % i)
        fig.savefig(r"./pe_intensity_%s_ct%i.png" % (name, i))


path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"

dm_proton = manage.DataManager(path_proton)
f_proton = dm_proton.get_h5_file()

plt.close("all")
Plot = EventBrowser(f_proton)
Plot.plotter()
Plot.draw_buttons()
plt.show()


def sig_norm(pixels):
    x = np.clip(pixels, 0, None)
    return np.log10(1 + x)


path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"

dm_gamma = manage.DataManager(path_gamma)
f_gamma = dm_gamma.get_h5_file()

plt.close("all")

Plot = EventBrowser(f_proton, map_fn=sig_norm)
Plot.plotter()
Plot.draw_buttons()
plt.show()

plt.close("all")
