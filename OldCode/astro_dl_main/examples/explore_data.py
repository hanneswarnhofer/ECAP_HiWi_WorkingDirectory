import matplotlib.pyplot as plt
from data import manage, generators
from hess.event import EventBrowser

path_proton = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"

path_gamma = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_gamma_20deg_0deg_0.0off_cone5.h5"

dm = manage.DataManager(path_gamma)

f = dm.get_h5_file()

plt.close("all")
Plot = EventBrowser(f)
Plot.plotter()
Plot.draw_buttons()
plt.show()


pip = generators.HESS([path_proton, path_gamma])
train_data, test_data = pip.make_training_data(test_split=0.15)
img_dict, mc_dict = train_data

Plot = EventBrowser({**img_dict, **mc_dict})
Plot.plotter()
Plot.draw_buttons()
plt.show()
