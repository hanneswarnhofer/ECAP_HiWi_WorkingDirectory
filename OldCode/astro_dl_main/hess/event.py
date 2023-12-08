import numpy as np
import h5py
from hess.config import config
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))


class EventBrowser:
    def __init__(self, file, map_fn=lambda x: x):

        self.fig, self.axes, = self.make_hess_figure()
        self.caxes = []
        self.map_fn = map_fn

        if type(file) == str:
            file = h5py.File(file, "r")

        assert type(file) == h5py.File or type(file) == dict, "please input a h5py file or data_dict to the EventBrowser"

        self.data_container = file
        self.cbar = []
        self.idx = 0
        self.ct14_geo, self.ct5_geo = config.make_hess_geometry()
        self.n_samples = None
        self.img_pos_x, self.img_pos_y = None, None
        self.ct_imgs = self.open_imgs()
        # self.plotter()
        # self.draw_buttons()

    def save_geo(self):
        import json
        from utils import NumpyEncoder
        self.ct14_geo = self.data_container["configuration/instrument/telescope/camera/geometry_0"][:]
        self.ct5_geo = self.data_container["configuration/instrument/telescope/camera/geometry_1"][:]
        data_dict = {"ct14_geo": self.ct14_geo, "ct5_geo": self.ct5_geo}
        with open('./' + 'geometry.json', 'w') as f:
            json.dump(data_dict, f, indent=2, cls=NumpyEncoder)

    def draw_buttons(self):
        axprev = self.fig.add_axes([0.7, 0.03, 0.075, 0.05])
        axnext = self.fig.add_axes([0.81, 0.03, 0.075, 0.05])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next_)

        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev_)

        self.fig.canvas.mpl_connect(bprev, self.prev_)
        self.fig.canvas.mpl_connect(bnext, self.next_)

    def make_hess_figure(self):
        fig = plt.figure(figsize=(14, 6))

        gs1 = GridSpec(2, 4)  # , left=0.05, right=0.45, wspace=0.05)
        ax1 = fig.add_subplot(gs1[:1, :1])
        ax2 = fig.add_subplot(gs1[:1, 1:2])
        ax3 = fig.add_subplot(gs1[1:2, :1])
        ax4 = fig.add_subplot(gs1[1:2, 1:2])

        ax5 = fig.add_subplot(gs1[:, 2:4])
        axes = np.array([ax1, ax2, ax3, ax4, ax5])
        self.gs1 = gs1

        for i, ax in enumerate(axes):
            ax.set_aspect("equal")
            ax.set_title('CT%i' % (i + 1))
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        return fig, axes

    @property
    def keys(self):
        self.tree.keys()

    def get_reco_params(self):
        return

    def nan_empty_tels(self, x):
        m = x < -900
        x[m] = np.nan
        return x

    def open_imgs(self):

        if type(self.data_container) == h5py.File:
            rel_path = self.data_container["dl1/event/telescope/images"]
            ct_imgs = {"ct%i" % k: rel_path["tel_00%i" % k] for k in range(1, 6)}
        elif type(self.data_container) == dict:
            ct_imgs = self.data_container
        else:
            raise TypeError("File type not supported, has to be dict or h5py.File")
        self.n_samples = ct_imgs["ct1"].shape[0]
        return ct_imgs

    def prev_(self, event):
        self.idx = np.max((0, self.idx - 1))
        self.plotter()
        # flush and redraw
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def next_(self, event):
        self.idx = np.min((self.n_samples, self.idx + 1))
        self.plotter()
        # flush and redraw
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def img2pos(self, data, camera):

        if self.img_pos_x is None:
            from data.image_mapper import ImageMapper
            pixel_positions = {"HESS-I": self.ct14_geo.get_pix_pos(), "FlashCam": self.ct5_geo.get_pix_pos()}
            camera_types = ["HESS-I", "FlashCam"]
            mapping_method = {k: "axial_addressing" for k in camera_types}

            self.mapper = ImageMapper(camera_types=camera_types, pixel_positions=pixel_positions,
                                      mapping_method=mapping_method)

            self.img_ct14_pos_x = self.mapper.map_image(self.ct14_geo.pos_x[:, np.newaxis], camera_type="HESS-I")[5:, ...].T
            self.img_ct14_pos_y = self.mapper.map_image(self.ct14_geo.pos_y[:, np.newaxis], camera_type="HESS-I")[5:, ...].T

            self.img_ct5_pos_x = self.mapper.map_image(self.ct5_geo.pos_x[:, np.newaxis], camera_type="FlashCam")
            self.img_ct5_pos_y = self.mapper.map_image(self.ct5_geo.pos_y[:, np.newaxis], camera_type="FlashCam")

        if camera == "HESS-I":
            x, y = self.img_ct14_pos_x, self.img_ct14_pos_y
        elif camera == "FlashCam":
            x, y = self.img_ct5_pos_x, self.img_ct5_pos_y
        else:
            raise KeyError("camera has to be of Type 'HESS-I' or 'FlashCam'")
        x, y, data = [u.squeeze().flatten() for u in [x, y, data]]
        m_x = x == 0
        m_y = y == 0
        m = m_x * m_y  # mask all empty image pixels (padded pixels)
        return x[~m], y[~m], data[~m]

    def plotter(self, idx=None):
        idx = self. idx if idx is None else idx
        # for ax in self.axes:
        #
        #     if ax.get_label() == '<colorbar>':
        #         ax.clear()
        #         # ax.remove()
        print("plot image:", idx)
        vmin = self.map_fn(-5)

        for i in range(0, 4):
            self.axes[i].clear()

            if type(self.data_container) == h5py.File:
                data = self.ct_imgs["ct%i" % (i + 1)][idx][3]
            else:
                data = self.ct_imgs["ct%i" % (i + 1)][idx]

            c = self.map_fn(self.nan_empty_tels(data))

            if len(c.shape) >= 2:
                # images are insertet

                x, y, c = self.img2pos(c, "HESS-I")
            else:
                # data is vector
                x, y = self.ct14_geo.pos_x, self.ct14_geo.pos_y

            sct_plt = self.axes[i].scatter(x, y,
                                           c=c, s=10, cmap='Reds', vmin=vmin)  # , norm=colors.LogNorm())

            try:
                self.caxes[i].clear()
                plt.colorbar(sct_plt, cax=self.caxes[i])
            except IndexError:
                plt.colorbar(sct_plt, ax=self.axes[i])
                self.caxes.append(self.fig.get_axes()[-1])

        self.axes[4].clear()

        if type(self.data_container) == h5py.File:
            data = self.ct_imgs["ct5"][idx][3]
        else:
            data = self.ct_imgs["ct5"][idx]

        c = self.map_fn(self.nan_empty_tels(data))

        if len(c.shape) >= 2:
            # images are insertet
            x, y, c = self.img2pos(c, "FlashCam")
        else:
            # data is vector
            x, y = self.ct5_geo.pos_x, self.ct5_geo.pos_y

        sct_plt = self.axes[4].scatter(x, y, c=c, s=20, cmap='Reds', vmin=vmin)  # , norm=colors.LogNorm())
        try:
            self.caxes[4].clear()
            plt.colorbar(sct_plt, cax=self.caxes[4])
        except IndexError:
            plt.colorbar(sct_plt, ax=self.axes[4])
            self.caxes.append(self.fig.get_axes()[-1])

        # plt.show()

        self.fig.tight_layout()
        # plt.show()

# def plotter(self, c="red"):
    #   x1 = np.random.choice(range(0,5), 2)
    #  y1 = np.random.choice(range(0,5), 2)
    #  print(x1, y1)
    #  # clear the ax and use new data
    #  self.ax.clear()
    #  self.ax.plot(x1, y1, c=c)


# plt.close("all")
# Plot = EventBrowser(f)
# Plot.plotter()
# Plot.draw_buttons()
# plt.show()
