#from config.config import make_hess_geometry
from dl1_data_handler.image_mapper import ImageMapper
import matplotlib.pyplot as plt
#from data import manage
import numpy as np
import os.path
import inspect
import json

#import hdf5plugin  # noqa
import h5py
import tables



class DataManager():
    """ Data class used to manage the HDF5 data files (simulations + Auger data).
        data_path: data_path of HDF5 file, (hint: use blosc compression to ensure adequate decompression speed,
        to mitigate training bottleneck due to a slow data pipeline)
        params:
            data_path = path to HDF5 datset
        optional params:
            stats: data statistics (stats.json - needed for scaling the dataset)
            tasks: list of tasks to be included (default: ['axis', 'core', 'energy', 'xmax'])
            generator_fn: generator function used for looping over data, generator function needs to have indices and
                          shuffle args.
            ad_map_fn: "advanced mapping function" the function used to map the final dataset. Here an additional
                       preprocessing can be implemented which is mapped during training on the
                       cpu (based on tf.data.experimental.map_and_batch)
    """

    def __init__(self, data_path, stats=None, tasks=['axis', 'impact', 'energy', 'classification']):
        ''' init of DataManager class, to manage simulated (CORSIKA/Offline) and measured dataset '''
        np.random.seed(1)
        self.data_path = data_path

    def open_ipython(self):
        from IPython import embed
        embed()

    @property
    def is_data(self):
        return self.type == "Data"

    @property
    def is_mc(self):
        return self.type == "MC"

    def get_h5_file(self):
        return h5py.File(self.data_path, "r")

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
                        file[key].dtype
                        print(iter_str + str(file[key]))
                    else:
                        print(iter_str + key)
                except AttributeError:
                    print(iter_str + key)
                    walk(file[key], "   " + iter_str)

        with h5py.File(self.data_path, "r") as file:
            print("filename:", file.filename)
            for key in file.keys():
                print(' - ' + key)
                walk(file[key], iter_str='   - ')

    def extract_info(self, path):
        with self.get_h5_file() as f:
            data = f[path]
            y = np.stack(data[:].tolist())

        return {k: y[:, i] for i, k in enumerate(data.dtype.names)}, dict(data.dtype.descr)

    def make_mc_data(self):
        return self.extract_info("simulation/event/subarray/shower")


def get_current_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(filename))


def make_hess_geometry(file=None):
    # quick fix for dl1 data handler to circumvent to use ctapipe
    if file is None:
        with open(os.path.join(get_current_path(), "astro_dl-main/hess/config/geometry2d3.json")) as f: 
            attr_dict = json.load(f)

        data_ct14 = attr_dict["ct14_geo"]
        data_ct5 = attr_dict["ct5_geo"]
    else:
        data_ct14 = file["configuration/instrument/telescope/camera/geometry_0"][:].tolist()
        data_ct5 = file["configuration/instrument/telescope/camera/geometry_1"][:].tolist()

    class Geometry():
        def __init__(self, data):
            self.pix_id, self.pix_x, self.pix_y, self.pix_area = np.stack(data).T.astype(np.float32)
            self.pos_x = self.pix_x
            self.pos_y = self.pix_y

        def get_pix_pos(self):
            return np.column_stack([self.pix_x, self.pix_y]).T

    return Geometry(data_ct14), Geometry(data_ct5)

path="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5"
#path = "/home/woody/caph/mppi067h/gamma_ray_reconstruction_with_ml/gnn/hess_datasets/phase2d3/phase2d3_proton_20deg_0deg_0.0off.h5"
dm = DataManager(path)
f = dm.get_h5_file()
images = f["dl1/event/telescope/images/tel_002"][0]
e = tables.open_file(path, mode="r")

for i in range(1, 5):
    print(i)
    x = e.get_node('/dl1/event/telescope/images/tel_00%i' %i).read()
    #x = f["dl1/event/telescope/images/tel_00%i" % i][:]
    #x = np.stack(np.stack(x.tolist(), axis=0)[:, 3].tolist(), axis=0)
    x = np.stack([data[-1] for data in x])
    m = np.mean(x, axis=-1) > -998
    images = x[m]

    fig, ax = plt.subplots(1, figsize=(11.5, 9))
    ax.hist(images.flatten(), bins=100)
    ax.set_yscale("log")
    ax.set_xlabel("pixel signal CT%i" % i)
    fig.savefig("./pe_intensity_proton_ct%i.png" % i)

plt.close("all")


def extract_info(f, path):
    x = f[path]
    y = np.stack(x[:].tolist())
    return {k: y[:, i] for i, k in enumerate(f[path].dtype.names)}


def plot(data, name):
    for k, val in data.items():
        print("plot", k)
        fig, ax = plt.subplots(1, figsize=(11.5, 9))

        if np.isnan(val.flatten()).sum() == val.flatten().shape[0]:
            print("skip because of nans")
            continue

        ax.hist(val.flatten(), bins=100)
        ax.set_xlabel(k)
        ax.set_ylabel("entries")
        ax.set_yscale("log")
        ax.set_ylim(1, None)
        fig.savefig("./%s_%s.png" % (k, name))


for i in range(1, 6):
    params = extract_info(f, "dl1/event/telescope/parameters/tel_00%i" % i)
    plot(params, name="tel_00%i" % i)

mc = extract_info(f, "simulation/event/subarray/shower")
plot(mc, name="mc")
plt.close("all")


# ct5: 44 x 44
# ct14: 36 x 36


def rotate(pix_pos, rotation_angle=0):
    rotation_angle = rotation_angle * np.pi / 180.0
    rotation_matrix = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                 [np.sin(rotation_angle), np.cos(rotation_angle)], ], dtype=float)

    pixel_positions = np.squeeze(np.asarray(np.dot(rotation_matrix, pix_pos)))
    return pixel_positions


geo_ct14, geo_ct5 = make_hess_geometry()
ct_14_mapper = ImageMapper(camera_types=["HESS-I"], pixel_positions={"HESS-I": rotate(geo_ct14.get_pix_pos())}, mapping_method={"HESS-I": "axial_addressing"})

#ct_5_mapper = ImageMapper(camera_types=["FlashCam"], pixel_positions={"FlashCam": rotate(geo_ct5.get_pix_pos())}, mapping_method={"FlashCam": "axial_addressing"})

test_img_ct14 = f["dl1/event/telescope/images/tel_001"][1][3][:, np.newaxis]
#test_img_ct5 = f["dl1/event/telescope/images/tel_005"][1][3][:, np.newaxis]

test_img_ct14 = np.ones(geo_ct14.pix_id.shape)[:, np.newaxis]
#test_img_ct5 = np.ones(geo_ct5.pix_id.shape)[:, np.newaxis]
test_img_ct14 = np.arange(geo_ct14.pix_id.shape[0])[:, np.newaxis]
#test_img_ct5 = np.arange(geo_ct5.pix_id.shape[0])[:, np.newaxis]


def plot_image(image, name=None):
    fig, ax = plt.subplots(1)
    ax.set_aspect(1)
    ax.pcolor(np.flip(image[:, :, 0], axis=(0)), cmap='viridis', vmin=-5)
    plt.show()
    fig.savefig("./binned_image%s.png" % name)


def re_index_ct14(image):
    return image[5:, :, :]


image_ct14 = ct_14_mapper.map_image(test_img_ct14, "HESS-I")
image_ct14 = re_index_ct14(image_ct14)
plot_image(image_ct14, name="ct14")
print("pixel fraction CT 1-4:", image_ct14.sum() / len(image_ct14.flatten()))


#image_ct5 = ct_5_mapper.map_image(test_img_ct5, "FlashCam")
#plot_image(image_ct5, name="ct5")


# import numpy as np
# from scipy.ndimage import rotate
# rot_img = rotate(image_ct5.squeeze(), angle=45)
# print("pixel fraction CT 5:", rot_img.sum() / len(image_ct5.flatten()))
# plot_image(rot_img[20:-20,10:-10, np.newaxis], name="rot_ct5")
# plt.close("all")


# Area FlashCam
# tan(60) = sqrt(3)
# length hexagonal site = h/2*tan(60) = h/2 * np.sqrt(3)
# area single triangle = 1/2 * h/2 * h/2 * tan(60) = h**2 / 8 * np.sqrt(3)
# maximum activated pixel fraction = 6 / 7 = 0.857
# maximum activated pixel fraction = 6 / 7 = 0.857
