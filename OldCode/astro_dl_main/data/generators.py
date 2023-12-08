import numpy as np
from data import manage
from tools.utils import to_list


class HDFLoader():
    """ Base class for loading HDF data into RAM --> dict """

    def __init__(self, h5_paths, mc_subpath, np_prepro_fn):
        self.h5_paths = to_list(h5_paths)
        self.mc_subpath = mc_subpath
        self.dms = []

        for path in self.h5_paths:
            self.dms.append(manage.DataManager(path))

        self.np_prepro_fn = np_prepro_fn if np_prepro_fn is not None else lambda x: x

    def apply_cuts(self, key, operation):

        def mask(data, operation):
            data_dict, mc_dict = data.tuple
            name = f'{mc_dict=}'.split('=')[0]
            mask = eval(name + operation)

            for k, val in data_dict.items():
                data_dict[k] = val[mask]

            for k, val in mc_dict.items():
                mc_dict[k] = val[mask]

        operation = "['%s']%s" % (key, operation)
        mask(self.train, operation)
        mask(self.test, operation)
        mask(self.valid, operation)

        return self.train, self.valid, self.test

    def extract_info(self, file, path):
        x = file[path]
        y = np.stack(x[:].tolist())
        return {k: y[:, i] for i, k in enumerate(file[path].dtype.names)}

    def make_mc_data(self, data_manager, mc_dict={}):
        # load MC data from h5 file of DataManager into RAM
        if mc_dict == {}:
            with data_manager.get_h5_file() as f:
                mc_dict = {k: [] for k in f[self.mc_subpath].dtype.names}

            mc_dict = {**mc_dict, **{"ct%i_loc_dist" % i: [] for i in range(1, 6)}, "energy_reco": [], "zeta_bdt": []}

        with data_manager.get_h5_file() as f:
            for i in range(1, 6):
                con = f["/dl1/event/telescope/parameters/tel_00%i" % i][:]
                ldis = np.stack(con[:].tolist()).astype(np.float32)[:, 8]
                mc_dict["ct%i_loc_dist" % i].append(ldis)

        with data_manager.get_h5_file() as f:
            print("extract mc information of file:\n", f.filename)
            con = f[self.mc_subpath][:]
            try:
                reco = f["/dl1/event/telescope/other_parameters"]["energy_reco"]
                mc_dict["energy_reco"].append(reco)
            except (ValueError, KeyError):
                if "energy_reco" in mc_dict.keys():
                    mc_dict.pop("energy_reco")

            try:
                zeta = f["/dl1/event/telescope/other_parameters"]["zetabdt"]
                mc_dict["zetabdt"].append(zeta)
            except (ValueError, KeyError):
                if "zetabdt" in mc_dict.keys():
                    mc_dict.pop("zetabdt")

        data = np.stack(con[:].tolist()).astype(np.float32)

        for i, k in enumerate(con.dtype.names):
            mc_dict[k].append(data[:, i])

        return mc_dict

    def get_train_val_test_indices(self, nsamples, test_frac, val_frac):
        np.random.seed(1)
        idx = np.arange(nsamples)
        np.random.shuffle(idx)
        n_val = int(nsamples * val_frac)
        n_test = int(nsamples * test_frac)
        n_train = nsamples - n_val - n_test
        return idx[: n_train], idx[n_train: n_test + n_train], idx[n_test + n_train:]

    def split(self, data_dict, idx_train, idx_val, idx_test):
        data_dict_test, data_dict_val = {}, {}
        nsamples = len(idx_train) + len(idx_val) + len(idx_test)

        for k, val in data_dict.items():
            if len(val) != nsamples:  # for unchanged features (reduce memory consumption), e.g., pixel positions
                data_dict_test[k] = val
                data_dict_val[k] = val
                data_dict[k] = val
            else:
                data_dict_test[k] = val[idx_test]
                data_dict_val[k] = val[idx_val]
                data_dict[k] = val[idx_train]

        return data_dict, data_dict_val, data_dict_test
