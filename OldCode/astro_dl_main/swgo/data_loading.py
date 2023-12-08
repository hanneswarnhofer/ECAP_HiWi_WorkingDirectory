from data.prepro import ang2vec
from tensorflow.keras.utils import Progbar
from data.data import PointCloud, Array, DataContainer
from pyswgo.data import DataHandler
import numpy as np
from data.prepro import to_one_hot


def load_data(file_list):

    fields = ["mc.logEnergy", "mc.Xmax", "mc.zenithAngle", "mc.azimuthAngle", "mc.coreX", "mc.coreY",
              "mc.corsikaParticleId", "mc.radiusWeight", "mc.eventWeight",
              "event.hit.charge", "event.hit.time", "event.nHit",
              "event.hit.xPMT", "event.hit.yPMT", "event.hit.zPMT",
              'rec.LHLatDistFitEnergy',
              'mc.delCore', 'mc.delAngle', 'mc.coreR'
              ]

    data_container = DataHandler.RecoDataParser(file_list, fields, read_nevents=True)
    data_container.cut_data("event.nHit>30", inplace=True)

    # import awkward as ak
    # builder = ak.ArrayBuilder()
    # builder.begin_list()
    n_samples = len(data_container.data)

    core = Array(np.zeros((n_samples, 2), dtype=np.float32), name="core")
    primary = Array(np.zeros(n_samples, dtype=np.float32), name="primary")
    energy = Array(np.zeros(n_samples, dtype=np.float32), name="energy", unit="GeV")
    shower_axis = Array(np.zeros((n_samples, 3), dtype=np.float32), name="shower_axis")
    xmax = Array(np.zeros(n_samples, dtype=np.float32), name="xmax")

    pc = PointCloud([], features={"feat": []}, name="swgo_pc")

    def log_prepro(x):
        x = np.log10(1 + x)
        return x

    print("Create Point Cloud")
    progbar = Progbar(n_samples)

    for i in range(n_samples):
        event = data_container.data.iloc[i]

        # SAVE MC INPUT
        core[i] = np.stack([event['mc.coreX'], event['mc.coreY']], axis=-1).astype(np.float32)
        primary[i] = event['mc.corsikaParticleId'].astype(np.float32)
        energy[i] = 10**event["mc.logEnergy"].astype(np.float32)
        shower_axis[i] = ang2vec(event["mc.azimuthAngle"], event["mc.zenithAngle"], deg=False)
        xmax[i] = event["mc.Xmax"].astype(np.float32)

        # SAVE FEATURE INPUT
        x, y, z = event['event.hit.xPMT'], event['event.hit.yPMT'], event['event.hit.zPMT']
        pos = np.stack([x, y], axis=-1).astype(np.float32)
        charge, time = event['event.hit.charge'], event['event.hit.time']
        time = (time - time.mean()) / 125  # roughly std of times
        charge = log_prepro(charge) / 75.  # roughly std per coord
        feature = np.stack([time, charge], axis=-1).astype(np.float32)

        # divide into lower and upper
        up = z < 0.065  # use large for floating point issues
        low = ~up
        pos_up, pos_low = pos[up], pos[low]
        feature_up, feature_low = feature[up], feature[low]

        # fill arr with upper PMT info. Fill this first. Will be faster as upper tank is triggered more often!
        true_pos = np.unique(pos, axis=0)
        features = np.zeros((true_pos.shape[0], 4))
        tank_pos = np.ones_like(true_pos) * np.nan
        tank_pos[:up.sum()] = pos_up
        features[:up.sum(), 0:2] = feature_up

        # fill arr with lower PMT info (if not in arr add to arr)
        last = up.sum()
        for plow, feat in zip(pos_low, feature_low):
            m = plow == tank_pos
            m = m.sum(axis=-1) == 2

            if m.sum() == 0:  # upper cell was not triggered
                features[last][2:4] = feat
                tank_pos[last] = plow
                last += 1
            elif m.sum() == 1:
                features[m, 2:4] = feat
            else:
                raise TypeError

        tank_pos -= core[i]  # do it here. Otherwise rounding issues when finding similar station positions
        tank_pos = tank_pos / 125.  # normalize
        pc.append(tank_pos, {"feat": features})

        # if np.array_equal(features[-1], np.array([0., 0., 0., 0.])):
        #     raise NameError

        # if np.isnan(pos).sum() > 0:
        #     raise NameError

        if i % 100 == 0:
            progbar.add(100)

    prims = primary.arr
    prims[prims == 1] = 0
    prims[prims == 14] = 1.
    primary.arr = prims
    primary.arr = to_one_hot(primary(), num_classes=2)
    labels = {l.name: l for l in [core, primary, energy, shower_axis, xmax]}
    feats = {pc.name: pc}
    data = feats, labels

    return DataContainer(data)


# x_std = []
# y_std = []
# std_ = []
# for _ in c:
#     std_.append(_.std())


# np.array(std_).mean()

# for _ in c:
#     x_std.append(_[:, 1].std())
#     y_std.append(_[:, 0].std())

# np.array(x_std).mean()
# np.array(y_std).mean()

# np.std(c, axis=-1)
# np.where(pos_low == pos_up)

# tank_pos, idx, mask = np.unique(pos, axis=0, return_index=True, return_counts=True)

# _, idx_low, idx_up = np.intersect1d(pos_low, pos_up, return_indices=True, axis=0)
# features[:]

# tank_pos, mask = np.unique(pos, axis=0, return_counts=True)

# mask = mask == 2
# features = np.zeros_like(tank_pos)
# features[mask == 1] = charge
# charge = event['event.hit.charge']
# time = event['event.hit.time']
# np.stack([charge[low], charge[low]], axis=-1)

# pos[mask]
# x_ = np.unique(x, return_inverse=True, return_counts=True)
# y_ = np.unique(y, return_inverse=True)

# charge = event['event.hit.charge']
# time = event['event.hit.time']


# event = data_container.data.iloc[0]
# event_data = pd.DataFrame(np.stack(([event['event.hit.xPMT'], event['event.hit.yPMT'], event['event.hit.zPMT'], event['event.hit.charge'], event['event.hit.time']]), axis=-1),
#                           columns=['x', 'y', 'z', 'pe', 'time'])
