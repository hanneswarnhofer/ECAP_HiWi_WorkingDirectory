from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers as l
from models.tf import tflayers, base, cnn_utils as cnnu


def get_ct_14_tower_resnext(shape, n_filters=64, cardinality=32, bn=False, name=""):
    inp = l.Input(shape=shape)
    x = cnnu.get_conv2d(n_filters, 3)(inp)

    if bn:
        x = l.BatchNormalization()(x)

    x = l.Activation("relu")(x)
    x = cnnu.get_resnext_se_block(x, n_filters, cardinality=cardinality, bn=bn, subsample=False)

    x = cnnu.get_resnext_se_block(x, 2 * n_filters, cardinality=cardinality, bn=bn, subsample=True)

    x = cnnu.get_resnext_se_block(x, 4 * n_filters, cardinality=cardinality, bn=bn, subsample=True)

    x = cnnu.get_resnext_se_block(x, 8 * n_filters, cardinality=cardinality, bn=bn, subsample=True)

    out = l.GlobalAvgPool2D()(x)
    return base.BaseModel(inp, out, name=name)


def get_ct_14_tower_locally(shape, name=""):
    inp = l.Input(shape=shape)
    x = l.LocallyConnected2D(16, (3, 3), padding="valid", activation="elu")(inp)
    x = l.LocallyConnected2D(16, (3, 3), padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(32, (3, 3), padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(32, (3, 3), strides=2, padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(64, (3, 3), padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(64, (3, 3), padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(128, (3, 3), strides=2, padding="valid", activation="elu")(x)
    x = l.LocallyConnected2D(128, (3, 3), padding="valid", activation="elu")(x)
    out = l.GlobalAvgPool2D()(x)
    return base.BaseModel(inp, out, name=name)


def get_model(img_dict, tasks, stats, bn=False, share_ct14=True):
    shapes = cnnu.get_shapes(img_dict)
    ct_14_inputs = [l.Input(shape=val, name=k) for k, val in shapes.items() if k != "ct5"]

    if share_ct14 is True:
        ct14_tower = get_ct_14_tower_resnext(shapes["ct1"], bn=bn, name="tower_ct14")
        ct14_tower_out = [ct14_tower(z) for z in ct_14_inputs]
    else:
        ct14_models = [get_ct_14_tower_resnext(val, name="%s_tower" % k, bn=bn) for k, val in shapes.items() if k != "ct5"]
        ct14_tower_out = [m(inp) for inp, m in zip(ct_14_inputs, ct14_models)]

    ct_5_input = l.Input(shape=shapes["ct5"], name="ct5")
    ct5_tower = get_ct_14_tower_resnext(shapes["ct5"], name="ct5_tower", bn=bn)
    ct5_tower_out = ct5_tower(ct_5_input)

    # ## Approach 1: input: (None, 128, 5)
    conc = K.stack(ct14_tower_out + [ct5_tower_out], axis=-1)
    #
    x = l.Conv1D(16, 1, activation="elu", padding='same')(conc)
    x = l.Conv1D(32, 1, activation="elu", padding='same')(x)
    x = l.Conv1D(64, 1, activation="elu", padding='same')(x)
    x = l.Conv1D(1, 1, activation="elu", padding='same')(x)

    # ## Approach 2: input: (None, 5, 128)
    # conc = K.stack(ct14_tower_out + [ct5_tower_out], axis=1)
    # x = l.Conv1D(64, 1, activation="elu", padding='same')(conc)
    # x = l.Conv1D(64, 1, activation="elu", padding='same')(x)
    # x = l.Conv1D(64, 1, activation="elu", padding='same')(x)

    x = l.Flatten()(x)
    # x = l.Dropout(0.5)(x)
    # x = l.Dense(128, activation="elu")(x)
    # x = l.Dropout(0.5)(x)

    out = []

    if "primary" in tasks:
        # y = l.Dense(128, activation="elu")(x)
        y = l.Dropout(0.6)(x)
        out.append(l.Dense(2, activation="softmax", kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001), name="primary")(x))

    if "energy" in tasks:
        # y = l.Dense(128, activation="elu")(x)
        # y = l.Dropout(0.3)(y)
        energy = l.Dense(1, kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001))(x)
        out.append(tflayers.NormToLabel(stats['energy'], name='energy')(energy))

    if "axis" in tasks:
        # y = l.Dense(128, activation="elu")(x)
        # y = l.Dropout(0.3)(y)
        y = l.Dense(3, kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001))(x)
        out.append(tflayers.EuclideanNorm(name="axis")(y))

    if "impact" in tasks:
        # y = l.Dense(128, activation="elu")(x)
        # y = l.Dropout(0.3)(y)
        impact = l.Dense(2)(x)
        out.append(tflayers.NormToLabel(stats['impact'], name='impact')(impact))

    return base.BaseModel(ct_14_inputs + [ct_5_input], out, name="CNN_model")


def dummy_model(inp, tasks):
    shapes = cnnu.get_shapes(inp)
    ct_14_inputs = [l.Input(shape=val, name=k) for k, val in shapes.items() if k != "ct5"]
    ct_5_input = l.Input(shape=shapes["ct5"], name="ct5")

    z = [l.Flatten()(x_) for x_ in ct_14_inputs]
    ct_5_out = l.Flatten()(ct_5_input)
    conc_ = l.Concatenate()(z + [ct_5_out])
    x = l.Dense(128, activation="elu")(conc_)
    x = l.Dropout(0.5)(x)
    x = l.Dense(128, activation="elu")(x)
    x = l.Dropout(0.5)(x)

    out = []

    if "primary" in tasks:
        y = l.Dense(128, activation="elu")(x)
        y = l.Dropout(0.5)(y)
        out.append(l.Dense(1, activation="sigmoid", name="primary")(y))

    if "energy" in tasks:
        y = l.Dense(128, activation="elu")(x)
        y = l.Dropout(0.5)(y)
        out.append(l.Dense(1, name="energy")(y))

    if "axis" in tasks:
        y = l.Dense(128, activation="elu")(x)
        y = l.Dropout(0.5)(y)
        y = l.Dense(3)(y)
        out.append(tflayers.EuclideanNorm(name="axis")(y))

    if "impact" in tasks:
        y = l.Dense(128, activation="elu")(x)
        y = l.Dropout(0.5)(y)
        out.append(l.Dense(2, name="impact")(y))

    return base.BaseModel(ct_14_inputs + [ct_5_input], out)
