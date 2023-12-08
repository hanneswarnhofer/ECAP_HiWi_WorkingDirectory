from tensorflow.keras import layers as lay


def get_shapes(inp_dict):
    return {k: val.shape[1:] for k, val in inp_dict.items()}


def get_init(activation):
    if activation == "selu":
        return 'lecun_normal'
    elif activation == "relu" or activation == "elu":
        return 'he_uniform'
    else:
        raise KeyError("No initialization was found for activation", activation)


def get_conv2d(filters, kernel_size, strides=(1, 1), padding="same", activation=None, **kwargs):
    defaults = {}

    if "kernel_initializer" not in kwargs.keys() and activation is not None:
        defaults["kernel_initializer"] = get_init(activation)

    defaults = {**defaults, **kwargs}  # kwargs overwrite defaults
    return lay.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, **defaults)


def squeeze_excite_module(tensor, n_filters, ratio=16):
    # not using bias in ResNet architecture suggested by the SE authors
    init = tensor
    se_shape = (1, 1, n_filters)

    se = lay.GlobalAveragePooling2D()(init)
    se = lay.Reshape(se_shape)(se)
    se = lay.Dense(n_filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = lay.Dense(n_filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = lay.multiply([init, se])
    return x


def get_resnext_module(x, n_filters, cardinality=32, bn=False, width=4, subsample=False, se_block=True, se_ratio=16):
    # default of paper = 256 filters (doubled after subsampling), c=32, width=4
    branch_fil = cardinality * width  # number of branch filters (see Fig 2 c of ResNeXt publication)

    if subsample is True:
        y = get_conv2d(branch_fil, 1, strides=2)(x)
        x = get_conv2d(n_filters, 1, strides=2)(x)  # projection shortcut
    else:
        y = get_conv2d(branch_fil, 1)(x)

    if bn is True:
        y = lay.BatchNormalization()(y)

    y = lay.Activation("relu")(y)
    y = get_conv2d(branch_fil, 3, groups=cardinality)(y)

    if bn:
        y = lay.BatchNormalization()(y)

    y = lay.Activation("relu")(y)
    y = get_conv2d(n_filters, 1)(y)

    if bn is True:
        y = lay.BatchNormalization()(y)

    if se_block is True:
        y = squeeze_excite_module(y, n_filters, se_ratio)

    out = lay.Add()([x, y])
    return lay.Activation("relu")(out)


def get_resnext_se_block(x, n_filters, cardinality, bn=False, width=4, subsample=False, ratio=16):
    x = get_resnext_module(x, n_filters, cardinality, bn, width, subsample=subsample, se_ratio=ratio)
    x = get_resnext_module(x, n_filters, cardinality, bn, width, subsample=False, se_ratio=ratio)
    x = get_resnext_module(x, n_filters, cardinality, bn, width, subsample=False, se_ratio=ratio)
    return x
