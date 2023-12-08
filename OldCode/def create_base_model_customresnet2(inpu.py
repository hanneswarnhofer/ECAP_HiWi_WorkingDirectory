def create_base_model_customresnet2(inputs,model_name,freeze=False):

    # Assuming filters_1, filters_2, filters_3 are defined
    filters_1, filters_2, filters_3 = 24, 52, 100

    # Block definitions
    blocks = [
    {"name": "B0_1", "strides": (2, 2), "first_block": True, "filters": filters_2},
    {"name": "B0_2", "strides": (1, 1), "first_block": False, "filters": filters_2},
    {"name": "B0_3", "strides": (1, 1), "first_block": False, "filters": filters_2},
    {"name": "B1_1", "strides": (2, 2), "first_block": True, "filters": filters_1},
    {"name": "B1_2", "strides": (1, 1), "first_block": False, "filters": filters_1},
    {"name": "B1_3", "strides": (1, 1), "first_block": False, "filters": filters_1},
    {"name": "B2_1", "strides": (2, 2), "first_block": True, "filters": filters_2},
    {"name": "B2_2", "strides": (1, 1), "first_block": False, "filters": filters_2},
    {"name": "B2_3", "strides": (1, 1), "first_block": False, "filters": filters_2},
    {"name": "B3_1", "strides": (2, 2), "first_block": True, "filters": filters_3},
    {"name": "B3_2", "strides": (1, 1), "first_block": False, "filters": filters_3},
    {"name": "B3_3", "strides": (1, 1), "first_block": False, "filters": filters_3},
    ]

    inputconv_name = model_name + "_InputConvolution"
    inputconv = Conv2D(64, (7, 7),strides = (2,2), padding='same',name=inputconv_name)(inputs)

    shortcut = inputconv

    for block_info in blocks:
        block_name = block_info["name"]
        block_filters = block_info["filters"]

        # Conv1x1-A
        xa_name = f"{model_name}{block_name}_Conv1x1-A_firstblock" if block_info["first_block"] else f"{model_name}{block_name}_Conv1x1-A"
        xa = Conv2D(block_filters // 8 if block_info["first_block"] else block_filters // 4, (1, 1), strides=block_info["strides"], padding='same', name=xa_name)(shortcut)
        xb = BatchNormalization()(xa)
        xc_name = f"{model_name}{block_name}_ReLU-A"
        xc = Activation('relu', name=xc_name)(xb)

        dropout1 = Dropout(dropout_rate)(xc)

        # Conv3x3-B
        xd_name = f"{model_name}{block_name}_Conv3x3-B"
        xd = Conv2D(block_filters // 4, (3, 3), strides=block_info["strides"], padding='same', name=xd_name)(dropout1)
        xe = BatchNormalization()(xd)
        xf_name = f"{model_name}{block_name}_ReLU-B"
        xf = Activation('relu', name=xf_name)(xe)

        dropout2 = Dropout(dropout_rate)(xf)

        # Conv1x1-C
        xg_name = f"{model_name}{block_name}_Conv1x1-C"
        xg = Conv2D(block_filters, (1, 1), strides=block_info["strides"], padding='same', name=xg_name)(dropout2)
        xh = BatchNormalization()(xg)

        # Residual Connection
        xi = Add()([xh, shortcut])
        xj_name = f"{model_name}{block_name}_ReLU-Final"
        xj = Activation('relu', name=xj_name)(xi)

        # Update shortcut for the next iteration
        shortcut = xj

    avgpool = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=False))(xj)
    #avgpool = GlobalAveragePooling2D(kernel_size=(2,2),name=avgpoolname)(x)

    dropout = Dropout(dropout_rate)(avgpool)
    dense = Dense(units=filters_3, activation='relu')(dropout) # try filters: 1024
    model = Model(inputs=inputs, outputs=dense,name=model_name)

    if freeze:
        for layer in model.layers:
            layer.trainable = False
            
    return model