import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Lambda,Reshape,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import Model, Sequential
from tensorflow import math
from keras import backend as K
from keras.regularizers import l2

from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, AveragePooling2D
from keras.models import Model

input_shape = (41, 41, 1)
#dropout_rate = 0.2

def create_base_model_customresnet(input_shape,model_name,dropout_rate,filters_1,freeze=False):
    def build_model(inputs):
        # Assuming filters_1, filters_2, filters_3 are defined
        #filters_1, filters_2, filters_3 = 512, 1024, 2048
        filters_2 = 2*filters_1 
        filters_3 = 4*filters_1

        # Block definitions
        blocks = [
        #{"name": "B0_1", "strides": (2, 2),"filters": filters_2},
        #{"name": "B0_2", "strides": (1, 1), "filters": filters_2},
        #{"name": "B0_3", "strides": (1, 1), "filters": filters_2},
        {"name": "B1_1", "strides": (2, 2),"filters": filters_1},
        #{"name": "B1_2", "strides": (1, 1), "filters": filters_1},
        #{"name": "B1_3", "strides": (1, 1), "filters": filters_1},
        {"name": "B2_1", "strides": (2, 2),"filters": filters_2},
        #{"name": "B2_2", "strides": (1, 1), "filters": filters_2},
        #{"name": "B2_3", "strides": (1, 1), "filters": filters_2},
        {"name": "B3_1", "strides": (2, 2),"filters": filters_3},
        #{"name": "B3_2", "strides": (1, 1), "filters": filters_3},
        {"name": "B3_3", "strides": (1, 1), "filters": filters_3},
        ]

        inputconv_name = model_name + "_InputConvolution"
        inputconv = Conv2D(64, (3, 3),strides = (2,2), padding='same',name=inputconv_name,input_shape=input_shape)(inputs)

        shortcut = inputconv

        for block_info in blocks:
            block_name = block_info["name"]
            block_filters = block_info["filters"]

            # Conv1x1-A
            xa_name = f"{model_name}{block_name}_Conv1x1-A"
            xa = Conv2D(block_filters // 4, (1, 1), strides=block_info["strides"], padding='same', name=xa_name)(shortcut)
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
            if block_info["strides"] == (1, 1):  # Add residual connection only if strides are (1, 1)
                xi = Add()([xh, shortcut])
            else:
                xi = xh
            xj_name = f"{model_name}{block_name}_ReLU-Final"
            xj = Activation('relu', name=xj_name)(xi)

            # Update shortcut for the next iteration
            shortcut = xj

        #avgpool = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=False))(xj)
        avgpool = GlobalAveragePooling2D(data_format="channels_first")(xj)

        dropout = Dropout(dropout_rate)(avgpool)
        flat = Flatten()(dropout)
        dense = Dense(units=1024, activation='relu')(flat) # try filters: 1024
        model = Model(inputs=inputs, outputs=dense,name=model_name)

        if freeze:
            for layer in model.layers:
                layer.trainable = False
                
        return model
    return build_model

#Define the model for the single-view CNNs
def create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size,freeze=False):
    def build_model(inputs):
        Conv1 = Conv2D(filters=25, kernel_size=kernel_size, padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,)(inputs)
        LeakyRelu1 = LeakyReLU(alpha=0.1)(Conv1)
        MaxPool1 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu1)

        #print("Before first Dropout")

        Dropout1 = Dropout(dropout_rate)(MaxPool1)
        Conv2 = Conv2D(filters=25, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout1)
        LeakyRelu2 = LeakyReLU(alpha=0.1)(Conv2) 
        MaxPool2 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu2)

        Dropout2 = Dropout(dropout_rate)(MaxPool2)
        Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
        LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
        MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

        Dropout3 = Dropout(dropout_rate)(MaxPool3)
        Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
        LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
        MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

        Dropout4 = Dropout(dropout_rate)(MaxPool4)
        Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
        LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
        MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

        ######################################################################

        Dropout1 = Dropout(dropout_rate)(MaxPool5)
        Conv_merged1 = Conv2D(filters=25,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout1)
        MaxPool6 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged1)

        Dropout2 = Dropout(dropout_rate)(MaxPool6)
        Conv_merged2 = Conv2D(filters=50,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout2)
        MaxPool7 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged2)

        Dropout3 = Dropout(dropout_rate)(MaxPool7)
        Conv_merged3 = Conv2D(filters=100,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout3)
        MaxPool8 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged3)

        Flat_merged1 = Flatten()(MaxPool8)
        Dropout4 = Dropout(dropout_rate)(Flat_merged1)
        dense_layer1 = Dense(units=100, activation='relu')(Dropout4)

        Dropout5 = Dropout(dropout_rate)(dense_layer1)
        dense_layer2 = Dense(units=50, activation='relu')(Dropout5)

        Dropout6 = Dropout(dropout_rate)(dense_layer2)
        dense_layer3 = Dense(units=1, activation='sigmoid')(Dropout6)

        model = Model(inputs=inputs, outputs=dense_layer3)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model
    return build_model

def create_base_model_multimoda(input_shape):
    rate = 0.2
    pool_size = 2
    kernel_size = 4
    reg = 0.00001
    model = Sequential()

    model.add(Conv2D(filters=25, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(rate))
    model.add(Conv2D(filters=100, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    return model


# Define the model for the combination of the previous CNNs and the final CNN for classification

def create_single_model(model):
    inputs = model.input
    x = model.output
    x = Dropout(0.5)(x)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_single = Model(inputs,outputs)
    #print_layer_dimensions(model_single)
    return model_single

def create_latefc_model(models,inputs):
    fusionlayer = concatenate([model.output for model in models],axis=-1)
    fusionlayer = Dense(units=1024,activation='relu')(fusionlayer)
    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi

def create_latemax_model(models,inputs):
    fusionlayer = Lambda(lambda x: tf.reduce_max(x, axis=0), output_shape=input_shape)([model.output for model in models])
    # fusionlayer = concatenate([model.output for model in models],axis=0)
    # fusionlayer = Maximum(axis=0)(fusionlayer)
    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi


# Define the model for the combination of the previous CNNs and the final CNN for classification

def run_model_multimoda(models,inputs,rate):
    #rate = 0.2 

    merged = concatenate(models)

    Dropout1 = Dropout(rate)(merged)
    Conv_merged1 = Conv2D(filters=25,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout1)
    MaxPool_merged1 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged1)

    Dropout2 = Dropout(rate)(MaxPool_merged1)
    Conv_merged2 = Conv2D(filters=50,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout2)
    MaxPool_merged2 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged2)

    Dropout3 = Dropout(rate)(MaxPool_merged2)
    Conv_merged3 = Conv2D(filters=100,kernel_size=[2,2],activation='relu',padding='same',input_shape=(48,48,1))(Dropout3)
    MaxPool_merged3 = MaxPooling2D(pool_size=2,padding='same')(Conv_merged3)

    Flat_merged1 = Flatten()(MaxPool_merged3)
    Dropout4 = Dropout(rate)(Flat_merged1)
    dense_layer_merged1 = Dense(units=100, activation='relu')(Dropout4)

    Dropout5 = Dropout(rate)(dense_layer_merged1)
    dense_layer_merged2 = Dense(units=50, activation='relu')(Dropout5)

    Dropout6 = Dropout(rate)(dense_layer_merged2)
    dense_layer_merged3 = Dense(units=1, activation='sigmoid')(Dropout6)

    model = Model(inputs=inputs, outputs=dense_layer_merged3)
    return model

def lr_with_warmup(epoch, initial_lr=0.256, warmup_epochs=5, total_epochs=250):
    lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate = initial_lr, decay_steps = total_epochs - warmup_epochs, alpha = 0.0)

    return tf.where(epoch < warmup_epochs, initial_lr, lr_schedule(epoch - warmup_epochs))

def scheduler(epoch,lr):
    if epoch < 25:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def create_multi_model(base, transfer, fusiontype, input_shape, kernel_size, dropout_rate, reg, pool_size, filters_1):
    print("\n #####################   MULTI VIEW MODEL   #######################")
    print("###### ",base, " ##### ",fusiontype," ######")
    if base == 'moda':
        if transfer == 'yes':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_moda(input_1,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_moda(input_2,kernel_size,dropout_rate,reg, pool_size, freeze=True)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_moda(input_3,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_moda(input_4,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)
        else: 
            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_1)
            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_2)
            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_3)
            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_4)   

        if fusiontype == "latefc":
            model_multi = create_latefc_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
        elif fusiontype == "latemax":
            model_multi = create_latemax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
        else: print("ERROR: Fusiontype not known!!")

    elif base == 'resnet':
        if transfer == 'yes':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_customresnet(input_1, 'Model1',dropout_rate,filters_1, freeze=True)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_customresnet(input_2, 'Model2',dropout_rate,filters_1, freeze=True)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_customresnet(input_3, 'Model3',dropout_rate,filters_1,freeze=True)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_customresnet(input_4,'Model4',dropout_rate,filters_1, freeze=True)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)
        else: 
            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_customresnet(input_shape,'Model1',dropout_rate,filters_1)(input_1)
            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_customresnet(input_shape,'Model2',dropout_rate,filters_1)(input_2)
            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_customresnet(input_shape,'Model3',dropout_rate,filters_1)(input_3)
            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_customresnet(input_shape,'Model4',dropout_rate,filters_1)(input_4)

        if fusiontype == "latefc":
            model_multi = create_latefc_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
        elif fusiontype == "latemax":
            model_multi = create_latemax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
        else: print("ERROR: Fusiontype not known!!")

    elif base == 'modamulti':
        input_1 = Input(shape=input_shape)
        cnn_model_1 = create_base_model_multimoda(input_shape)(input_1)
        input_2 = Input(shape=input_shape)
        cnn_model_2 = create_base_model_multimoda(input_shape)(input_2)
        input_3 = Input(shape=input_shape)
        cnn_model_3 = create_base_model_multimoda(input_shape)(input_3)
        input_4 = Input(shape=input_shape)
        cnn_model_4 = create_base_model_multimoda(input_shape)(input_4)

        model_multi = run_model_multimoda([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate)

    return model_multi






######### NOT USED #############################################

def compile_model(model):
    # Label smoothing
    label_smoothing = 0.1
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=label_smoothing)

    # Compile the model with the LearningRateScheduler
    model.compile(
        optimizer=optimizers.SGD(momentum=0.875),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model

def compile_model2(model):
    # Weight decay
    weight_decay = 3.0517578125e-05  # 1/32768
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))

    # Label smoothing
    label_smoothing = 0.1
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    # Compile the model with the LearningRateScheduler
    model.compile(
        optimizer=optimizers.SGD(momentum=0.875),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model
    
def create_multi_model2(base, transfer, fusiontype, input_shape, kernel_size, dropout_rate, reg, pool_size, filters_1):

    def load_weights_and_freeze(model):
        model.load_weights('single_cnn_weights_partial.h5', by_name=True)
        for layer in model.layers:
            layer.trainable = False

    def create_cnn_input(input_shape):
        input_layer = Input(shape=input_shape)
        return input_layer

    def create_cnn_model(input_layer, model_func,transfer, **kwargs):
        cnn_model = model_func(input_layer, **kwargs)(input_layer)

        if transfer == 'yes':
            load_weights_and_freeze(cnn_model)

        return cnn_model

    if base == 'moda':
        model_func = create_base_model_moda
        model_input = lambda i: create_cnn_input(input_shape)
        create_model_func = lambda i: create_cnn_model(model_input, model_func, transfer,input_shape,kernel_size,dropout_rate,reg,pool_size)
        input_layers = [model_input(i) for i in range(1,5)]
        cnn_models  = [create_model_func(i) for i in range(1, 5)]
        #cnn_models = [create_model_func(i)[1] for i in range(1, 5)]
    elif base == 'resnet':
        model_func = create_base_model_customresnet
        create_model_func = lambda i: create_cnn_model(input_shape, model_func, transfer, 'Model{}'.format(i), dropout_rate, filters_1)
        input_layers = [create_model_func(i)[0] for i in range(1, 5)]
        cnn_models = [create_model_func(i)[1] for i in range(1, 5)]
    else:
        model_func = create_base_model_multimoda
        create_model_func = lambda i: create_cnn_model(input_shape, model_func, transfer,dropout_rate)
        input_layers = [create_model_func(i)[0] for i in range(1, 5)]
        cnn_models = [create_model_func(i)[1] for i in range(1, 5)]

    if fusiontype == "latefc":
        model_multi = create_latefc_model(cnn_models, input_layers)
    elif fusiontype == "latemax":
        model_multi = create_latemax_model(cnn_models, input_layers)
    else:
        raise ValueError("ERROR: Fusiontype not known!!")

    return model_multi

def create_early_model(models,inputs,ftype,block):
    
    def inner_block(input, filters,outer_nr,inner_nr, strides=(1, 1), firstblock = False,):
        shortcut = input

        bname = model_name + outer_nr + inner_nr

        if input.shape[-1] != filters:
            inputconvname = bname + "_InpuConv"
            input = Conv2D(filters, (1, 1), strides=strides,kernel_regularizer=l2(reg), padding='same',name=inputconvname)(input)
            input = BatchNormalization()(input)    

        if firstblock == True: 
            xaname = bname + "_Conv1x1-A_firstblock"
            xa = Conv2D(filters // 4, (1, 1), strides=(2,2),kernel_regularizer=l2(reg), padding='same',name=xaname)(input)
            firstblockconvname = bname + "_FirstBlockConv"
            shortcut = Conv2D(filters, (1, 1), strides=(2,2),kernel_regularizer=l2(reg), padding='same',name=firstblockconvname)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        else: 
            xaname = bname + "_Conv1x1-A"
            xa = Conv2D(filters // 4, (1, 1), strides=strides,kernel_regularizer=l2(reg), padding='same',name=xaname)(input)

        xbname = bname + "_BatchNorm-A" 
        xb = BatchNormalization()(xa)
        xcname = bname + "_ReLU-A" 
        xc = Activation('relu',name=xcname)(xb)
        xdname = bname + "_Conv3x3-B" 
        xd = Conv2D(filters // 4, (3, 3), strides=strides,kernel_regularizer=l2(reg), padding='same',name = xdname)(xc)
        xename = bname + "_BatchNorm-B"
        xe = BatchNormalization()(xd)
        xfname = bname + "_ReLU-B"
        xf = Activation('relu',name=xfname)(xe)
        xgname = bname +"_Conv1x1-C"
        xg = Conv2D(filters, (1, 1), strides=strides,kernel_regularizer=l2(reg), padding='same',name=xgname)(xf)
        xhname = bname + "_BatchNorm-C"
        xh = BatchNormalization()(xg)

        xiname = bname + "_ResidualConnection"
        xi = Add()([xh, shortcut])
        xjname = bname + "_ReLU-Final"
        xj = Activation('relu',name=xjname)(xi)
        return xj
    
    def outer_block(input,num_filters,block_nr):
        x1 = inner_block(input,num_filters,block_nr,'_1',firstblock=True)
        x2 = inner_block(x1,num_filters,block_nr,'_2')
        x3 = inner_block(x2,num_filters,block_nr,'_3')
        return x3
    
    ## nn1_block_idx = 35 for Block 0_3 Final Activation Output
    #nn1_block_idx = 36

    layer1name = "Model1" + block + "_3_ReLU-Final"
    layer2name = "Model2" + block + "_3_ReLU-Final"
    layer3name = "Model3" + block + "_3_ReLU-Final"
    layer4name = "Model4" + block + "_3_ReLU-Final"
    nn1_block_names = [layer1name,layer2name,layer3name,layer4name]



    #nn1_block_names = ["Model1B1_3_ReLU-Final","Model2B1_3_ReLU-Final","Model3B1_3_ReLU-Final","Model4B1_3_ReLU-Final"]

    for  model, layer_name in zip(models, nn1_block_names):
        print(model.get_layer(name=layer_name).output.shape)

    #layer_outputs =     [model.layers[nn1_block_idx].output for model in models]

    #nn1_block_names = ["Model1B1_3_ReLU-Final", "Model2B1_3_ReLU-Final", "Model3B1_3_ReLU-Final", "Model4B1_3_ReLU-Final"]
    layer_outputs = [model.get_layer(name=layer_name).output for model, layer_name in zip(models, nn1_block_names)]

    fusionlayer = Concatenate(axis=-1)(layer_outputs)
    print("Fusionlayer after Concatenation:" ,fusionlayer.shape)

    #fusionlayer = Reshape((11, 11, 1024, 4))(fusionlayer)
    #fusionlayer = Reshape((*fusionlayer.shape[1:],4))(fusionlayer)
    fusionlayer = Reshape((fusionlayer.shape[1], fusionlayer.shape[2], fusionlayer.shape[3] // 4, 4))(fusionlayer)

    print("Fusionlayer after Reshape:" ,fusionlayer.shape)

    max_pooling_function = Lambda(lambda x: K.max(x, axis=-1, keepdims=False))
    fusionfilters = fusionlayer.shape[3]
    print("Fusionfilters: ",fusionfilters)
      
    #fusionlayer = Conv3D(filters=fusionfilters, kernel_size=(1, 1, 4*fusionfilters), strides = (1,1,1), activation='relu',padding="same")(fusionlayer)
    
    if ftype == "earlyconv":
        conv2d_list = []
        for i in range(4):  # Assuming you have 4 channels
            slice_layer = Lambda(lambda x: x[:, :, :, i])(fusionlayer)
            conv2d = Conv2D(filters=fusionfilters, kernel_size=(1, 1), padding='same', activation='relu')(slice_layer)
            conv2d_list.append(conv2d)
            #fusionlayer = conv2d_list
        fusionlayer= concatenate(conv2d_list, axis=-1)
    elif ftype == "earlymax":     
        fusionlayer = max_pooling_function(fusionlayer) 
    else: print ("Fusiontype earlymax or earlyconv must be specified")
    # Concatenate the results along the channel axis
    
    
    
    
    print("Fusionlayer after MaxPooling:",fusionlayer.shape)

    model_name = "CNN2"

    if block == "B0":
        b1 = outer_block(fusionlayer,512,'B1')
        b2 = outer_block(b1,1024,'B2')
        b3 = outer_block(b2,2048,'B3')
    elif block == "B1":
        b2 = outer_block(fusionlayer,1024,'B2')
        b3 = outer_block(b2,2048,'B3')
    elif block == "B2":
        b3 = outer_block(fusionlayer,2048,'B3')   
    elif block == "B3":
        b3 = fusionlayer
    else: print('Choose from B0 to B3 please!')

    avgpool = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=False))(b3)
    dense1024 = Dense(units=1024,kernel_regularizer=l2(reg), activation='relu')(avgpool) # softmax - relu?

    x = Dropout(0.5)(dense1024)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)

    return model_multi