import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Lambda,Reshape,Embedding, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l2

from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, AveragePooling2D
from keras.models import Model

input_shape = (41, 41, 1)
#dropout_rate = 0.2

def create_base_model_customresnet2(input_shape,model_name,dropout_rate,filters_1,filters_2,filters_3,freeze=False):
    def build_model(inputs):
        # Assuming filters_1, filters_2, filters_3 are defined
        #filters_1, filters_2, filters_3 = 512, 1024, 2048

        # Block definitions
        blocks = [
        {"name": "B0_1", "strides": (2, 2),"filters": filters_2},
        {"name": "B0_2", "strides": (1, 1), "filters": filters_2},
        {"name": "B0_3", "strides": (1, 1), "filters": filters_2},
        #{"name": "B1_1", "strides": (2, 2),"filters": filters_1},
        #{"name": "B1_2", "strides": (1, 1), "filters": filters_1},
        #{"name": "B1_3", "strides": (1, 1), "filters": filters_1},
        #{"name": "B2_1", "strides": (2, 2),"filters": filters_2},
        #{"name": "B2_2", "strides": (1, 1), "filters": filters_2},
        #{"name": "B2_3", "strides": (1, 1), "filters": filters_2},
        #{"name": "B3_1", "strides": (2, 2),"filters": filters_3},
        #{"name": "B3_2", "strides": (1, 1), "filters": filters_3},
        #{"name": "B3_3", "strides": (1, 1), "filters": filters_3},
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



def create_base_model_customresnet(input_shape, model_name, dropout_rate ,filters_1,filters_2,filters_3,start_layer=-1,freeze=False, dynamic_input_shape=False ):

    def inner_block(input, filters,outer_nr,inner_nr, strides=(1, 1), firstblock = False,):
        shortcut = input

        bname = model_name + outer_nr + inner_nr

        if input.shape[-1] != filters:
            input = Conv2D(filters, (1, 1), strides=strides, padding='same')(input)
            input = BatchNormalization()(input)    

        if firstblock == True: 
            xaname = bname + "_Conv1x1-A_firstblock"
            xa = Conv2D(filters // 8, (1, 1), strides=(2,2), padding='same',name=xaname)(input)
            shortcut = Conv2D(filters, (1, 1), strides=(2,2), padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        else: 
            xaname = bname + "_Conv1x1-A"
            xa = Conv2D(filters // 4, (1, 1), strides=strides, padding='same',name=xaname)(input)

        xbname = bname + "_BatchNorm-A" 
        xb = BatchNormalization()(xa)
        xcname = bname + "_ReLU-A" 
        xc = Activation('relu',name=xcname)(xb)
    
        dropout1 = Dropout(dropout_rate)(xc)

        xdname = bname + "_Conv3x3-B"
        xd = Conv2D(filters // 4, (3, 3), strides=strides, padding='same',name = xdname)(dropout1)
        xename = bname + "_BatchNorm-B"
        xe = BatchNormalization()(xd)
        xfname = bname + "_ReLU-B"
        xf = Activation('relu',name=xfname)(xe)

        dropout2 = Dropout(dropout_rate)(xf)

        xgname = bname +"_Conv1x1-C"
        xg = Conv2D(filters, (1, 1), strides=strides, padding='same',name=xgname)(dropout2)
        xhname = bname + "_BatchNorm-C"
        xh = BatchNormalization()(xg)

        xiname = bname + "_ResidualConnection"
        xi = Add()([xh, shortcut])
        xjname = bname + "_ReLU-Final"
        xj = Activation('relu',name=xjname)(xi)
        return xj
    
    def outer_block(input,num_filters,block_nr):
        x1 = inner_block(input,num_filters,block_nr,'_1',firstblock=True)
        x2 = inner_block(x1,num_filters,block_nr,'_2')  #CHANGED !!
        x3 = inner_block(x2,num_filters,block_nr,'_3')  #CHANGED !!
        return x3                                 #    CHANGED !!

    if dynamic_input_shape:
        input_shape = inputs.shape.as_list()[1:]  # Get shape excluding batch size
        input_shape = tuple(input_shape)  # Convert to tuple
        inputs = Input(shape=input_shape)
        
    inputconv_name = model_name + "_InputConvolution"
    inputconv = Conv2D(64, (7, 7),strides = (2,2), padding='same',name=inputconv_name)(inputs)
    #maxpool = MaxPooling2D(strides=(2,2), padding='same')(inputconv)
    
    x = outer_block(inputconv, filters_2,'B0')
        
    x = outer_block(x,filters_1,'B1')
    x = outer_block(x,filters_2,'B2')
    x = outer_block(x,filters_3,'B3')  #CHANGED !!
    #x = Flatten()(x)


    avgpool = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=False))(x)
    avgpoolname = model_name + "_GlobalAvgPool"
    #avgpool = GlobalAveragePooling2D(kernel_size=(2,2),name=avgpoolname)(x)

    dropout = Dropout(dropout_rate)(avgpool)

    densename = model_name + "_Dense1024softmax"
    dense = Dense(units=filters_3, activation='relu')(dropout) # try filters: 1024
    model = Model(inputs=inputs, outputs=dense,name=model_name)

    if start_layer != -1:
        for _ in range(0,start_layer):
            model.layers.pop(0)

    if freeze:
        for layer in model.layers:
            layer.trainable = False

    #print_layer_dimensions(model)
            
    return model

#Define the model for the single-view CNNs
def create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size,freeze=False):
    def build_model(inputs):
        Conv1 = Conv2D(filters=25, kernel_size=kernel_size, padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,)(inputs)
        LeakyRelu1 = LeakyReLU(alpha=0.1)(Conv1)
        MaxPool1 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu1)

        #print("Before first Dropout")

        Dropout1 = Dropout(dropout_rate)(MaxPool1)
        Conv2 = Conv2D(filters=30, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout1)
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

        Dropout5 = Dropout(dropout_rate)(MaxPool5)
        Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
        MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

        Dropout6 = Dropout(dropout_rate)(MaxPool6)
        Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
        MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

        Flat = Flatten()(MaxPool7)
        Dense1 = Dense(units=1024, activation='relu')(Flat)

        model = Model(inputs=inputs, outputs=Dense1)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model
    return build_model

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

def create_multi_model(kernel_size,dropout_rate,reg,pool_size,ftype,base,transfer):
    if base == 'moda':
        if transfer == 'yes':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model(input_1,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model(input_2,kernel_size,dropout_rate,reg, pool_size, freeze=True)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model(input_3,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model(input_4,kernel_size,dropout_rate,reg,pool_size, freeze=True)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)
        else: 
            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model(input_1,kernel_size,dropout_rate,reg,pool_size)
            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model(input_2,kernel_size,dropout_rate,reg,pool_size)
            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model(input_3,kernel_size,dropout_rate,reg,pool_size)
            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model(input_4,kernel_size,dropout_rate,reg,pool_size)   
    elif base == 'resnet':
        if transfer == 'yes':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_customresnet2(input_1, 'Model1',dropout_rate,filters_1,filters_2,filters_3, freeze=True)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_customresnet2(input_2, 'Model2',dropout_rate,filters_1,filters_2,filters_3, freeze=True)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_customresnet2(input_3, 'Model3',dropout_rate,filters_1,filters_2,filters_3,freeze=True)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_customresnet2(input_4,'Model4',dropout_rate,filters_1,filters_2,filters_3, freeze=True)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)
        else: 
            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_customresnet2(input_1,'Model1',dropout_rate,filters_1,filters_2,filters_3)
            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_customresnet2(input_2,'Model2',dropout_rate,filters_1,filters_2,filters_3)
            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_customresnet2(input_3,'Model3',dropout_rate,filters_1,filters_2,filters_3)
            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_customresnet2(input_4,'Model4',dropout_rate,filters_1,filters_2,filters_3)     