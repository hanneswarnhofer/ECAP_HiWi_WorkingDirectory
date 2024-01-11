

from HESS_CNN_ProcessingDataFunctions import *
from HESS_CNN_CreateModelsFunctions import *

print("Functions Defined.")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int,default=32)
parser.add_argument("-r", "--rate", type=float,default=0.2)
parser.add_argument("-reg", "--regulization", type=float,default=0.00001)
parser.add_argument("-t", "--threshold", type=float,default=60)
parser.add_argument("-c", "--cut", type=int,default=2)
parser.add_argument("-ne", "--numevents", type=int,default=100000)
parser.add_argument("-ft","--fusiontype",type=str,default="latefc")
parser.add_argument("-n","--normalize",type=str,default="nonorm")
parser.add_argument("-loc","--location",type=str,default="alex")
parser.add_argument("-transfer","--transfer",type=str,default="no")
parser.add_argument("-base","--base",type=str,default='moda')
parser.add_argument("-lr",'--learningrate',type=float,default=0.001)
parser.add_argument("-plt",'--plot',type=str,default='no')
parser.add_argument("-fil",'--filter',type=int,default=512)

args = parser.parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
dropout_rate = args.rate
reg = args.regulization
sum_threshold = args.threshold
cut_nonzero = args.cut
num_events = args.numevents
fusiontype = args.fusiontype
normalize = args.normalize
location = args.location
transfer = args.transfer
base = args.base
learning_rate = args.learningrate
plot = args.plot
filters_1 = args.filter

print("############################################################################")
print("\n #####################    FUSIONTYPE: ",fusiontype,"   #######################")
print("\n")
print("\n Epochs: ", num_epochs)
print("\n Batch Size: ", batch_size)
print("\n Regularization: ", reg)
print("\n Events: ", num_events)
print("\n Learning Rate: ", learning_rate)
print("\n Filters 1: ",filters_1)
print("\n Transfer: ", transfer)
print("\n Threshold: ", sum_threshold)
print("\n Nonzero Cut: ", cut_nonzero)
print("\n Plotting Events: ", plot)
print("\n Base CNN: ", base)
print("\n")

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "CustomResNet2-Block0only" 

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
print("Date-Time: ", formatted_datetime)
name_str, name_single_str = create_strings(fnr,formatted_datetime,batch_size,dropout_rate,reg,num_epochs,fusiontype,transfer,base,normalize,filters_1)



data , labels = dataloader(num_events, location)
mapped_images , mapped_labels = datamapper(data,labels,num_events,cut_nonzero,sum_threshold)
single_train_data, single_train_labels, single_test_data, single_test_labels, train_data, train_labels, test_data, test_labels = data_splitter(mapped_images,mapped_labels,plot,formatted_datetime,location)

patience = 8
input_shape = (41, 41, 1)
pool_size = 2
kernel_size = 2

filters_2 = 2*filters_1 
filters_3 = 4*filters_1

print("##################################################################")
#learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate) #, clipnorm=1.)
from keras.losses import BinaryCrossentropy
loss_fn = BinaryCrossentropy(from_logits=True)
early_stopping_callback_1=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=patience,verbose=1,mode='auto')

print("\n #####################   SINGLE VIEW MODEL   #######################")
inputs = Input(shape=input_shape)

if base == 'moda':
    base_cnn = create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size)(inputs)
elif base == 'resnet':
    base_cnn = create_base_model_customresnet2(input_shape,'SingleModel',dropout_rate,filters_1,filters_2,filters_3)(inputs)
else: print("Unknown Base Model Specified! Must be 'moda' or 'resnet' .")

single_view_model = create_single_model(base_cnn)
single_view_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#single_view_model.summary()

save_model(single_view_model,name_single_str,location)

single_history = single_view_model.fit(single_train_data, single_train_labels, epochs=num_epochs, batch_size=batch_size,validation_data=(single_test_data,single_test_labels),callbacks=[early_stopping_callback_1])

#base_cnn_weights = single_view_model.get_layer('lambda').get_weights()
create_history_plot(single_history,name_single_str)


print("\n #####################   MULTI VIEW MODEL   #######################")
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
            cnn_model_1 = create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_1)
            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_2)
            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_3)
            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_4)   
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
        cnn_model_1 = create_base_model_customresnet2(input_shape,'Model1',dropout_rate,filters_1,filters_2,filters_3)(input_1)
        input_2 = Input(shape=input_shape)
        cnn_model_2 = create_base_model_customresnet2(input_shape,'Model2',dropout_rate,filters_1,filters_2,filters_3)(input_2)
        input_3 = Input(shape=input_shape)
        cnn_model_3 = create_base_model_customresnet2(input_shape,'Model3',dropout_rate,filters_1,filters_2,filters_3)(input_3)
        input_4 = Input(shape=input_shape)
        cnn_model_4 = create_base_model_customresnet2(input_shape,'Model4',dropout_rate,filters_1,filters_2,filters_3)(input_4)

from keras.losses import BinaryCrossentropy

# Create the loss function with from_logits=True
loss_fn = BinaryCrossentropy(from_logits=True)
early_stopping_callback_1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience) #,verbose=1,mode='min')

if fusiontype == "latefc":
    model_multi = create_latefc_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
elif fusiontype == "latemax":
    model_multi = create_latemax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])

else: print("ERROR: Fusiontype not known!!")

#model_multi.summary()

save_model(model_multi,name_str,location)

model_multi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=[early_stopping_callback_1])

print("... Finished the Fitting")

# Save the history files for later usage in other scripts
create_history_plot(history,name_str)


#### TO DO:
## Some mistake in the data preprocessing?? Compare to other Seq python files!!
## Load the weights also for NN2, but keep it trainable!
## Make some more Example Images! Overview with a lot of images on pdf pages!
## Clean up the data loading process!
## Clean up the other functions
## Run with different complexity levels of CustomResNet
