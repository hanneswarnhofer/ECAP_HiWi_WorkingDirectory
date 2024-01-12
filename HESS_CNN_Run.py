
from HESS_CNN_ProcessingDataFunctions import *
from HESS_CNN_CreateModelsFunctions import *

print("Functions Defined.")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int,default=256)
parser.add_argument("-r", "--rate", type=float,default=0.1)
parser.add_argument("-reg", "--regulization", type=float,default=0.00001)
parser.add_argument("-t", "--threshold", type=float,default=60)
parser.add_argument("-c", "--cut", type=int,default=2)
parser.add_argument("-ne", "--numevents", type=int,default=100000)
parser.add_argument("-ft","--fusiontype",type=str,default="latefc")
parser.add_argument("-n","--normalize",type=str,default="nonorm")
parser.add_argument("-loc","--location",type=str,default="alex")
parser.add_argument("-transfer","--transfer",type=str,default="no")
parser.add_argument("-base","--base",type=str,default='moda')
parser.add_argument("-lr",'--learningrate',type=float,default=0.2)
parser.add_argument("-plt",'--plot',type=str,default='no')
parser.add_argument("-fil",'--filter',type=int,default=512)
parser.add_argument("-single",'--single',type=str,default='yes')

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
single = args.single

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
print("\n Single View: ", single)
print("\n")

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "CustomResNetBlocks11_21_31_33_vsMoDA" 


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
print("Date-Time: ", formatted_datetime)
name_str, name_single_str = create_strings(fnr,formatted_datetime,batch_size,dropout_rate,reg,num_epochs,fusiontype,transfer,base,normalize,filters_1)

data , labels = dataloader(num_events, location)
mapped_images , mapped_labels = datamapper(data,labels,num_events,cut_nonzero,sum_threshold)
single_train_data, single_train_labels, single_test_data, single_test_labels, train_data, train_labels, test_data, test_labels = data_splitter(mapped_images,mapped_labels,plot,formatted_datetime,location)


patience = 10
input_shape = (41, 41, 1)
pool_size = 2
kernel_size = 2



print("##################################################################")

my_callbacks = [
    keras.callbacks.LearningRateScheduler(scheduler),
    keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=patience,verbose=1,mode='auto'),]

    #keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(f'val_loss: {logs["val_loss"]:.4f} - val_accuracy: {logs["val_accuracy"]:.4f}')),]
    #keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='min', verbose=1),]

if single == 'yes':

    print("\n #####################   SINGLE VIEW MODEL   #######################")
    print("###### ",base, " ##### ",fusiontype," ######")

    inputs = Input(shape=input_shape)

    if base == 'moda':
        base_cnn = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(inputs)
    elif base == 'resnet':
        base_cnn = create_base_model_customresnet(input_shape,'SingleModel',dropout_rate,filters_1)(inputs)
    else: print("Unknown Base Model Specified! Must be 'moda' or 'resnet' .")

    single_view_model = create_single_model(base_cnn)
    #compiled_single_view_model = compile_model(single_view_model)

    single_view_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #single_view_model.compile(keras.optimizers.SGD(), loss='mse')
    #single_view_model.summary()

    save_model(single_view_model,name_single_str,location)
    single_history = single_view_model.fit(single_train_data, single_train_labels, epochs=num_epochs, batch_size=batch_size,validation_data=(single_test_data,single_test_labels),callbacks=my_callbacks)

    #base_cnn_weights = single_view_model.get_layer('lambda').get_weights()
    base_str_single = base + "_singleview"
    create_history_plot(single_history,name_single_str,base=base_str_single)



    
model_multi = create_multi_model(base, transfer, fusiontype, input_shape, kernel_size, dropout_rate, reg, pool_size, filters_1)
save_model(model_multi,name_str,location)
#model_multi.summary()

model_multi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=my_callbacks)
#history = compiled_multi_view_model.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels))
print("... Finished the Fitting")

# Save the history files for later usage in other scripts and plot results
base_str_multi = base + "_multiview_" + fusiontype
create_history_plot(history,name_str,base_str_multi)


#### TO DO:
## Some mistake in the data preprocessing?? Compare to other Seq python files!!
## Load the weights also for NN2, but keep it trainable!
## Make some more Example Images! Overview with a lot of images on pdf pages!
## Clean up the data loading process!
## Clean up the other functions
## Run with different complexity levels of CustomResNet
