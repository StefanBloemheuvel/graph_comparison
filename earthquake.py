#%%

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Which dataset to choose', choices=['ci','cw'])
parser.add_argument('algorithm', help='Which algorithm to use', choices=['gabriel','kmeans','optics','knn_weighted','knn_unweighted','minmax','relative_neighborhood','gaussian','dtw','correlation','mic'])
args = parser.parse_args()
# args = parser.parse_args(args=[])
print(args.dataset)
print(args.algorithm)


print()

import time
import datetime

def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S") 

print(print_time())
import sys
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('we have a GPU available')
else:
    print('no GPU available, so exit')
    sys.exit(0)
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('dataset', help='Which network_choice to use', choices=['ci','cw'], nargs='?', default='ci')
# parser.add_argument('algorithm', help='Which algorithm to use', choices=['gabriel','kmeans_own','optics','knn_weighted','knn_unweighted','minmax','dbscan','new_gaussian','gaussian','relative_neighborhood','correlation_top'], nargs='?', default='gabriel')
# parser.add_argument('random_state_here', help='Which random state to choose', choices=[1,2,3,4,5], nargs='?',  default=1,type=int)
# parser.add_argument('k', help='k', type=int)

# args = parser.parse_args()
# print('network = ',args.dataset)
# print('algorithm = ',args.algorithm)
# print('random state = ',args.random_state_here)
# print('k  = ',args.k)

import numpy as np
import sys
max_iteration = 16
if args.algorithm == 'gabriel':
    options_list = [0]
if args.algorithm == 'relative_neighborhood':
    options_list = [0]
if args.algorithm == 'gaussian':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
if args.algorithm == 'kmeans':
    options_list = [i for i in range(2,40)]
if args.algorithm == 'optics':
    options_list = [i for i in range(2,40)]
if args.algorithm == 'knn_weighted':
    options_list = [i for i in range(2,40)]
if args.algorithm == 'knn_unweighted':
    options_list = [i for i in range(2,40)]
if args.algorithm == 'minmax':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
if args.algorithm == 'dtw' or args.algorithm == 'correlation' or args.algorithm == 'mic':
    options_list = [round(i,2) for i in np.arange(0.05,0.95,0.05)]
print(f'went for {args.dataset} and {args.algorithm} \n')
print(f' options are = {options_list}')

print('#' * 40)
print('#' * 40)
print('#' * 40)

# print()
import itertools
import yaml
from tensorflow.python.eager import context
import random
from tensorflow.python.framework import ops


import spektral
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random

import math
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

from spektral.layers import GCNConv
from tensorflow.keras.layers import *
from spektral.utils import gcn_filter
from sklearn.model_selection import train_test_split
import datetime

from tensorflow.python.client import device_lib

import sys
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print(f'no gpu available')
    sys.exit(0)
seed = 1
def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")  

def normalize(inputs): # Houden
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized)      

def targets_to_list(targets): # Houden
    targets = targets.transpose(2,0,1)

    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
        
    return targetList


seed = 1
import tensorflow
def k_fold_split(inputs, targets): # houden

    # make sure everything is seeded
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    tensorflow.random.set_seed(seed)
    
    p = np.random.permutation(len(targets))
    
    print('min of p = ',np.array(p)[50:100].min())
    print('max of p = ',np.array(p)[50:100].max())
    print('mean of p = ',np.array(p)[50:100].mean())
    inputs = inputs[p]
    targets = targets[p]

    
    ind = int(len(inputs)/5)
    inputsK = []
    targetsK = []

    for i in range(0,5-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])

    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
  
    
    return inputsK, targetsK
        
def merge_splits(inputs, targets, k): # houden
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]

    for i in range(z+1, 5):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
    
    return inputsTrain, targetsTrain, inputs[k], targets[k]

def build_model(input_shape): # houden

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    graph_input = layers.Input(shape=(39,39), name='graph_input')
    graph_features = layers.Input(shape=(39,2), name='graph_features')

    conv1 = layers.Conv1D(filters=32, kernel_size=125, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv1')(wav_input)
    conv1 = layers.Conv1D(filters=64, kernel_size=125, strides=2,  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const), name='conv2')(conv1)

    conv1_new = tf.keras.layers.Reshape((39,conv1.shape[2] * conv1.shape[3]))(conv1)    
    conv1_new = layers.concatenate(inputs=[conv1_new, graph_features], axis=2)

    conv1_new = GCNConv(64, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])
    conv1_new = GCNConv(64, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))([conv1_new, graph_input])

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(0.4, seed=seed)(conv1_new)

    merged = layers.Dense(128, dtype='float32')(conv1_new)

    pga = layers.Dense(39, dtype='float32')(merged)
    pgv = layers.Dense(39, dtype='float32')(merged)
    sa03 = layers.Dense(39, dtype='float32')(merged)
    sa10 = layers.Dense(39, dtype='float32')(merged)
    sa30 = layers.Dense(39, dtype='float32')(merged)
    
    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=[pga, pgv, sa03, sa10, sa30]) #, pgv, sa03, sa10, sa30
    rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.)
    final_model.compile(optimizer=rmsprop, loss='mse')
    
    return final_model

from tensorflow import keras

es = keras.callbacks.EarlyStopping(patience=100, verbose=0, min_delta=0.001, monitor='val_loss', mode='min',baseline=None, restore_best_weights=True)

import sys

#%%
from geoconnector.newest_graph_maker import graph_generator

        
test_set_size = 0.2
for i in options_list:
    print()
    print()
    print(f'current combination = {i}')
    if args.dataset == 'ci':
        inputs = np.float32(np.load('data/inputs_ci.npy', allow_pickle = True))
        targets = np.load('data/targets.npy', allow_pickle = True)
        graph_features = np.load('data/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
    
    if args.dataset == 'cw':
        inputs = np.float32(np.load('data/othernetwork/inputs_cw.npy', allow_pickle = True))
        targets = np.load('data/othernetwork/targets.npy', allow_pickle = True)
        graph_features = np.load('data/othernetwork/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
        
        
    graph_generator_obj = graph_generator()
    if args.algorithm == 'gabriel':
        graph_generator_obj.gabriel(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    if args.algorithm == 'kmeans':
        print('went for kmeans_own')
        graph_generator_obj.kmeans(f'sensor_locations/sensor_locations_{args.dataset}.csv',num_clusters=i)
    if args.algorithm == 'optics':
        graph_generator_obj.optics(f'sensor_locations/sensor_locations_{args.dataset}.csv', min_samples=i)
    if args.algorithm == 'knn_weighted':
        graph_generator_obj.knn_weighted(f'sensor_locations/sensor_locations_{args.dataset}.csv', k=i)
    if args.algorithm == 'knn_unweighted':
        graph_generator_obj.knn_unweighted(f'sensor_locations/sensor_locations_{args.dataset}.csv', k=i)
    if args.algorithm == 'minmax':
        graph_generator_obj.minmax(f'sensor_locations/sensor_locations_{args.dataset}.csv', cutoff=i)
    if args.algorithm == 'dbscan':
        graph_generator_obj.dbscan(f'sensor_locations/sensor_locations_{args.dataset}.csv', eps=i, min_samples=3)
    if args.algorithm == 'gaussian':
        graph_generator_obj.gaussian(f'sensor_locations/sensor_locations_{args.dataset}.csv',i)
    # if args.algorithm == 'gaussian':
        # graph_generator_obj.gaussian(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    if args.algorithm == 'relative_neighborhood':
        graph_generator_obj.relative_neighborhood(f'sensor_locations/sensor_locations_{args.dataset}.csv')
    if args.algorithm == 'kmeans_own':
        graph_generator_obj.kmeans_own(f'sensor_locations/sensor_locations_{args.dataset}.csv', num_clusters=i)
    if args.algorithm == 'correlation_top':
        graph_generator_obj.correlation_top(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy',threshold=i)
    
    if args.algorithm == 'dtw' or args.algorithm == 'correlation' or args.algorithm == 'mic' and args.dataset in ['ci','cw']:
        print(f'went for no clips')
        graph_generator_obj.from_signal(f'sensor_locations/sensor_locations_{args.dataset}.csv', f'sensor_locations/inputs_{args.dataset}.npy', variant=args.algorithm, clips=False,threshold=i)

    graph_generator_obj.create_normalized_laplacian_matrix()
    graph_generator_obj.summary_statistics()
    if graph_generator_obj.number_of_edges == 0:
        print('no edges so no solution')
        print('\n' * 5)
        continue
    
    if graph_generator_obj.data.shape[0] >= graph_generator_obj.number_of_edges:
        print('only self edges')
        print(graph_generator_obj.data.shape[0])
        print(graph_generator_obj.number_of_edges)
        print('\n' * 5)
        continue
    else:
        print(f'number of edges is > {graph_generator_obj.data.shape[0]}, namely {graph_generator_obj.number_of_edges}')
        print('\n' * 5)
        
    adj2 = graph_generator_obj.normalized_laplacian_matrix
    print(f'graph after creation:')
    print(f'{adj2[0:5,0:5]}')
        
    graph_input = np.array([adj2] * inputs.shape[0])

    train_inputs, test_inputs, traingraphinput , testgraphinput, train_graphfeature, test_graphfeature, train_targets, test_targets = train_test_split(inputs,graph_input, graph_features, targets, test_size=test_set_size, random_state=1)
    testInputs = normalize(test_inputs[:, :, :1000, :])        



    inputsK, targetsK = k_fold_split(train_inputs, train_targets)

    del train_inputs
    del inputs

    mse_list = []

    import time

    start = time.time()

    

    for k in range(0,3):
        print(f'current k = {k}')
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        np.random.permutation(seed)
        tensorflow.random.set_seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed)# 3. Set `numpy` pseudo-random generator at a fixed value
        tf.random.set_seed(seed)
        np.random.RandomState(seed)
        np.random.seed(seed)
        context.set_global_seed(seed)
        ops.get_default_graph().seed = seed

        #pip install tensorflow-determinism needed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed)

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(inputsK, targetsK, k)

        train_graphinput = traingraphinput[0:trainInputsAll.shape[0],:,:]
        train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0],:,:]

        val_graphinput = traingraphinput[0:valInputsAll.shape[0],:,:]
        val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0],:,:]

        trainInputs = normalize(trainInputsAll[:, :, :1000, :])
        valInputs = normalize(valInputsAll[:, :, :1000, :])

        model = build_model(valInputs[0].shape)

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{args.dataset}_{k}.h5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True
            ),
        ]

        # print(model.summary())
        
        history = model.fit(x=[trainInputs,train_graphinput,train_graphfeatureinput], 
                            y=targets_to_list(trainTargets),
                epochs=100, batch_size=30,
                validation_data=([valInputs,val_graphinput,val_graphfeatureinput], targets_to_list(valTargets)),verbose=0,callbacks=[my_callbacks])#
        epochs_runned =  len(history.history['loss'])
        
        import spektral
        model = keras.models.load_model(f'models/{args.dataset}_{k}.h5',custom_objects={'GCNConv':spektral.layers.GCNConv})

        print()
        print('total number of epochs ran = ',len(history.history['loss']))
        print('Fold number:' + str(k))
        predictions = model.predict([testInputs,testgraphinput, test_graphfeature])

        new_predictions = np.array(predictions)
        new_predictions = np.swapaxes(new_predictions,0,2)
        new_predictions = np.swapaxes(new_predictions,0,1)
        
        print(new_predictions.shape)
        
        # Function to calculate mean absolute error
        def MAE(y_true, y_pred):
            return np.mean(abs(y_true - y_pred))
        
        def MSE(Y, YH):
            return np.square(Y - YH).mean()
        mse = MSE(new_predictions,test_targets)
        print(f'current mse was {mse:.4f}')
        mse_list.append(mse)

        keras.backend.clear_session()
        tf.keras.backend.clear_session()

    end = time.time()

    print(f'Took {int(end - start)} seconds')
    print('-')
    print('-')
    print('-')
    print('-')
    print('all averages = ')
    print(f'mse score = ,{np.array(mse_list).mean():.3f}')

    with open(f"new_earthquake.csv", 'a') as f:
        if f.tell() == 0:
            print('a new file or the file was empty')
            f.write(f'time,dataset,algorithm,k,mse,number_of_edges\n')
        else:
            print('file existed, appending')


    # with open(f"individual_results/{args.algorithm}_{args.dataset}.csv", "a") as text_file:
    #     print(f'{print_time()},{sys.argv[0]},PGV,{args.dataset},{args.algorithm},{np.array(mse_list).mean():.3f},{args.random_state_here}', file=text_file)

    with open(f"final_results.csv", "a") as text_file:
        print(f'{print_time()},{args.dataset},{args.algorithm},{i},{MSE(new_predictions, test_targets):.4f},{MAE(new_predictions, test_targets):.4f},{graph_generator_obj.number_of_edges}', file=text_file)

