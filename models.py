# holds different models that can be pulled into the learner.
# each method takes data as input and will output cleaned data and model
import tensorflow as tf
import numpy as np
import pandas as pd

def singleHighOutput(data, lstm_layer_size=128, lstm_layer_count=2):
    dataset_x = []
    dataset_y = []

    train_x = []
    train_y = []

    for i in range(30, len(data) - 1):
        dataset_x.append(data[i-30:i])
        dataset_y.append(data[i+1][0])
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = splitdata(dataset_x, dataset_y, [.8,0,.2])

    #reshape
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 6))
    train_y = np.reshape(train_y, (train_y.shape[0], 1))

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 6))
    test_y = np.reshape(test_y, (test_y.shape[0], 1))

    layers = []

    for i in range(0, lstm_layer_count):
        if i != lstm_layer_count - 1:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=True, input_shape=(30,6), kernel_initializer='glorot_uniform')) 
        else:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=False, input_shape=(30,6), kernel_initializer='glorot_uniform'))
        layers.append(tf.keras.layers.Dropout(.2))
        
    layers.append(tf.keras.layers.Dense(6, kernel_initializer='glorot_uniform'))
    layers.append(tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform'))


    model = tf.keras.Sequential(layers)

    return (model, train_x, train_y, valid_x, valid_y, test_x, test_y)

def next5HighOutput(data, lstm_layer_size=64, lstm_layer_count=4):
    dataset_x = []
    dataset_y = []

    train_x = []
    train_y = []

    for i in range(30, len(data) - 5):
        dataset_x.append(data[i-30:i])
        dataset_y.append([row[0] for row in data[i+1:i+6]])
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = splitdata(dataset_x, dataset_y, [.8,0,.2])

    #reshape
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 6))
    train_y = np.reshape(train_y, (train_y.shape[0], 5))

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 6))
    test_y = np.reshape(test_y, (test_y.shape[0], 5))

    layers = []

    for i in range(0, lstm_layer_count):
        if i != lstm_layer_count - 1:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=True, input_shape=(30,6), kernel_initializer='glorot_uniform')) 
        else:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=False, input_shape=(30,6), kernel_initializer='glorot_uniform'))
        layers.append(tf.keras.layers.Dropout(.3))
        
    layers.append(tf.keras.layers.Dense(125, kernel_initializer='glorot_uniform'))
    layers.append(tf.keras.layers.Dense(5, kernel_initializer='glorot_uniform'))


    model = tf.keras.Sequential(layers)

    return (model, train_x, train_y, valid_x, valid_y, test_x, test_y)

def next1AllOutput(data, lstm_layer_size=256, lstm_layer_count=3):
    dataset_x = []
    dataset_y = []

    train_x = []
    train_y = []

    for i in range(30, len(data) - 1):
        dataset_x.append(data[i-30:i])
        dataset_y.append(data[i+1])
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = splitdata(dataset_x, dataset_y, [.8,0,.2])

    #reshape
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 6))
    train_y = np.reshape(train_y, (train_y.shape[0], 6))

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 6))
    test_y = np.reshape(test_y, (test_y.shape[0], 6))

    layers = []

    for i in range(0, lstm_layer_count):
        if i != lstm_layer_count - 1:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=True, input_shape=(30,6), kernel_initializer='glorot_uniform')) 
        else:
            layers.append(tf.keras.layers.LSTM(lstm_layer_size, return_sequences=False, input_shape=(30,6), kernel_initializer='glorot_uniform'))
        layers.append(tf.keras.layers.Dropout(.2))
        
    layers.append(tf.keras.layers.Dense(6, kernel_initializer='glorot_uniform'))
    layers.append(tf.keras.layers.Dense(6, kernel_initializer='glorot_uniform'))


    model = tf.keras.Sequential(layers)

    return (model, train_x, train_y, valid_x, valid_y, test_x, test_y)


def splitdata(dataset_x, dataset_y, splitdef):

    train_size = splitdef[0]
    valid_size = splitdef[1]
    test_size = splitdef[2]

    if train_size + valid_size + test_size != 1:
        print('data split definition does not add to 1!')
        return None
    
    #80:20 test train split
    train_x = dataset_x[0:int(len(dataset_x)*train_size)]
    valid_x = []
    if valid_size == 0:
        valid_x = dataset_x[int(len(dataset_x)*train_size):int(len(dataset_x)*(train_size+valid_size))]
    test_x = dataset_x[int(len(dataset_x)*(train_size + valid_size)):len(dataset_x)]

    train_y = dataset_y[0:int(len(dataset_y)*train_size)]
    valid_y = []
    if valid_size == 0:
        valid_y = dataset_y[int(len(dataset_y)*train_size):int(len(dataset_y)*(train_size+valid_size))]
    test_y = dataset_y[int(len(dataset_y)*(train_size + valid_size)):len(dataset_y)]

    return (train_x, train_y, valid_x, valid_y, test_x, test_y)
