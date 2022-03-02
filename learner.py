import tensorflow as tf
import numpy as np
import pandas as pd

#load a single data file for now
data = open('stocks/AG.csv', 'r')

#split data up
#lets grab a line count
lines = data.readlines()
print("lines:",len(lines))
#split train,test
data = []

maxes = [0] * 7
for i in range(1,int(len(lines))):
    line = lines[i].replace('\n', '')
    split = line.split(',')
    for j in range(1, len(split)):
        if maxes[j] < float(split[j]):
            maxes[j] = float(split[j])


for i in range(1,int(len(lines))):
    line = lines[i].replace('\n', '') 
    temparr = []
    split = line.split(',')
    temparr.append(float(split[1]) / maxes[1])
    temparr.append(float(split[2]) / maxes[2])
    temparr.append(float(split[3]) / maxes[3])
    temparr.append(float(split[4]) / maxes[4])
    temparr.append(float(split[5]) / maxes[5])
    temparr.append(float(split[6]) / maxes[6])
    data.append(temparr)


#for i in range(stopindex, stopindex+int((len(lines)*.2))):
#    line = lines[i].replace('\n', '')
#    temparr = []
#    split = line.split(',')
#    temparr.append(float(split[1]))
#    temparr.append(float(split[2]))
#    temparr.append(float(split[3]))
#    temparr.append(float(split[4]))
#    temparr.append(float(split[5]))
#    temparr.append(float(split[6]))
#    test.append(temparr)


#print(train)
#print(test)

def slide(slide_size, start_index, arr):
    return arr[start_index : start_index + slide_size]


dataset_x = []
dataset_y = []

train_x = []
train_y = []

for i in range(30, len(data) - 1):
    dataset_x.append(data[i-30:i])
    dataset_y.append(data[i+1][0])


#80:20 test train split
train_x = dataset_x[0:int(len(dataset_x)*.8)]
test_x = dataset_x[int(len(dataset_x)*.8):len(dataset_x)]

train_y = dataset_y[0:int(len(dataset_y) * .8)]
test_y = dataset_y[int(len(dataset_y) * .8) : len(dataset_y)]

#reshape
train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 6))
train_y = np.reshape(train_y, (train_y.shape[0], 1))

test_x = np.array(test_x)
test_y = np.array(test_y)
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 6))
test_y = np.reshape(test_y, (test_y.shape[0], 1))

#shape verification
print(train_x[0])
print(train_y[0])

model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30,6), kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.LSTM(128, return_sequences=False, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(6, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')
    ])

def loss(labels, logits):
    return tf.keras.losses.mse(labels, logits)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

print(model.summary())

model.fit(train_x, train_y, epochs=50, batch_size=4)

#test_case = test_x[0]

#test_case = np.reshape(test_case, (1, test_x.shape[1], test_x.shape[2]))

test_predictions = model.predict(test_x)
for i in range(0, len(test_predictions)):
    print(test_predictions[i], '-', test_y[i])



