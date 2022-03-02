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

import models

model, train_x, train_y, valid_x, valid_y, test_x, test_y = models.singleHighOutput(data)

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

#do visualizations here

