import tensorflow as tf
import numpy as np
import pandas as pd
import sys


#load a single data file for now
data = open('stocks/AG.csv', 'r')

#split data up
#lets grab a line count
lines = data.readlines()
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

final_errors = {}

for prev_days in range(5,15):
    
    #get_model = getattr(models, sys.argv[2])
    #model, train_x, train_y, valid_x, valid_y, test_x, test_y = get_model(data)

    #model, train_x, train_y, valid_x, valid_y, test_x, test_y = models.singleHighOutput(data, lstm_layer_size=512, lstm_layer_count=3)
    #model, train_x, train_y, valid_x, valid_y, test_x, test_y = models.next1AllOutput(data)
    model, train_x, train_y, valid_x, valid_y, test_x, test_y = models.next5HighOutput(data, lstm_layer_size=64, lstm_layer_count=4, prev_days=prev_days)

    def loss(labels, logits):
        return tf.keras.losses.mse(labels, logits)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

    epochs = None

    try:
        epochs = int(sys.argv[1])
    except IndexError:
        epochs = 10

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=4)

    #test_case = test_x[0]

    #test_case = np.reshape(test_case, (1, test_x.shape[1], test_x.shape[2]))

    self_error = 0
    count = 0

    test_predictions = model.predict(test_x)
    for i in range(0, len(test_predictions)):
        self_error += abs(test_predictions[i] - test_y[i])
        count+=1
        #print(test_predictions[i], '-', test_y[i])

    print('prev_days', str(prev_days))
    #print('final error', str(self_error/count))
    final_errors[prev_days] = history.history['loss'][-1]
    print(history.history)


#grab random 30 day splice
import random as rand
import matplotlib.pyplot as plt
def predict5():
    rand_start = rand.randint(0, len(data)-36)
    prior = data[rand_start:rand_start+30]
    prior = np.array(prior)
    prior = np.reshape(prior, (1, 30, 6))

    output = model.predict(prior)

    #create plot for actual and generated data
    actual = [row[0] for row in data[rand_start:rand_start+35]]

    guessed = [row[0] for row in data[rand_start:rand_start+30]]
    #guessed.append(guess for guess in output)
    for guess in output[0]:
        guessed.append(guess)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #build test_x + its outputs

    ax.plot(actual, color='tab:blue')
    ax.plot(guessed, color='tab:orange')
    plt.show()

def predict5With1():
    guesses = []
    rand_start = rand.randint(0, len(data)-36)
    for i in range(0, 5):
        prior = data[rand_start+i:rand_start+30] 
        #OOPS ok so the issue is that prior is already a 30 long list of 6 long lists... ie its 30x6. and then we tried to just add
        #a single digit entry.. a 1x1 to the end which doesnt make sense. Might have to build a model with just 30x1 to see performance
        prior.extend(guesses)
        prior = np.array(prior)
        prior = np.reshape(prior, (1, 30, 6))

        output = model.predict(prior)
        guesses.append(output[0])
 
    actual = [row[0] for row in data[rand_start:rand_start+35]]

    guessed = [row[0] for row in data[rand_start:rand_start+30]] 
    for guess in output[0]:
        guessed.append(guess)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #build test_x + its outputs

    ax.plot(actual, color='tab:blue')
    ax.plot(guessed, color='tab:orange')
    plt.show()

def buildErrorByDays():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = []
    y = []
    for prev_days in sorted(list(final_errors.keys())):
        x.append(prev_days)
        y.append(final_errors[prev_days])
    ax.plot(x, y)
    plt.show()

#add logic here but for now its hard coded
buildErrorByDays()
