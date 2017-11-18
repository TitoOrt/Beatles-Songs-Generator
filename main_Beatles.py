# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:21:33 2017

@author: joseluis
"""
#Thids is a project for ... we'll see ;)


import pylyrics3 #web lyrics scraper
import time 
import json
import pandas as pd
import sys

import tensorflow as TF

#%%
#This extracts all the songs from the beatles and stores them as a dictionary in python 
btls = pylyrics3.get_artist_lyrics('beatles')

#We will save a copy in case the web site goes down eventually
btlsDF = pd.DataFrame(list(btls.items()), columns = ("Song","Lyrics"))

#%%
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#%%
raw_text = btlsDF['Lyrics'].str.cat().lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

#%%
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

#%%
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

#%%

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#%%
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


#%%

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#%%
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)









#%%



#%%


# load the network weights
filename = "weights-improvement-12-1.2382-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


#%%
int_to_char = dict((i, c) for i, c in enumerate(chars))

#%%
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
#%%
# generate characters
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
for i in range(300):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")






