#
# Character-Based Neural Network LSTM for predicting characters as part of the Speech Recognition
#

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, Conv2D, MaxPooling2D, Lambda, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard
from xml.dom import minidom

data = ""
embed_dim = 128
lstm_out = 200
batch_size = 32

german2 = minidom.parse('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Polit\\AA.xml')
items2 = german2.getElementsByTagName('rohtext')
for elem in items2:
    data = data + elem.childNodes[0].data


tokenizer = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1

sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-2:i+3]
    sequences.append(sequence)

X = [l[0:4] for l in sequences[1:-1]]
Y = [l[-1] for l in sequences[1:-1]]
model = Sequential()
model.add(Embedding(2500, embed_dim, input_length=X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

