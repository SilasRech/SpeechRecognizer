# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
import numpy as np
import recognizer.feature_extraction as fe
import recognizer.tools as tools
from itertools import chain
import random

def wav_to_posteriors(model, audio_file, parameters):

    feats = fe.compute_features_with_context(audio_file, **parameters)

    # predict posteriors with the trained model
    posteriors = model.predict(feats)

    return posteriors


def generator(x_dirs, y_dirs, hmm, sampling_rate, parameters):
    feats_list = []
    target_list = []

    length_feats = 0
    length_target = 0
    number_features = parameters['num_ceps'] * 3
    number_context = parameters['left_context'] + parameters['right_context'] + 1

    for i in range(len(x_dirs)):
        # Compute audiofile to feature matrix
        # get path to audio file
        audio_file = x_dirs[i]

        # compute features
        feats = fe.compute_features_with_context(audio_file, **parameters)
        
        # get label
        target_dir = y_dirs[i]
        
        # calculate window size and hop size
        window_size_samples = tools.sec_to_samples(parameters['window_size'], sampling_rate)
        window_size_samples = 2 ** tools.next_pow2(window_size_samples)

        hop_size_samples = tools.sec_to_samples(parameters['hop_size'], sampling_rate)

        # calculatge target
        target = tools.praat_file_to_target(target_dir, sampling_rate, window_size_samples, hop_size_samples, hmm)
        
        # append to list with features and targets
        length_feats += len(feats)
        length_target += len(target)
        feats_list.append(feats)
        target_list.append(target)

    target_list = list(chain.from_iterable(target_list))
    feats_list = list(chain.from_iterable(feats_list))

    feats_list_new = np.reshape(np.array(feats_list), newshape=(length_feats, number_features, number_context))
    target_list_new = np.reshape(np.array(target_list), newshape=(length_feats, hmm.get_num_states()))

    return feats_list_new, target_list_new
           

def train_model(model, model_dir, x_dirs, y_dirs, hmm, sampling_rate, parameters, steps_per_epoch=10,
                epochs=10, viterbi_training=False):
    # load train and test data
    data_train = x_dirs[0]
    target_train = y_dirs[0]

    data_test = x_dirs[1]
    target_test = y_dirs[1]

    # prepare data for testing (just take first 300)
    data_test_new, target_test_new = generator(data_test[:300], target_test[:300], hmm, sampling_rate, parameters)

    number_chunks = 20

    # epochs
    for j in range(epochs):
        # shuffle data
        shuffled_list = list(zip(data_train,target_train))
        random.shuffle(shuffled_list)
        data_train, target_train = zip(*shuffled_list) 
        
        # get position of 'batches'
        data_chunks = np.linspace(0, len(x_dirs[0]), num=number_chunks).astype(int)
        
        # batches
        for i in range(number_chunks-1):
            print('Epoch:' + ' ' + str(j) + '\t' + 'Batch:' + ' ' + str(i))
            data_train_new, target_train_new = generator(data_train[data_chunks[i]:data_chunks[i+1]], target_train[data_chunks[i]:data_chunks[i+1]], hmm, sampling_rate, parameters)
    
            mycallbacks = [tf.keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)]
    
            model.fit(data_train_new,
                      target_train_new,
                      epochs=1,
                      batch_size=150,
                      validation_data=(data_test_new, target_test_new),
                      callbacks=mycallbacks,
                      shuffle=True)

            model.save(model_dir)

#
def dnn_model(input_shape, output_shape):
    # TODO  Implementieren Sie hier Aufgabe 7.3

    keras.backend.clear_session()

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    # add batch normalization
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation='relu', name='dense_3'))
    # add dropout layer
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(output_shape, name='dense_5'))
    # change optimizer to 'Nadam'
    model.compile(optimizer='Nadam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    return model
