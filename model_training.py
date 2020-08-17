import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
import argparse
import glob
import os
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


def train_model(datadir, hmm, model_dir, parameters, epochs):

    # get paths to data and sort them
    x_dirs = glob.glob(datadir + "/TRAIN/wav/*.wav")
    x_dirs = sorted(x_dirs)
    y_dirs = glob.glob(datadir + "/TRAIN/TextGrid/*.TextGrid")
    y_dirs = sorted(y_dirs)
    x_dirs_test = glob.glob(datadir + "/TEST/wav/*.wav")
    x_dirs_test = sorted(x_dirs_test)
    y_dirs_test = glob.glob(datadir + "/TEST/TextGrid/*.TextGrid")
    y_dirs_test = sorted(y_dirs_test)

    # calculate input shape for model
    input_shape = [parameters['num_ceps']*3, parameters['left_context'] + parameters['right_context'] + 1]

    len = (input_shape[0], input_shape[1])

    model = rec.dnn_model(len, hmm.get_num_states())

    rec.train_model(model, model_dir, [x_dirs, x_dirs_test], [y_dirs, y_dirs_test], hmm, sampling_rate, parameters, steps_per_epoch=10, epochs=epochs)


if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /my_path/TIDIGITS-ASE
    # call:
    # python uebung7.py <data/dir>
    # e.g., python uebung7.py /my_path/TIDIGITS-ASE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Data dir')
    args = parser.parse_args()
    print(args)

    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
                  'hop_size': 10e-3,
                  'feature_type': 'MFCC_D_DD',
                  'n_filters': 24,
                  'fbank_fmin': 0,
                  'fbank_fmax': 8000,
                  'num_ceps': 13,
                  'left_context': 6,
                  'right_context': 6,
                  }
    sampling_rate = 16000

    # number of epoches (not too many for the beginning)
    epochs = 3

    # define a name for the model, e.g., 'dnn'
    model_name = 'dnn'
    # directory for the model
    model_dir = os.path.join('exp', model_name + '.h5')
    if not os.path.exists('exp'):
        os.makedirs('exp')

    # default HMM
    hmm = HMM.HMM()

    # train DNN
    train_model(args.datadir, hmm, model_dir, parameters, epochs)

    model = tf.keras.models.load_model(model_dir)
    model = keras.Sequential([model, keras.layers.Softmax()])
    
    # get posteriors for test file
    post = rec.wav_to_posteriors(model, 'data/TEST-MAN-AH-3O33951A.wav', parameters)
    plt.imshow(post.transpose(), origin='lower')
    plt.xlabel('Frames')
    plt.ylabel('HMM states')
    plt.colorbar()
    plt.show()
