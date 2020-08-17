import glob
import os
import recognizer.hmm as HMM
import recognizer.dnn_recognizer as rec
import tensorflow as tf
import recognizer.tools as tools
import random
import argparse
from tensorflow import keras


def test_model(datadir, hmm, model_dir, parameters):
  
    # load test data
    x_dirs = glob.glob(datadir + "/TEST/wav/*.wav")
    x_dirs = sorted(x_dirs)
    y_dirs = glob.glob(datadir + "/TEST/lab/*.lab")
    y_dirs = sorted(y_dirs)
    
    # shuffle data
    shuffled_list = list(zip(x_dirs,y_dirs))
    random.shuffle(shuffled_list)
    x_dirs, y_dirs = zip(*shuffled_list)
    
    # load model 
    model = tf.keras.models.load_model(model_dir)
    model = keras.Sequential([model, keras.layers.Softmax()])
    
    # initialize deletions D, insertions I, substitutions S and number of words N
    N_ges = 0
    D_ges = 0
    I_ges = 0
    S_ges = 0
    
    for index in range(len(y_dirs)):
        # read an split labels
        with open(y_dirs[index], 'r') as f:
            seq = f.read()
            ref_seq = seq.split()
            f.close()
        
        # get transcription from wav files
        posteriors = rec.wav_to_posteriors(model, x_dirs[index], parameters)
        word_seq = hmm.posteriors_to_transcription(posteriors)
        print('REF:' + ' ', sep = '', end = '', flush = True)
        print(ref_seq)
        print('OUT:' + ' ', sep = '', end = '', flush = True)
        print(word_seq)
        
        # calculate parameters
        N, D, I, S = tools.needlemann_wunsch(ref_seq, word_seq)
        
        # accumulate
        N_ges += N
        D_ges += D
        I_ges += I
        S_ges += S
        
        print('I:' + ' ' + str(I) + '\t' + 'D:' + ' ' + str( D) + '\t' + 'S:' + ' ' + str(S) + '\t' + 'N:' + ' ' + str(N))
        
        # calculate word error rate WER
        WER = 100 * ((D_ges + I_ges + S_ges)/N_ges)
        print(WER)
        print(str(index) + ' of ' + str(len(y_dirs)))
        
    return WER


if __name__ == "__main__":
    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung11.py <data/dir>
    # e.g., python uebung11.py /media/public/TIDIGITS-ASE
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Data dir')
    args = parser.parse_args()
    

    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_D_DD',
        'n_filters': 24,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        'left_context': 6,
        'right_context': 6}

    # default HMM
    hmm = HMM.HMM()

    # define a name for the model, e.g., 'dnn'
    model_name = 'dnn'
    # directory for the model
    model_dir = os.path.join('exp', model_name + '.h5')

    # test DNN
    wer = test_model(args.datadir, hmm, model_dir, parameters)

    print('--' * 40)
    print("Total WER: {}".format(wer))
