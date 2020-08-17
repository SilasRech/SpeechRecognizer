import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()

    statesequence = [1, 2, 3, 3, 37, 38, 39, 40, 41, 42, 42, 1, 2, 3, 0] # please remind the changings in hmm class for optimizing the dnn

    words = hmm.getTranscription(statesequence)
    print(words) # ['ONE', 'TWO', 'THREE']

    plt.imshow(np.exp(hmm.logA))
    plt.xlabel('in Zustand j')
    plt.ylabel('von Zustand i')
    plt.colorbar()
    plt.show()
