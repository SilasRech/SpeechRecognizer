# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

###############################################################################
# functions
###############################################################################
def get_data():
    
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    
    trainY = np_utils.to_categorical(trainY, 10)
    testY = np_utils.to_categorical(testY, 10)
    
    label_names = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    
    data_dict = {"train" : {"image" : trainX, 
                            "label" : trainY},
                 "test" : {"image" : testX, 
                           "label" : testY}, 
                 "labels": label_names
                 }
    
    return data_dict
  

def get_model():
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10))
    model.compile(optimizer="adam", 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=["accuracy"])
    
    model.summary()
    return model


###############################################################################
# main part
###############################################################################
keras.backend.clear_session()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
label_names = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

fig = plt.figure()
for i in range(12):
    ax = fig.add_subplot(3, 4, i + 1)
    ax.imshow(x_train[i,:,:],cmap="gist_gray")
    ax.set_title(label_names[y_train[i]], fontsize=12)
plt.subplots_adjust(hspace=0.4)
plt.show()
    
data = get_data()
model = get_model()

epochs = 10
batch_size = 20
loss_hist = model.fit(data["train"]["image"],
                      data["train"]["label"],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(data["test"]["image"],
                                       data["test"]["label"]))
model.save("fmnist_ff.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), loss_hist.history['loss'], label="train_loss")
plt.plot(np.arange(0, epochs), loss_hist.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, epochs), loss_hist.history['acc'], label="train_acc")
plt.plot(np.arange(0, epochs), loss_hist.history['val_acc'], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
plt.show()


##############################################################################
# Aufgabe 6.1
##############################################################################

# load model
model = tf.keras.models.load_model("fmnist_ff.h5")
# add Softmax layer
model = keras.Sequential([model, keras.layers.Softmax()])
model.summary()

# predict 
preds = model.predict(x_test)

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.imshow(x_test[4,:,:], cmap='gist_gray')
ax = fig.add_subplot(2,1,2)
ax.barh(label_names,preds[4,:])














