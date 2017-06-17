
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import os
from keras.models import model_from_json
import sys
from dataSource import DataSource
from encoder import Model

#x=np.load('X.npy')
#y=np.load('Y.npy' )
#y = to_categorical(y)

def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(2, init='lecun_uniform'))
    model.add(Activation('softmax'))

    return model


def train():
    encoder = Model()
    if not os.path.exists("model.h5"):
        print("trained model not found, making new model")
        bm=baseline_model()
    else:
        print("trained model found, loading it...")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        bm = model_from_json(loaded_model_json)
        bm.load_weights("model.h5")

    bm.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    datasource = DataSource("../data/small_train.txt")

    oi = 0
    for x_train, y_train, x_test, y_test in train_bitches(20000):
        print("OpenAI Batch ", oi)
        oi += 1
        x_train = encoder.transform(x_train)
        x_test = encoder.transform(x_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        bm.fit(x_train, y_train,
                        batch_size=256, nb_epoch=1,
                        verbose=1, validation_data=(x_test, y_test), shuffle=False)
    model_json = bm.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    bm.save_weights("model.h5")
    print("Saved model to disk")

    score = bm.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def evaluate():
    print("loading model...")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    print("Loaded model from disk")
    score = loaded_model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        evaluate()
