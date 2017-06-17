
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import os
from keras.models import model_from_json

x=np.load('X.npy')
y=np.load('Y.npy' )
y = to_categorical(y)

def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(2, init='lecun_uniform'))
    model.add(Activation('softmax'))

    return model


cross_validation_indices = np.array(random.sample(list(np.arange(len(y))), int(len(y) * 0.1) ))
train_indices = np.array(list(set(np.arange(len(y))) - set(cross_validation_indices)))

x_train, x_test= x[train_indices], x[cross_validation_indices]
y_train, y_test = y[train_indices], y[cross_validation_indices]


if not os.path.exists("model.h5"):
    bm=baseline_model()
    bm.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    bm.fit(x_train, y_train,
                        batch_size=1000, nb_epoch=1,
                        verbose=1, validation_data=(x_test, y_test), shuffle=True)
    model_json = bm.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    bm.save_weights("model.h5")
    print("Saved model to disk")

    score = bm.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

else:
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
