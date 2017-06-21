import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import backend as K
import os
from keras.models import model_from_json
import sys
from dataSource import DataSource
from encoder import Model

#x=np.load('X.npy')
#y=np.load('Y.npy' )
#y = to_categorical(y)
encoder = Model()

def log(*s):
    print("TRAIN_OPENAI:", s)

def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(2, init='lecun_uniform'))
    model.add(Activation('softmax'))

    return model

def new_baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), init='glorot_normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2000, init='glorot_normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, init='lecun_uniform'))
    model.add(Activation('softmax'))

    return model

def train(num_epochs):
    if not os.path.exists("model.h5"):
        log("trained model not found, making new model")
        bm=baseline_model()
    else:
        log("trained model found, loading it...")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        bm = model_from_json(loaded_model_json)
        bm.load_weights("model.h5")

    bm.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    datasource = DataSource("../data/full_train.txt", 20000)

    with K.get_session():
        for oi in range(num_epochs):
            log("OpenAI Batch ", oi)
            x, y = datasource.train_bitches()
            x = encoder.transform(x)
            y = to_categorical(y, 2)
            x_train, y_train, x_test, y_test = datasource.make_cvset(x, y, 0.1)
            bm.fit(x_train, y_train, batch_size=256, epochs=1, verbose=1, 
                    validation_data=(x_test, y_test), shuffle=False)
            if (oi + 1) % 30 == 0:
                log("saving model")
                model_json = bm.to_json()
                with open("output/" + str(oi) + ".json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                bm.save_weights("output/" + str(oi) + ".h5")
        make_submission(bm)
        model_json = bm.to_json()
        with open("output/final.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        bm.save_weights("output/final.h5")
        K.clear_session()
        log("Saved model to disk")

def make_submission(model):
    submit_examples = list(open("../twitter-datasets/test_data.txt", "r").readlines())
    submit_examples = [s.strip() for s in submit_examples]  
    splitter = [s.split(',', 1) for s in submit_examples]
    sentences = np.array([s[1] for s in splitter])
    ids = [s[0] for s in splitter]
    x = encoder.transform(sentences)
    result = np.argmax(model.predict(x, batch_size=256, verbose=1), axis=1)
    result = ['1' if r == 1 else '-1' for r in result]
    log('saving submissions')
    with open("../twitter-datasets/openaisubmission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(result):
            f.write(str(ids[i]) + "," + str(p) + "\n")

def load_and_submit(id='final'):
    log("loading model...")
    json_file = open('output/' + id + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("output/" + id + ".h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    log("Loaded model from disk")
    make_submission(loaded_model)
    #score = loaded_model.evaluate(x_test, y_test, verbose=1)
    #log('Test score:', score[0])
    #log('Test accuracy:', score[1])

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]))
    elif sys.argv[1] == "submit":
        load_and_submit(sys.argv[2])
    else:
        raise NotImplementedError
