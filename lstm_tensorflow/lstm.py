from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from data_source import *
from keras.utils.np_utils import to_categorical
import sys
import numpy as np

max_features = 20000
max_len = 40 # Tweak this parameters depeding on your need
batch_size = 32
LSTM_SIZE = 1024
DROPOUT = 0.2
EPOCHS = 10

def tokenize_data(X_raw, Y_raw):
    tokenizer = Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    X_processed = sequence.pad_sequences(sequences, maxlen=max_len)
    Y_processed = to_categorical(np.asarray(Y_raw), 2)

    return X_processed, Y_processed

def run_lstm(small_or_big):
    (x_train, y_train), (x_test, y_test) = get_train_test_data(small_or_big)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
  
    x_train, y_train = tokenize_data(x_train, y_train)
    x_test, y_test = tokenize_data(x_test, y_test)

    print('Build the model')
    model = Sequential()
    
    # Embeddding Layer
    model.add(Embedding(max_features, 128))
    
    # LSTM Layer
    model.add(LSTM(LSTM_SIZE, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    
    # Dense Layer
    model.add(Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    print('Train the model')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=EPOCHS,
            validation_data=(x_test, y_test))
    
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    model.save('model.h5')
    model.save_weights('weights.h5')
    
    print('Test score: ', score)
    print('Test accuracy: ', acc)
    
if __name__ == "__main__":
    run_lstm(sys.argv[-1])
