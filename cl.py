
from keras.datasets import imdb
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
from sklearn.metrics import accuracy_score
import os

#Global parameters fo here, these are just dummy examples

top_words = 5000
numpy.random.seed(7)
max_review_length = 500
embedding_vecor_length = 32

#Load data
def load_data():
    #TODO
    #For know will fill it with some dummy data
    # load the dataset but only keep the top n words, zero the rest
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    return (X_train,y_train,X_test,y_test)

#Create model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    return model

#Train model and tell us how good it is
def train(X_train, y_train,X_test, y_test,iterations,batch):
    model=baseline_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())


    model.fit(X_train, y_train,epochs=iterations, batch_size=batch, verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy: %.2f%%" % (scores[1]*100))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

#you can predict on X using this method where positive is the label number for positive samples and negative the label for the rest.
def predict(X,positive,negative):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_predict=loaded_model.predict(X)
    predictor = lambda x: negative if x < 0.5 else positive
    pfunc = numpy.vectorize(predictor)

    return pfunc(y_predict)

#So we should load the data first here
(X_train,y_train,X_test,y_test)=load_data()

#We can train it here the model will be saved to the disk automatically, so next time when you run the code just comment this , training comes with a cost :)
train(X_train,y_train,X_test,y_test, 1,64)

#We can just load the model from the dist and predict using it , here is a simple test to see if model is loaded correctly and accuracy comlies with result of us

print(accuracy_score(predict(X_test,1,0),y_test))

