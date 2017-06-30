from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam
from keras.utils.np_utils import to_categorical

import numpy as np
import os

POS = 1
NEG = -1
STEPS_PER_CKPT = 200000
CKPT_DIR = "data/checkpoints/"


class BaseModel:
    def __init__(self, vocab, data_source, lstm_size, drop_prob, seq_length,
                 arch):
        self.vocab = vocab
        self.data_source = data_source

        self.drop_prob = drop_prob
        self.seq_length = seq_length

        # Create checkpoint folder if not present
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)


        self.model = None
        self.embedding_layer = None
        self.arch = arch

        # Create the embedding layer and load pretrained embeddings.
        if self.arch == "conv2":
            embedding_matrix = self.data_source.get_embeddings("glove")
            self.embedding_layer = Embedding(
                input_dim=self.vocab.vocab_size,
                output_dim=self.data_source.embedding_dim,
                weights=[embedding_matrix],
                input_length=seq_length,
                trainable=True)

            #embedding_matrix = self.data_source.get_embeddings("word2vec")
            #self.embedding_layer2 = Embedding(
            #    input_dim=self.vocab.vocab_size,
            #    output_dim=self.data_source.embedding_dim,
            #    weights=[embedding_matrix],
            #    input_length=seq_length,
            #    trainable=True)
        else:
            embedding_matrix = self.data_source.get_embeddings()
            self.embedding_layer = Embedding(
                input_dim=self.vocab.vocab_size,
                output_dim=self.data_source.embedding_dim,
                weights=[embedding_matrix],
                input_length=seq_length,
                trainable=True)

    """
    This method should create a keras model.
    """
    def create_model(self, ckpt_file=None):
        raise NotImplementedError("Please implement this method")

    def train(self, batch_size, loss='categorical_crossentropy'):
        X_val, y_val, openai_features = None, None, None
        if self.arch is not "ensemble":
            X_val, y_val = self.data_source.validation()
        else:
            X_val, y_val, val_openai_features = self.data_source.validation()

        y_val = to_categorical((y_val + 1) / 2, num_classes=2)

        opt = Adam(lr=0.0001)
        opt_name = "adam"
#         opt = RMSprop(lr=0.001, decay=0.95)
#         opt_name = "RMSP"
        self.model.compile(
            loss=loss,
            optimizer=opt,
            metrics=['accuracy'])
        print(self.model.summary())

        iteration = 0
        while True:
            checkpoint = ModelCheckpoint(
                filepath=CKPT_DIR + self.arch + '_lstm_' + opt_name + \
                        '_ckpt-' + str(iteration) + '-{val_loss:.2f}.hdf5')
            curr_X_train, curr_y_train, openai_features = None, None, None
            if self.arch is not "ensemble":
                curr_X_train, curr_y_train = self.data_source.next_train_batch(
                    STEPS_PER_CKPT)
            else:
                curr_X_train, curr_y_train, openai_features = self.data_source.next_train_batch(
                    STEPS_PER_CKPT, with_openai_features=True)
            curr_y_train = to_categorical((curr_y_train + 1) / 2, num_classes=2)

            input_X, val_X = None, None
            if self.arch is not "ensemble":
                input_X = curr_X_train
                val_X = X_val
            else:
                input_X = [curr_X_train, openai_features]
                val_X = [X_val, val_openai_features]
            print(input_X[0].shape, input_X[1].shape)

            self.model.fit(
                input_X,
                curr_y_train,
                validation_data=(val_X, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=1,
                callbacks=[checkpoint])
            if self.arch == "conv_lstm":
                y_test = self.predict()
                with open("data/ensemble_test_outputs/conv_lstm_test_out_"+
                        str(iteration)+".txt", "w") as f:
                    f.write("Id,Prediction\n")
                    for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                        f.write(str(idx+1) + "," + str(y) + "\n")
            iteration += STEPS_PER_CKPT

    """
    Evaluate the model on the validation set.
    """
    def eval(self):
        X, y, openai_features = None, None, None
        if self.arch is not "ensemble":
            X, y = self.data_source.validation()
        else:
            X, y, openai_features = self.data_source.validation()
        y = to_categorical((y + 1) / 2, num_classes=2)

        if self.arch is not "ensemble":
            scores = self.model.evaluate(X, y, verbose=0)
        else:
            scores = self.model.evaluate([X, openai_features], y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    """
    Get the predictions of the model on a test set.
    """
    def predict(self):
        X, openai_features = None, None
        if self.arch is not "ensemble":
            X = self.data_source.test()
        else:
            X, openai_features = self.data_source.test()
        if self.arch is not "ensemble":
            y_predict = np.argmax(self.model.predict(X), axis=1)
        else:
            y_predict = np.argmax(
                    self.model.predict([X, openai_features]), axis=1)
        y_predict = y_predict * 2 - 1

        return y_predict

    """
    Store model in JSON format and store the weights in HDFS.
    """
    def save_model(self, filepath, filename):
        model_json = self.model.to_json()
        with open(filepath + "/" + filename + ".json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(filepath + "/" + filename + ".h5")
        print("Saved model to disk")

    """
    Load model.
    """
    def load_model(self, filepath, filename):
        model_json = None
        with open(filepath + "/" + filename + ".json", "r") as f:
            model_json = f.read()

        self.model = model_from_json(model_json)

        # Load model weights.
        self.model.load_weights(filepath + "/" + filename + ".h5")
        print("Loaded model from disk")
