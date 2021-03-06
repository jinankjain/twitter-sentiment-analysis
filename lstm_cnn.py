import argparse
from base_model import BaseModel
from data_source import DataSource
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from utils.vocabulary import Vocabulary

DROPOUT = 0.2
LSTM_SIZE = 1024
SEQ_LEN = 40

NUM_EPOCHS = 2
BATCH_SIZE = 64

# Convolutional Layer Params
KERNEL_SIZE = 3
FILTERS = 64
POOL_SIZE = 4
CONV_DROP = 0.25

class LSTMCNN(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length)

    def create_model(self, ckpt_file=None):
        if ckpt_file is None:
            self.model = Sequential()
            self.model.add(self.embedding_layer)
            self.model.add(Dropout(CONV_DROP))
            self.model.add(Conv1D(FILTERS,
                                  KERNEL_SIZE,
                                  padding='valid',
                                  activation='relu',
                                  strides=1))
            self.model.add(MaxPooling1D(pool_size=POOL_SIZE))
            self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                recurrent_dropout=self.drop_prob,
                                unroll=True))
            self.model.add(Dense(2, activation='softmax'))

            # TODO: maybe try to do pooling over hidden states like here:
            # http://deeplearning.net/tutorial/lstm.html
        else:
            # Load model from checkpoint file.
            self.model = load_model(ckpt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', dest='is_train', action='store_true')
    parser.add_argument('--is_eval', dest='is_train', action='store_false')
    parser.add_argument('--ckpt_file', type=str, default=None, nargs="?",
                        help='Path to checkpoint file')

    args = parser.parse_args()

    vocab = Vocabulary("data/vocab.pkl")
    data_source = DataSource(
        vocab=vocab,
        labeled_data_file="data/small_train.txt",
        test_data_file="data/test_data.txt",
        embedding_file="data/glove.twitter.27B.200d.txt",
        embedding_dim=200,
        seq_length=SEQ_LEN)

    model = LSTMCNN(vocab, data_source, LSTM_SIZE, DROPOUT)

    if args.is_train:
        model.create_model()
        model.train(NUM_EPOCHS, BATCH_SIZE)
    else:
        model.create_model(args.ckpt_file)
        print("Evaluating on validation set...")
        model.eval()

        print("Predicting labels on test set...")
        y_test = model.predict()
        with open("data/test_output.txt", "w") as f:
            f.write("Id,Prediction\n")
            for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                f.write(str(idx) + "," + str(y) + "\n")
