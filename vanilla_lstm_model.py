import argparse
from base_model import BaseModel
from data_source import DataSource
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from utils.vocabulary import Vocabulary

DROPOUT = 0.2
LSTM_SIZE = 1024
SEQ_LEN = 40

NUM_EPOCHS = 20
BATCH_SIZE = 64


class VanillaLSTMModel(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN, lstm_arch='vanilla'):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length)
        self.lstm_arch = lstm_arch

    def create_model(self, ckpt_file=None):
        if ckpt_file is None:
            self.model = Sequential()
            self.model.add(self.embedding_layer)

            if self.lstm_arch == "vanilla":
                self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))
            elif self.lstm_arch == "multi_layer":
                # Two LSTM layers.
                self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True,
                                    return_sequences=True))
                self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))
                # Add 2 fully-connected layers.
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(512, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
            elif self.lstm_arch == "bidi":
                lstm_layer = LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                  recurrent_dropout=self.drop_prob,
                                  implementation=2, unroll=True,
                                  return_sequences=True)
                self.model.add(Bidirectional(lstm_layer, merge_mode='sum'))

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
    parser.add_argument('--vanilla', type=str, dest='lstm_arch',
                        action='store_const', const='vanilla')
    parser.add_argument('--bidi', type=str, dest='lstm_arch',
                        action='store_const', const='bidi')
    parser.add_argument('--multi_layer', type=str, dest='lstm_arch',
                        action='store_const', const='multi_layer')

    args = parser.parse_args()

    vocab = Vocabulary("data/vocab.pkl")
    data_source = DataSource(
        vocab=vocab,
        labeled_data_file="data/small_train.txt",
        test_data_file="data/test_data.txt",
        embedding_file="data/glove.twitter.27B.25d.txt",
        embedding_dim=25,
        seq_length=SEQ_LEN)

    model = VanillaLSTMModel(vocab, data_source, LSTM_SIZE, DROPOUT,
                             args.lstm_arch)

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
