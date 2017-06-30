import argparse
import h5py
from base_model import BaseModel
from data_source import DataSource
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Merge
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from utils.vocabulary import Vocabulary

DROPOUT = 0.2
LSTM_SIZE = 1024
SEQ_LEN = 40

OPENAI_FEATURE_SIZE = 4096
OPENAI_REDUCED_SIZE = 25
BATCH_SIZE = 64


class VanillaLSTMModel(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN, arch=None):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length, arch)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128

    def create_model(self, ckpt_file=None):
        if ckpt_file is None:
            if self.arch == "vanilla":
                self.model = Sequential()
                self.model.add(self.embedding_layer)
                self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "gru":
                self.model = Sequential()
                self.model.add(self.embedding_layer)
                self.model.add(GRU(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "conv":
                conv_filters = []
                for filter_size in self.filter_sizes:
                    conv_filters.append(Sequential())
                    conv_filters[-1].add(self.embedding_layer)
                    conv_filters[-1].add(Conv1D(filters=self.num_filters, kernel_size=filter_size,
                                          strides=1, padding='valid', activation='relu'))
                    conv_filters[-1].add(MaxPooling1D(pool_size=(self.seq_length - filter_size + 1)))
                self.model = Sequential()
                self.model.add(Merge(conv_filters, mode='concat'))
                self.model.add(Flatten())
                self.model.add(Dropout(2*DROPOUT))
                # TODO: maybe add another Dense layer
                self.model.add(Dense(512, activation='softmax'))
                self.model.add(Dropout(2*DROPOUT))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "conv2":
                # https://github.com/fchollet/keras/issues/233
                # input tensor has shape (samples, maxlen)
                emb_channels = []
                emb_channels.append(Sequential()) 
                #emb_channels[0].add(self.embedding_layer)
                #emb_channels.append(Sequential()) 
                #emb_channels[1].add(self.embedding_layer2)
                #self.model = Sequential()
                #self.model.add(Merge(emb_filters, mode='concat'))
                #self.model.add(Reshape((SEQ_LEN, self.data_source.embedding_dim, len(emb_channels))) # reshape into 4D tensor (samples, maxlen, 256, 2)
                # comment out bottom two lines and uncomment top lines + embedding_layer2 loading in base_model for multi channels
                self.model = Sequential()
                self.model.add(self.embedding_layer)
                # VGG-like convolution stack
                self.model.add(Conv2D(32, 3, padding='valid', activation='relu'))
                self.model.add(Conv2D(32, 3, activation='relu'))
                self.model.add(MaxPooling2D(pool_size=(2, 2)))
                self.model.add(Dropout(0.25))
                self.model.add(Flatten())
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "swisscheese":
                self.model = Sequential()
                filter_size = 3
                self.model.add(self.embedding_layer)
                self.model.add(Conv1D(filters=self.num_filters, kernel_size=filter_size,
                                        strides=1, padding='valid', activation='relu'))
                self.model.add(MaxPooling1D(pool_size=3))
                self.model.add(Conv1D(filters=self.num_filters/2, kernel_size=filter_size,
                                        strides=1, padding='valid', activation='relu'))
#                 self.model.add(MaxPooling1D(pool_size=(..TODO.. - filter_size + 1)))

                #self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(2, activation='softmax'))

            elif self.arch == "conv_lstm":
                print("Start")
                branch1 = Sequential()
                branch1.add(self.embedding_layer)
                branch1.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                 recurrent_dropout=self.drop_prob,
                                 implementation=2, unroll=True))

                print("Created first model")

                conv_filters = []
                for filter_size in self.filter_sizes:
                    conv_filters.append(Sequential())
                    conv_filters[-1].add(self.embedding_layer)
                    conv_filters[-1].add(Conv1D(filters=self.num_filters, kernel_size=filter_size,
                                          strides=1, padding='valid', activation='relu'))
                    conv_filters[-1].add(MaxPooling1D(pool_size=(self.seq_length - filter_size + 1)))
                branch2 = Sequential()
                branch2.add(Merge(conv_filters, mode='concat'))
                branch2.add(Flatten())
                branch2.add(Dropout(2*DROPOUT))
                print("Created second model")

                self.model = Sequential()
                self.model.add(Merge([branch1, branch2], mode='concat'))

                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "ensemble":
                print("Start")
                branch1 = Sequential()
                branch1.add(self.embedding_layer)
                branch1.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                 recurrent_dropout=self.drop_prob,
                                 implementation=2, unroll=True))

                print("Created first model")

                branch2 = Sequential()
                branch2.add(Dense(OPENAI_REDUCED_SIZE,
                    activation="linear", input_shape=(OPENAI_FEATURE_SIZE,)))
                print("Created second model")

                self.model = Sequential()
                self.model.add(Merge([branch1, branch2], mode='concat'))

                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "multi_layer":
                self.model = Sequential()
                self.model.add(self.embedding_layer)
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

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "bidi":
                self.model = Sequential()
                self.model.add(self.embedding_layer)
                lstm_layer = LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                  recurrent_dropout=self.drop_prob,
                                  implementation=2, unroll=True)
                self.model.add(Bidirectional(lstm_layer, merge_mode='sum'))

                self.model.add(Dense(512, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(2, activation='softmax'))
            else:
                raise NotImplementedError("Architecture is not implemented:",
                                          self.arch)

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
    parser.add_argument('--train_file', type=str,
                        default="data/small_train.txt",
                        help='Path to training set file')
    parser.add_argument('--loss', type=str,
                        default="categorical_crossentropy",
                        help='Loss function to be used')
    parser.add_argument('--emb_len', type=int, default=200,
                        help='Dimension of Glove embeddings.')
    parser.add_argument('--vanilla', dest='lstm_arch',
                        action='store_const', const='vanilla')
    parser.add_argument('--gru', dest='lstm_arch',
                        action='store_const', const='gru')
    parser.add_argument('--bidi', dest='lstm_arch',
                        action='store_const', const='bidi')
    parser.add_argument('--multi_layer', dest='lstm_arch',
                        action='store_const', const='multi_layer')
    parser.add_argument('--ensemble', dest='lstm_arch',
                        action='store_const', const='ensemble')
    parser.add_argument('--conv', dest='lstm_arch',
                        action='store_const', const='conv')
    parser.add_argument('--conv2', dest='lstm_arch',
                        action='store_const', const='conv2')
    parser.add_argument('--conv_lstm', dest='lstm_arch',
                        action='store_const', const='conv_lstm')
    parser.add_argument('--swisscheese', dest='lstm_arch',
                        action='store_const', const='swisscheese')
    parser.add_argument('--embedding_type', type=str, default="glove", nargs="?",
                        help='Type of word embedding Word2Vec or Glove')
    args = parser.parse_args()

    vocab = Vocabulary("data/vocab.pkl")
    openai_features_dir = None
    if args.lstm_arch == "ensemble":
        openai_features_dir = "/mnt/ds3lab/tifreaa/openai_features/"

    data_source = DataSource(
        vocab=vocab,
        labeled_data_file=args.train_file,
        test_data_file="data/test_data.txt",
        embedding_file="data/glove.twitter.27B.{0}d.txt".format(args.emb_len),
        embedding_dim=args.emb_len,
        seq_length=SEQ_LEN,
        embedding_type=args.embedding_type,
        openai_features_dir=openai_features_dir)

    print("ARCH", args.lstm_arch)

    model = VanillaLSTMModel(
        vocab=vocab,
        data_source=data_source,
        lstm_size=LSTM_SIZE,
        drop_prob=DROPOUT,
        arch=args.lstm_arch)
    print("Initialized model")

    if args.is_train:
        model.create_model()
        print("Created model. Starting training.")
        model.train(batch_size=BATCH_SIZE, loss=args.loss)
    else:
        model.create_model(args.ckpt_file)
#         print("Evaluating on validation set...")
#         model.eval()

        print("Predicting labels on test set...")
        y_test = model.predict()
        with open("data/test_output.txt", "w") as f:
            f.write("Id,Prediction\n")
            for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                f.write(str(idx+1) + "," + str(y) + "\n")
