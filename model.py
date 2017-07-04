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


class Model(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN, arch=None):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length, arch)
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 256

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

                self.model.add(Dense(512, activation='softmax'))
                self.model.add(Dropout(2*DROPOUT))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "seq_conv1":
                self.model = Sequential()
                filter_size = 3
                self.model.add(self.embedding_layer)
                self.model.add(Conv1D(filters=self.num_filters/2, kernel_size=filter_size,
                                        strides=1, padding='valid', activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))

                self.model.add(Conv1D(filters=int(self.num_filters/4), kernel_size=filter_size,
                                        strides=1, padding='valid', activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))

                self.model.add(Flatten())
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "seq_conv2":
                self.model = Sequential()
                filter_size = 3
                self.model.add(self.embedding_layer)
                self.model.add(Conv1D(filters=self.num_filters, kernel_size=filter_size,
                                        strides=1, padding='same', activation='relu'))
                self.model.add(Conv1D(filters=int(self.num_filters/2), kernel_size=filter_size,
                                        strides=1, padding='same', activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))

                self.model.add(Conv1D(filters=int(self.num_filters/4), kernel_size=filter_size,
                                        strides=1, padding='same', activation='relu'))
                self.model.add(Conv1D(filters=int(self.num_filters/4), kernel_size=filter_size,
                                        strides=1, padding='same', activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))

                self.model.add(Flatten())
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(512, activation='relu'))
                self.model.add(Dropout(2*DROPOUT))
                self.model.add(Dense(2, activation='softmax'))

            elif self.arch == "conv_lstm":
                # Ensemble of sentence-level embedding obtained from an LSTM and
                # a sentence-level embedding obtained from a CNN.
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
                # Ensemble of sentence-level embedding obtained from an LSTM and
                # a sentence-level embedding obtained from the OpenAI features.
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
    parser.add_argument('--seq_conv1', dest='lstm_arch',
                        action='store_const', const='seq_conv1')
    parser.add_argument('--seq_conv2', dest='lstm_arch',
                        action='store_const', const='seq_conv2')
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

    model = Model(
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
    elif args.is_eval:
        model.create_model(args.ckpt_file)

        print("Predicting labels on test set...")
        y_test, y_probs = model.predict()
        with open("data/" + args.lstm_arch + "_test_output.txt", "w") as f:
            f.write("Id,Prediction\n")
            for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                f.write(str(idx+1) + "," + str(y) + "\n")
    else:
        model.create_model(args.ckpt_file)

        print("Predicting labels on test set...")
        y_test, y_probs = model.predict()
        with open("data/" + args.lstm_arch + "_test_output.txt", "w") as f:
            f.write("Id,Prediction\n")
            for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                f.write(str(idx+1) + "," + str(y) + "\n")
        with open("data/" + args.lstm_arch + "_test_probs.txt", "w") as f:
            f.write("Id,Probabilities\n")
            for idx, y in zip(np.arange(y_probs.shape[0]), y_probs):
                f.write(str(idx+1) + "," + str(y[1]) + "\n")

        print("Predicting labels on validation set...")
        y_val, y_probs = model.predict(data_source.validation()[0])
        with open("data/" + args.lstm_arch + "_validation_probs.txt", "w") as f:
            f.write("Id,Probabilities,Predicted label,True label\n")
            for idx, probs, pred_y, true_y in zip( np.arange(y_probs.shape[0]), y_probs,
                    y_val, data_source.validation()[1]):
                f.write(str(idx+1) + "," + str(probs[1]) + "," + str(pred_y) +
                        "," + str(true_y) + "\n")
