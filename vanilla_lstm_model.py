import argparse
import h5py
from base_model import BaseModel
from data_source import DataSource
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
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
BATCH_SIZE = 64


class VanillaLSTMModel(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN, arch=None):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length, arch)

    def create_model(self, ckpt_file=None):
        if ckpt_file is None:
            if self.arch == "vanilla":
                self.model = Sequential()
                self.model.add(self.embedding_layer)
                self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))

                self.model.add(Dense(2, activation='softmax'))
            elif self.arch == "ensamble":
                print("Start")
                temp_model = Sequential()
                temp_model.add(self.embedding_layer)
                temp_model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                                    recurrent_dropout=self.drop_prob,
                                    implementation=2, unroll=True))

                first_input = Input(shape=(self.embedding_layer.input_dim,))
                print("GATA PANA AICI", temp_model)
                first_model = temp_model(first_input)
                print("Created first model")

                second_input = Input(shape=(OPENAI_FEATURE_SIZE,))
                openai_model = Sequential()
                openai_model.add(Embedding(
                    input_dim=OPENAI_FEATURE_SIZE,
                    output_dim=self.embedding_layer.output_dim,
                    input_length=OPENAI_FEATURE_SIZE,
                    trainable=True))
                second_model = openai_model(second_input)
                print("Created second model")

                result = Sequential()
                # Add 3 fully-connected layers.
                input_len = LSTM_SIZE + self.embedding_layer.output_dim
                result.add(Dense(2048, input_shape=(input_len,), activation='relu'))
                result.add(Dropout(2*DROPOUT))
                result.add(Dense(1024, activation='relu'))
                result.add(Dropout(2*DROPOUT))
                result.add(Dense(512, activation='relu'))
                result.add(Dropout(2*DROPOUT))

                result.add(Dense(2, activation='softmax'))
                final = result(concatenate([first_model, second_model]))
                result = Model(
                    inputs=[first_input, second_input],
                    outputs=[final])
                self.model = result
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
    parser.add_argument('--bidi', dest='lstm_arch',
                        action='store_const', const='bidi')
    parser.add_argument('--multi_layer', dest='lstm_arch',
                        action='store_const', const='multi_layer')
    parser.add_argument('--ensamble', dest='lstm_arch',
                        action='store_const', const='ensamble')
    parser.add_argument('--embedding_type', type=str, default="glove", nargs="?",
                        help='Type of word embedding Word2Vec or Glove')
    args = parser.parse_args()

    vocab = Vocabulary("data/vocab.pkl")
    openai_features_dir = None
    if args.lstm_arch == "ensamble":
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
