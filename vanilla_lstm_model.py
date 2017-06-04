from base_model import BaseModel
from data_source import DataSource
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from utils.vocabulary import Vocabulary

DROPOUT = 0.2
LSTM_SIZE = 1024
SEQ_LEN = 40

NUM_EPOCHS = 10
BATCH_SIZE = 64


class VanillaLSTMModel(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT, seq_length=SEQ_LEN):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob,
                           seq_length)

    def create_model(self):
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                            recurrent_dropout=self.drop_prob,
                            unroll=True))
        self.model.add(Dense(1, activation='sigmoid'))

if __name__ == "__main__":
    vocab = Vocabulary("data/vocab.pkl")
    data_source = DataSource(
        vocab=vocab,
        labeled_data_file="data/small_train.txt",
        test_data_file="data/test_data.txt",
        embedding_file="data/glove.twitter.27B.25d.txt",
        embedding_dim=25,
        seq_length=SEQ_LEN)
    model = VanillaLSTMModel(vocab, data_source, LSTM_SIZE, DROPOUT)
    model.create_model()
    model.train(NUM_EPOCHS, BATCH_SIZE)
