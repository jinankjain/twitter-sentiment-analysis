from base_model import BaseModel
from data_source import DataSource
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from vocabulary import Vocabulary

DROPOUT = 0.2
LSTM_SIZE = 1024

NUM_EPOCHS = 10
BATCH_SIZE = 64


class VanillaLSTMModel(BaseModel):
    def __init__(self, vocab, data_source, lstm_size=LSTM_SIZE,
                 drop_prob=DROPOUT):
        BaseModel.__init__(self, vocab, data_source, lstm_size, drop_prob)

    def create_model(self):
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(LSTM(LSTM_SIZE, dropout=self.drop_prob,
                            recurrent_dropout=self.drop_out, unroll=True))
        self.model.add(Dense(1, activation='sigmoid'))

if __name__ == "__main__":
    vocab = Vocabulary("data/vocab.pkl")
    data_source = DataSource(
        vocab=vocab,
        labeled_data_file="data/small_train.txt",
        test_data_file="data/test_data.txt",
        embedding_file="data/glove.twitter.27B.25d.txt",
        embedding_dim=25)
    model = VanillaLSTMModel(vocab, data_source, LSTM_SIZE, DROPOUT)
#     model.train()
