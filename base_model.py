from keras.layers import Embedding
# from keras.layers.embeddings import Embedding
from keras.models import model_from_json

POS = 1
NEG = -1


class BaseModel:
    def __init__(self, vocab, data_source, lstm_size, drop_prob, seq_length):
        self.vocab = vocab
        self.data_source = data_source

        self.drop_prob = drop_prob

        # Create the embedding layer and load pretrained embeddings.
        embedding_matrix = self.data_source.get_embeddings()
        self.embedding_layer = Embedding(
            self.vocab.vocab_size,
            self.data_source.embedding_dim,
            weights=[embedding_matrix],
            input_length=seq_length,
            trainable=True)

        self.model = None

    """
    This method should create a keras model.
    """
    def create_model(self):
        raise NotImplementedError("Please implement this method")

    def train(self, num_epochs, batch_size):
        X_train, y_train = self.data_source.train()

        self.model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

        print(X_train.shape)
        print(y_train.shape)
        self.model.fit(
            X_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1)

        # TODO: save the model at the end of training

    """
    Evaluate the model on the validation set.
    """
    def eval(self):
        X, y = self.data_source.validation()
        scores = self.model.evaluate(X, y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    """
    Get the predictions of the model on a test set.
    """
    def predict(self):
        X = self.data_source.test()
        y_predict = self.model.predict(X)
        print(y_predict)
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
