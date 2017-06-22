from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

import numpy as np

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

        # Create checkpoint folder if not present
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        # Create the embedding layer and load pretrained embeddings.
        embedding_matrix = self.data_source.get_embeddings()
        self.embedding_layer = Embedding(
            self.vocab.vocab_size,
            self.data_source.embedding_dim,
            weights=[embedding_matrix],
            input_length=seq_length,
            trainable=True)

        self.model = None
        self.arch = arch

    """
    This method should create a keras model.
    """
    def create_model(self, ckpt_file=None):
        raise NotImplementedError("Please implement this method")

    def train(self, batch_size):
        X_val, y_val = self.data_source.validation()

        y_val = to_categorical((y_val + 1) / 2, num_classes=2)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        print(self.model.summary())

        checkpoint = ModelCheckpoint(
            filepath=CKPT_DIR+self.arch+'_lstm_ckpt-{epoch:02d}-{val_loss:.2f}.hdf5')
        while True:
            curr_X_train, curr_y_train = self.data_source.next_train_batch(
                STEPS_PER_CKPT)
            curr_y_train = to_categorical((curr_y_train + 1) / 2, num_classes=2)
            self.model.fit(
                curr_X_train,
                curr_y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=1,
                callbacks=[checkpoint])

    """
    Evaluate the model on the validation set.
    """
    def eval(self):
        X, y = self.data_source.validation()
        y = to_categorical((y + 1) / 2, num_classes=2)

        scores = self.model.evaluate(X, y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    """
    Get the predictions of the model on a test set.
    """
    def predict(self):
        X = self.data_source.test()
        y_predict = np.argmax(self.model.predict(X), axis=1)
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
