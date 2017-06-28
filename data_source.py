from base_data_source import BaseDataSource
import numpy as np
import random
import gensim

# TRAIN_RATIO = 0.9  # the rest is used as validation set
VALIDATION_SIZE = 10000
OPENAI_SAMPLES_PER_BATCH = 10000


class DataSource(BaseDataSource):
    def __init__(self, vocab, labeled_data_file, test_data_file,
                 embedding_file, embedding_dim, seq_length, embedding_type,
                 openai_features_dir=None):
        self._train = None
        self._validation = None
        self._test = None

        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        self.openai_features_dir = openai_features_dir
        self.curr_openai_batch = None

        self.start = 0

        # Read labeled data.
        with open(labeled_data_file, "r") as f:
            content = [line.strip().split(" ", 1) for line in f.readlines()]

            labeled_data = (
                np.array([self.vocab.get_tok_ids(s[1], self.seq_length)
                          for s in content]),
                np.array([int(s[0]) for s in content]))

            num_train = len(content) - VALIDATION_SIZE
            self._train = (
                labeled_data[0][:num_train],
                labeled_data[1][:num_train])
            for i in np.arange(5):
                print("DA", len(self._train[0][i]))
            self._validation = (
                labeled_data[0][num_train:],
                labeled_data[1][num_train:])
            if self.openai_features_dir is not None:
                # Load last batch.
                validation_openai_features = np.load(
                        self.openai_features_dir+"X249.npy")
                self._validation = (
                    labeled_data[0][num_train:],
                    labeled_data[1][num_train:],
                    validation_openai_features)
        print("Loaded training set and validation set")

        # Read test data.
        with open(test_data_file, "r") as f:
            content = [line.strip().split(",", 1)[1] for line in f.readlines()]
            self._test = np.array([self.vocab.get_tok_ids(s, self.seq_length)
                                   for s in content])

            if self.openai_features_dir is not None:
                # Load test batch.
                test_openai_features = np.load(
                        self.openai_features_dir+"test_X.npy")
                self._test = (self._test, test_openai_features)
        print("Loaded test set")

        if embedding_type == "glove":
            # Read embeddings.
            embedding_dict = {}
            with open(embedding_file, "r") as f:
                content = [line.strip().split(" ") for line in f.readlines()]
                embedding_dict = dict(
                    (l[0], [float(x) for x in l[1:]]) for l in content)
            print("Loaded embeddings")

            # Construct the embedding matrix. Row i has the embedding for the token
            # with tokID i.
            self.embedding_matrix = []
            for i in np.arange(self.vocab.vocab_size):
                token = self.vocab.sorted_vocab[i]

                # TODO: Figure out what to do with "<unk>". Currently not in the
                # pretrained embeddings.

                # TODO: Maybe figure out what to do with tokens that appear in our
                # vocabulary, but don't appear in the word embeddings.
                # XXX: currently we insert a random embedding for words that are not
                # in the embedding matrix, but what we could do is ignore these
                # words when we create the vocabulary.
                if token not in embedding_dict:
                    self.embedding_matrix.append([
                        random.random()
                        for i in np.arange(self.embedding_dim)])
                else:
                    self.embedding_matrix.append(embedding_dict[token])

            self.embedding_matrix = np.array(self.embedding_matrix)
        elif embedding_type == "word2vec":
            print("Loading word2vec")
            # Read embeddings.
            model =  gensim.models.KeyedVectors.load_word2vec_format('data/word2vec.bin', binary=True, unicode_errors='ignore')
            # Construct the embedding matrix. Row i has the embedding for the token
            # with tokID i.
            self.embedding_matrix = []
            for i in np.arange(self.vocab.vocab_size):
                token = self.vocab.sorted_vocab[i]

                # TODO: Figure out what to do with "<unk>". Currently not in the
                # pretrained embeddings.

                # TODO: Maybe figure out what to do with tokens that appear in our
                # vocabulary, but don't appear in the word embeddings.
                # XXX: currently we insert a random embedding for words that are not
                # in the embedding matrix, but what we could do is ignore these
                # words when we create the vocabulary.
                if token not in model:
                    self.embedding_matrix.append([
                        random.random()
                        for i in np.arange(self.embedding_dim)])
                else:
                    self.embedding_matrix.append(model[token])

            self.embedding_matrix = np.array(self.embedding_matrix)

        print("Contructed embedding matrix")

    def get_embeddings(self):
        return self.embedding_matrix

    def train(self, num_samples=None):
        if num_samples is None:
            num_samples = self._train[0].shape[0]

        return (
            self._train[0][:num_samples],
            self._train[1][:num_samples])

    def validation(self, num_samples=None):
        if num_samples is None:
            return self._validation

        return (
            self._validation[0][:num_samples],
            self._validation[1][:num_samples])

    def test(self, num_samples=None):
        if num_samples is None:
            return self._test

        return self._test[:num_samples]

    def next_train_batch(self, num_samples=None, with_openai_features=False):
        num_train = self._train[0].shape[0]

        openai_features = None
        if with_openai_features:
            if self.curr_openai_batch is None:
                # Load first batch.
                self.curr_openai_batch = np.load(
                        self.openai_features_dir+"X0.npy")

            openai_features = self.curr_openai_batch[self.start:self.start+num_samples]
            if openai_features.shape[0] < num_samples:
                # Load next OpenAI batch (i.e. 10000 samples).
                next_start = (self.start + num_samples) % num_train
                next_batch_id = next_start / OPENAI_SAMPLES_PER_BATCH
                self.curr_openai_batch = np.load(
                        self.openai_features_dir+"X"+next_batch_id+".npy")

                # Load the rest of the samples.
                num_rest = (self.start + num_samples) % OPENAI_SAMPLES_PER_BATCH
                openai_features = np.concatenate((
                    openai_features, self.curr_openai_batch[:num_rest]))

        X = self._train[0][self.start:self.start+num_samples]
        y = self._train[1][self.start:self.start+num_samples]

        self.start = (self.start + num_samples) % num_train

        if X.shape[0] < num_samples:
            X = np.concatenate((X, self._train[0][:self.start]))
            y = np.concatenate((y, self._train[1][:self.start]))

        if openai_features is None:
            return (X, y)
        else:
            return (X, y, openai_features)
