from base_data_source import BaseDataSource
import numpy as np
import random

TRAIN_RATIO = 0.8  # the rest is used as validation set


class DataSource(BaseDataSource):
    def __init__(self, vocab, labeled_data_file, test_data_file,
                 embedding_file, embedding_dim):
        self.train = None
        self.validation = None
        self.test = None

        self.vocab = vocab
        self.embedding_dim = embedding_dim

        # Read labeled data.
        with open(labeled_data_file, "r") as f:
            content = [line.strip().split(" ", 1) for line in f.readlines()]
            labeled_data = (
                np.array([self.vocab.get_tok_ids(s[1]) for s in content]),
                np.array([int(s[0]) for s in content]))

            num_train = int(len(content) * TRAIN_RATIO)
            self.train = (
                labeled_data[0][:num_train],
                labeled_data[1][:num_train])
            self.validation = (
                labeled_data[0][num_train:],
                labeled_data[1][num_train:])
        print("Loaded training set and validation set")

        # Read test data.
        with open(test_data_file, "r") as f:
            content = [line.strip().split(",", 1)[1] for line in f.readlines()]
            self.test = np.array([self.vocab.get_tok_ids(s) for s in content])
        print("Loaded test set")

        # Read embeddings.
        embedding_dict = {}
        with open(embedding_file, "r") as f:
            content = [line.strip().split(" ") for line in f.readlines()]
            print(content[:10])
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
            if token not in embedding_dict:
                self.embedding_matrix += [
                    random.random()
                    for i in np.arange(self.embedding_dim)]
            else:
                self.embedding_matrix += [embedding_dict[token]]

        self.embedding_matrix = np.array(self.embedding_matrix)
        print("Contructed embedding matrix")

    def get_embeddings(self):
        return self.embedding_matrix

    def train(self, num_samples=None):
        if num_samples is None:
            num_samples = self.train[0].shape[0]

        return (
            self.train[0][:num_samples],
            self.train[1][:num_samples])

    def validation(self, num_samples=None):
        if num_samples is None:
            num_samples = self.validation[0].shape[0]

        return (
            self.validation[0][:num_samples],
            self.validation[1][:num_samples])

    def test(self, num_samples=None):
        if num_samples is None:
            num_samples = self.test[0].shape[0]

        return (
            self.test[0][:num_samples],
            self.test[1][:num_samples])
