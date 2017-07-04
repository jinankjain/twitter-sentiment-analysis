import numpy as np
import pickle

UNK_ID = 0
PAD_ID = 1


class Vocabulary:
    def __init__(self, filename):
        self.sorted_vocab = []  # on position i is the token with tokID i
        self.vocab = {}  # maps tokens to tokIDs
        with open(filename, "rb") as f:
            self.sorted_vocab = pickle.load(f)

        self.vocab_size = len(self.sorted_vocab)
        self.vocab = dict(zip(self.sorted_vocab, np.arange(self.vocab_size)))

        self.one_hot = {}
        for token, idx in self.vocab.items():
            enc = np.zeros(self.vocab_size)
            enc[idx] = 1
            self.one_hot[token] = enc

    """
    Gets a tweet (a string) and returns a numpy array of indexes. If a token is
    not in the vocabulary, its ID is considered to be 0.
    """
    def get_tok_ids(self, text, seq_length):
        tokens = text.split()
        pad_count = seq_length - len(tokens)

        # XXX: currently, if a sequence is longer than seq_length, we trim it to
        # seq_length. However, we might want to consider discarding it
        # altogether.
        if len(tokens) > seq_length:
            tokens = tokens[:seq_length]

        return np.array([
            self.vocab[tok] if tok in self.vocab else UNK_ID
            for tok in tokens] + [PAD_ID] * pad_count)

    def get_tf_encoding(self, text):
        # Get token IDs.
        tok_ids = self.get_tok_ids(text)

        # Get term frequencies.
        tf = np.bincount(tok_ids)

        # Pad encoding vector to vocab_size.
        enc = np.lib.pad(tf, (0, self.vocab_size-tf.shape[0]), 'constant',
                         constant_values=0)
        return enc


if __name__ == "__main__":
    v = Vocabulary("../data/vocab.pkl")
    enc = v.get_tf_encoding("something about you and me is crrrazy")
    print(np.nonzero(enc)[0])
    print(sum(enc))
