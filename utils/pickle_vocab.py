#!/usr/bin/env python3
import pickle
import sys


def main(root):
    vocab = dict()
    with open(root+'/vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(root+'/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    root = sys.argv[1]
    main(root)
