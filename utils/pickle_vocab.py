#!/usr/bin/env python3
import pickle
import sys


def main(root):
    vocab = {"<unk>": 0}
    with open(root+'/vocab_trimmed.txt') as f:
        for idx, line in enumerate(f):
            tok = line.split()[1]
            vocab[tok] = idx + 1

    with open(root+'/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    root = sys.argv[1]
    main(root)
