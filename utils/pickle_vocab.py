#!/usr/bin/env python3
import pickle
import sys


def main(root):
    vocab = []
    with open(root+'/vocab_trimmed.txt') as f:
        vocab = [line.strip().split()[1] for line in f.readlines()]
        vocab = ["<unk>"] + vocab

    with open(root+'/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == '__main__':
    root = sys.argv[1]
    main(root)
