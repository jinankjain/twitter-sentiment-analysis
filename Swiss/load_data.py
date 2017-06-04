#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
from tensorflow.contrib import learn
import random

vocab_processor = None

def get_train_data(LIM=1000):

        # Load data from files

        positive_examples = list(open("../twitter-datasets/train_pos.txt", "r").readlines())
        positive_examples = [s.strip() for s in positive_examples[:LIM]]   # -1000
        negative_examples = list(open("../twitter-datasets/train_neg.txt", "r").readlines())
        negative_examples = [s.strip() for s in negative_examples[:LIM]]

        x_text = positive_examples + negative_examples

        x_text = [sent for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        # Build vocabulary
        max_length_of_sentence = max([len(x.split()) for x in x_text])
        global vocab_processor
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length_of_sentence)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(1234)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/cross-validation set
        cross_validation_indices = np.array(random.sample(list(np.arange(len(y))), int(len(y) * 0.1) ))
        train_indices = np.array(list(set(np.arange(len(y))) - set(cross_validation_indices)))

        x_train, x_dev = x_shuffled[train_indices], x_shuffled[cross_validation_indices]
        y_train, y_dev = y_shuffled[train_indices], y_shuffled[cross_validation_indices]

        return [x_train, x_dev, y_train, y_dev, vocab_processor]

def get_submission_data():
        # Load data from files

        submit_examples = list(open("../twitter-datasets/test_data.txt", "r").readlines())
        submit_examples = [s.strip() for s in submit_examples]  
        splitter = [s.split(',', 1) for s in submit_examples]
        sentences = [s[1] for s in splitter]
        ids = [s[0] for s in splitter]

        x_submit = np.array(list(vocab_processor.transform(sentences)))

        return [x_submit, ids]


def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


