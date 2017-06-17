import os

BIG_DATASET = '../data/full_train.txt'
SMALL_DATASET = '../data/small_train.txt'

TEST_PERCENTAGE = 0.1

def get_train_test_data(big_or_small):
    dataset = ""
    if big_or_small == "big":
        dataset = BIG_DATASET
    else:
        dataset = SMALL_DATASET

    data = open(dataset, 'r')
    labels = []
    tweets = []
    for line in data:
        label, tweet = line.strip().split(' ', 1)
        if (int(label) == -1):
            labels.append(0)
        else:
            labels.append(1)
        tweets.append(tweet)

    dataset_size = len(labels)
    test_size = int(TEST_PERCENTAGE*dataset_size)
    
    return ((tweets[test_size:], labels[test_size:]), (tweets[:test_size], labels[:test_size]))
    
