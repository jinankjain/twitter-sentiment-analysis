import re
from tqdm import tqdm
import os

"""
Reads the csv dataset available at
http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
and splits it into two files (.pos and .neg) containing the positive and
negative tweets.
Does some word preprocessing during the parsing.
"""

# import pandas as pd
# df = pd.read_csv('twitter-sentiment-dataset/sentiment-dataset.csv',
#                  error_bad_lines=False)
#
# df.SentimentText = df.SentimentText.str.strip()
# df.SentimentText = df.SentimentText.str.replace(r'http://.*', '<link/>')
# df.SentimentText = df.SentimentText.str.replace('#', '<HASHTAG/> ')
# df.SentimentText = df.SentimentText.str.replace('&quot;', ' \" ')
# df.SentimentText = df.SentimentText.str.replace('&amp;', ' & ')
# df.SentimentText = df.SentimentText.str.replace('&gt;', ' > ')
# df.SentimentText = df.SentimentText.str.replace('&lt;', ' < ')

DATASET_FOLDER = 'twitter-sentiment-dataset/'
if not os.path.exists(DATASET_FOLDER):
    os.mkdir(DATASET_FOLDER)

try:
    full_dataset = open("../twitter-sentiment-analysis/data/small_train.txt", "r")
    pos_dataset = open("twitter-sentiment-dataset/tw-data.pos", "w")
    neg_dataset = open("twitter-sentiment-dataset/tw-data.neg", "w")
except IOError:
    print "Failed to open file"
    quit()

csv_lines = full_dataset.readlines()
i = 0.0

for line in tqdm(csv_lines):
    i += 1.0
    line = line.split(" ", 1)
    tweet = line[1].strip()
    new_tweet = ''

    for word in tweet.split():
        # String preprocessing
        word = word.replace('#', '<HASHTAG/> ')
        word = word.replace('&quot;', ' \" ')
        word = word.replace('&amp;', ' & ')
        new_tweet = ' '.join([new_tweet, word])

    tweet = new_tweet.strip() + '\n'

    if line[0].strip() == '1':
        pos_dataset.write(tweet)
    else:
        neg_dataset.write(tweet)
