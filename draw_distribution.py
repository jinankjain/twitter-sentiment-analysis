import numpy as np
import pylab as pl
import scipy.stats as stats

DATA = "data/test_data.txt"

word_lengths = []

with open(DATA, 'r') as f:
    for line in f:
        line = line.strip().split(' ')[1:]
        word_lengths.append(len(line))
word_lengths = sorted(word_lengths)
fit = stats.norm.pdf(word_lengths, np.mean(word_lengths), np.std(word_lengths))
pl.plot(word_lengths,fit,'-o')
pl.hist(word_lengths,normed=True) 
pl.show()
