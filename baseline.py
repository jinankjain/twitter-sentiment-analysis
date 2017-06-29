from sklearn.linear_model import SGDClassifier
import numpy as np
import os
from sklearn.externals import joblib

OPENAI_DIRECTORY=""
GLOVE_DIRECTORY="glove.twitter.27B/"
GLOVE_FILE="glove.twitter.27B.200d.txt"

def sgd():

    svm = SGDClassifier(loss='hinge')
    for i in range(0,250):
        print(i)
        X= np.load(OPENAI_DIRECTORY+"X"+ str(i)+".npy")
        Y = np.load(OPENAI_DIRECTORY+"Y" + str(i) + ".npy")
        shuffle_indices = np.random.permutation(np.arange(len(Y)))
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        svm.partial_fit(X,Y,classes=np.unique(Y))
    joblib.dump(svm, 'openai.pkl')

def get_embeddings():
    embeddings = {}
    with open(os.path.join(GLOVE_DIRECTORY, GLOVE_FILE), 'r',encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings

def embd_to_features(sentence,embedding):
    sum=np.zeros(200)
    i=0
    for w in sentence:
        if w in embedding:
            sum=sum+embedding[w]
            i=i+1
    scale=1/(i+1)
    return [scale*e for e in sum]

def glove():
    s = get_embeddings()
    positive_examples = list(open("train_pos_full.txt", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]  # -1000
    negative_examples = list(open("train_neg_full.txt", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    x = positive_examples + negative_examples
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    X= x[shuffle_indices]
    Y = y[shuffle_indices]
    svm = SGDClassifier(loss='hinge')
    for i in range(0,250):
        X_i=[embd_to_features(e,s) for e in X[(i-1)*10000:i*10000]]
        Y_i = Y[(i - 1) * 10000:i * 10000]
        svm.partial_fit(X_i,Y_i,classes=np.unique(Y_i))
    joblib.dump(svm, 'glove.pkl')

sgd()
glove()

