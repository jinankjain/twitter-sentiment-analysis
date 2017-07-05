from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

OPENAI_DIRECTORY="/mnt/ds3lab/tifreaa/openai_features/"

def sgd():
    svm = SGDClassifier(loss='hinge')
    for i in range(0,150):
        print(i)
        X= np.load(OPENAI_DIRECTORY+"X"+ str(i)+".npy")
        Y = np.load(OPENAI_DIRECTORY+"Y" + str(i) + ".npy")
        shuffle_indices = np.random.permutation(np.arange(len(Y)))
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        svm.partial_fit(X,Y,classes=np.unique(Y))
    temp = []
    pred = svm.predict(X[0:10000])
    for e in pred:
        if e >= 0.5:
            temp.append(1)
        else:
            temp.append(0)
    print(accuracy_score(Y[:10000], temp))
    joblib.dump(svm, 'models/openai_baseline.pkl')

sgd()
