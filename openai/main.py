from encoder import Model
import numpy as np
import random
from sklearn import svm
from sklearn.metrics import accuracy_score

model = Model()

positive_examples = list(open("train_pos_full.txt", "r").readlines())
positive_examples = [s.strip() for s in positive_examples]   # -1000
negative_examples = list(open("train_neg_full.txt", "r").readlines())
negative_examples = [s.strip() for s in negative_examples]

x = positive_examples + negative_examples

x_text = [sent for sent in x]

positive_labels = [1 for _ in positive_examples]
negative_labels = [0 for _ in negative_examples]

y = np.concatenate([positive_labels, negative_labels], 0)
x= model.transform(x_text)

shuffle_indices = np.random.permutation(np.arange(len(y)))

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

np.save('X.npy', x_shuffled )
np.save('Y.npy', y_shuffled )

cross_validation_indices = np.array(random.sample(list(np.arange(len(y))), int(len(y) * 0.1) ))
train_indices = np.array(list(set(np.arange(len(y))) - set(cross_validation_indices)))

x_train, x_test= x_shuffled[train_indices], x_shuffled[cross_validation_indices]
y_train, y_test = y_shuffled[train_indices], y_shuffled[cross_validation_indices]

moodel = svm.SVC(kernel='rbf')
moodel.fit(x_train, y_train)

score = accuracy_score(y_test,moodel.predict(x_test))
print('Test accuracy:', score)

submit_examples = list(open("test_data.txt", "r").readlines())
submit_examples = [s.strip() for s in submit_examples]
splitter = [s.split(',', 1) for s in submit_examples]
sentences = [s[1] for s in splitter]
ids = [s[0] for s in splitter]
predictions=moodel.predict(model.transform(sentences))
predictions = [-1 if p == 0 else 1 for p in predictions]
with open("predict" + ".csv", "w") as f:
    f.write("Id,Prediction\n")
    for i, p in enumerate(predictions):
        f.write(str(ids[i]) + "," + str(p) + "\n")