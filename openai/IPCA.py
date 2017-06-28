from encoder import Model
import numpy as np
import pickle
# from sklearn.decomposition import IncrementalPCA

positive= list(open("../data/twitter-datasets/train_pos_full.txt", "r",encoding='utf8').readlines())
negative = list(open("../data/twitter-datasets/train_neg_full.txt", "r",encoding='utf8').readlines())
print(np.shape(positive))
print(np.shape(negative))
model=Model()
# ipca = IncrementalPCA(n_components=500)

for i in range(73,250):
    print(i)

    positive_examples = [s.strip() for s in positive[(i) * 5000:(i + 1) * 5000]]  # -1000

    negative_examples = [s.strip() for s in negative[(i) * 5000:(i + 1) * 5000]]

    x = positive_examples + negative_examples

    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)

    x_text = [sent for sent in x]
    print(np.shape(x_text))
    print(np.shape(y))
    x = model.transform(x_text)
    np.save("/mnt/ds3lab/tifreaa/openai_features/X"+ str(i)+".npy",x)
    np.save("/mnt/ds3lab/tifreaa/openai_features/Y" + str(i) + ".npy", y)
    print(np.shape(x))
#     ipca.partial_fit(x)

# pickle.dump(ipca, open("pca", 'wb'))
