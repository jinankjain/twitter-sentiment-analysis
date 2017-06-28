from encoder import Model
import numpy as np
import pickle
# from sklearn.decomposition import IncrementalPCA

test_data = list(open("../data/twitter-datasets/test_data.txt", "r",encoding='utf8').readlines())
print(np.shape(test_data))
model=Model()
# ipca = IncrementalPCA(n_components=500)

x = [s.strip() for s in test_data]
x_text = [sent for sent in x]
print(np.shape(x_text))
x = model.transform(x_text)
np.save("/mnt/ds3lab/tifreaa/openai_features/test_X.npy",x)
print(np.shape(x))
#     ipca.partial_fit(x)

# pickle.dump(ipca, open("pca", 'wb'))
