import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

OPENAI_DIR = "/mnt/ds3lab/tifreaa/openai_features/"

def load_data(first_batch, num_batches):
    X = np.load(OPENAI_DIR+"X"+str(first_batch)+".npy")
    y = np.load(OPENAI_DIR+"Y"+str(first_batch)+".npy")
    i = 1
    while i < num_batches:
        curr_id = first_batch + i
        X_batch = np.load(OPENAI_DIR+"X"+str(curr_id)+".npy")
        y_batch = np.load(OPENAI_DIR+"Y"+str(curr_id)+".npy")
        X = np.concatenate((X, X_batch))
        y = np.concatenate((y, y_batch))
        i += 1

    # Shuffle data.
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    return X, y

if __name__ == "__main__":
    X, y = load_data(0, 50)
    print("Loaded training data", X.shape, y.shape)
    val_X, val_y = load_data(249, 1)
    print("Loaded validation data", val_X.shape, val_y.shape)

#     pca = PCA(n_components=256, whiten=True, svd_solver="full")
#     pca.fit(X[:10000])
#     print("Finished PCA")
#
#     with open("models/openai_PCA_model.pkl", "wb") as f:
#         pickle.dump(pca, f)

    pca = None
    with open("models/openai_PCA_model.pkl", "rb") as f:
        pca = pickle.load(f)

#     pca_X = pca.transform(X)
#     print(pca_X.shape)
#     svm = SVC(kernel="rbf")
#     svm.fit(pca_X, y)
#     print("Finished training SVM")
#
#     with open("models/openai_SVM_model.pkl", "wb") as f:
#         pickle.dump(svm, f)
#
#     pca_val_X = pca.transform(val_X)
#     print(pca_val_X.shape)
#     acc = svm.score(pca_val_X, val_y)
#     print("Validation accuracy:", acc)

    pca_X = pca.transform(X)
    print(pca_X.shape)
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(pca_X, y)
    print("Finished training Random Forest")

    with open("models/openai_RF_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    pca_val_X = pca.transform(val_X)
    print(pca_val_X.shape)
    acc = rf.score(pca_val_X, val_y)
    print("Validation accuracy:", acc)
