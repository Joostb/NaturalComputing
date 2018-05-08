import numpy as np
import scipy.io as sio
import scipy as sp
from sklearn.svm import SVC
from tqdm import tqdm


def transform_data(source_path, target_path):
    mat_source = sio.loadmat(source_path, squeeze_me=True)

    X_source = mat_source['x']
    y_source = mat_source['y']

    mat_target = sio.loadmat(target_path, squeeze_me=True)

    X_target = mat_target['x']
    y_target = mat_target['y']

    Cs = np.cov(X_source.T) + np.identity(X_source.shape[1])
    Ct = np.cov(X_target.T) + np.identity(X_target.shape[1])

    Ds = np.matmul(X_source, sp.linalg.fractional_matrix_power(Cs, -0.5))
    np.save('Ds.npy', Ds)
    print("Saved Ds")

    _Ds = np.matmul(Ds, sp.linalg.fractional_matrix_power(Ct, 0.5))
    np.save('_Ds.npy', _Ds)
    print("Saved _DS")

    return _Ds, y_source, X_target, y_target


def load_data(source_path, target_path):
    mat_source = sio.loadmat(source_path, squeeze_me=True)
    # X_source = np.load("_Ds.npy")
    X_source = mat_source['x']
    y_source = mat_source['y']

    mat_target = sio.loadmat(target_path, squeeze_me=True)

    X_target = mat_target['x']
    y_target = mat_target['y']

    print("Loaded Data")
    return X_source.real, y_source, X_target, y_target


def train_and_evaluate_classifier(X_source, y_source, X_target, y_target):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    best_model = None
    best_acc = 0

    X_source = (X_source - np.min(X_source)) / (np.max(X_source) - np.min(X_source))
    X_target = (X_target - np.min(X_target)) / (np.max(X_target) - np.min(X_target))

    for C in tqdm(Cs):
        clf = SVC(C=C)
        clf.fit(X_source.real, y_source)
        acc = clf.score(X_target, y_target)
        tqdm.write("C: {:-6f}, \t acc: {}".format(C, acc))
        if acc > best_acc:
            best_model = clf
            best_acc = acc

    preds = best_model.predict(X_target)
    np.save("preds.npy", preds)
    return best_model, preds


def main():
    source_path = "../data/office-caltech/office-caltech-vgg-sumpool-amazon-fc6.mat"
    target_path = "../data/office-caltech/office-caltech-vgg-sumpool-Caltech-fc6.mat"

    # X_source, y_source, X_target, y_target = transform_data(source_path, target_path)
    X_source, y_source, X_target, y_target = load_data(source_path, target_path)
    train_and_evaluate_classifier(X_source, y_source, X_target, y_target)


if __name__ == "__main__":
    main()
