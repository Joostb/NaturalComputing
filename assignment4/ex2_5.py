import sklearn.datasets as datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def experiment_depth():
    n_estimators = [10, 50, 100, 150]
    lrs = np.arange(0.01, 10, 0.1)
    X, y = datasets.load_digits(return_X_y=True)

    X = X / 16

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    scores = np.zeros((len(n_estimators), len(lrs)))
    for i, estimator in enumerate(n_estimators):
        for d, lr in enumerate(tqdm(lrs)):
            base_estimator = DecisionTreeClassifier(max_depth=3, max_features='log2')
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=estimator, learning_rate=lr)
            clf.fit(X_train, y_train)
            scores[i][d] = clf.score(X_test, y_test)

    # Plot results
    plt.figure()
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    for i in range(len(n_estimators)):
        plt.plot(lrs, scores[i], label="n_estimator: {}".format(n_estimators[i]))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_depth()
