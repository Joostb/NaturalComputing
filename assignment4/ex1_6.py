import sklearn.datasets as datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def experiment_depth():
    depths = [1, 2, 5, 10, 100, None]

    X, y = datasets.load_digits(return_X_y=True)

    X = X / 16

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scores = np.zeros((len(depths), 1))
    for d, depth in enumerate(tqdm(depths)):
        clf = RandomForestClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        scores[d] = clf.score(X_test, y_test)

    # Plot results
    plt.figure()
    plt.plot(scores)
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    x_labels = [str(depth) if depth else "None" for depth in depths]
    plt.xticks(np.arange(0, len(scores)), x_labels)
    plt.show()


def experiment_surf(max_depth=100, max_trees=50):
    forest_sizes = np.arange(1, max_trees+1)
    depths = np.arange(1, max_depth+1)

    X, y = datasets.load_digits(return_X_y=True)

    X = X/16

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scores = np.zeros((len(depths), len(forest_sizes)))
    for d, depth in enumerate(tqdm(depths)):
        for t, trees in enumerate(forest_sizes):
            clf = RandomForestClassifier(n_estimators=trees, max_depth=depth)
            clf.fit(X_train, y_train)
            scores[d, t] = clf.score(X_test, y_test)

    # Plot results
    Fs, Ds = np.meshgrid(forest_sizes, depths)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(Ds, Fs, scores, cmap=cm.coolwarm)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Trees in Forest")
    ax.set_zlabel("Accuracy")
    plt.show()


def gridSearch():
    #parameters
    n_estimators = [1,10, 20, 30, 40]
    max_depth = [1, 2, 5, 10, 100, None]
    max_features = [None, 'sqrt', 'log2']

    random_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features}
    
    X, y = datasets.load_digits(return_X_y=True)

    X = X / 16
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
   
    #random forest
    rf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, cv=3, verbose=2)

    clf = grid_search.fit(x_train, y_train)

    param = grid_search.best_params_
    print(param)
    print(clf.score(x_test, y_test))

if __name__ == "__main__":
    experiment_surf(max_depth=20, max_trees=10)
    experiment_depth()
    gridSearch()
