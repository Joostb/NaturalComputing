import sklearn.datasets as datasets
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def experiment_depth():
    n_estimators = [10, 50, 100, 150]
    depths = [1,2,3,5,10]
    lrs = np.arange(0.01, 10, 0.1)
    X, y = datasets.load_digits(return_X_y=True)

    X = X / 16

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    scores = np.zeros((len(n_estimators), len(lrs)))
    # scores = np.zeros((len(lrs)))
    for i, estimator in enumerate(n_estimators):
        for d, lr in enumerate(tqdm(lrs)):
            base_estimator=DecisionTreeClassifier(max_depth=3, max_features='log2')
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=estimator, learning_rate=lr )
            clf.fit(X_train, y_train)
            scores[i][d] = clf.score(X_test, y_test)

    print(scores)
    # Plot results
    plt.figure()
    # plt.plot(scores)
    plt.xlabel("learning rate")
    plt.ylabel("Accuracy")
    # x_labels = [str(n_estimator) if n_estimator else "None" for n_estimator in n_estimators]

    for i in range(len(n_estimators)):
        plt.plot(lrs, scores[i], label="n_estimator: {}".format(n_estimators[i]))
        # plt.xticks(np.arange(0, len(scores[i])), x_labels)
    plt.legend()
    plt.show()


def experiment_surf(max_depth=100, max_trees=50):
    
    X,y = datasets.load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


    n_estimators = np.arange(1, 20, 2)
    learning_rates = np.arange(0.01, 10, 0.1)
    scores = np.zeros((len(n_estimators), len(learning_rates)))

    for d, n_estimator in enumerate(tqdm(n_estimators)):
        for t, lr in enumerate(learning_rates):

            base_estimator=DecisionTreeClassifier(max_depth=10, max_features='log2')
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimator, learning_rate=lr )
            clf.fit(X_train, y_train)
            scores[d, t] = clf.score(X_test, y_test)

    
    # Plot results
    Fs, Ds = np.meshgrid(n_estimators, learning_rates)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(Ds, Fs, scores, cmap=cm.coolwarm)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Trees in Forest")
    ax.set_zlabel("Accuracy")
    plt.show()

    plt.hist(scores.flatten())
    plt.show()


def joost():
	#parameters
	n_estimators = [10, 50, 100, 350, 500]
	max_depth = [20, 50, 70, 90]
	max_features = ['auto', 'sqrt', 'log2']

	random_grid = {'n_estimators': n_estimators,
				   'max_depth': max_depth,
				   'max_features': max_features
				   }

	#data
	iris = load_iris()

	X = iris.data
	y = iris.target

	x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

	#random forest
	rf = RandomForestClassifier()

	grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, cv=3, verbose=2)

	clf = grid_search.fit(x_train, y_train)

	param = grid_search.best_params_
	print(param)
	print(clf.score(x_test, y_test))

if __name__ == "__main__":
    # experiment_surf(max_depth=20, max_trees=10)
    # experiment_surf()
    experiment_depth()
	# joost()
