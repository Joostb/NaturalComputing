from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


#parameters
n_estimators = [10, 30, 60, 100]
max_depth = [2, 4, 10, 30]

#data
iris = load_iris()

X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

scores = []
for i,e in enumerate(n_estimators):
    for j, d in enumerate(max_depth):
        rf = RandomForestClassifier(n_estimators=e, max_depth=d)
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        
        X_train = X[idx]
        y_train = y[idx]
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        
        clf = rf.fit(X_train, y_train)
        
        scores.append(clf.score(X_train, y_train))
        
print(scores)

