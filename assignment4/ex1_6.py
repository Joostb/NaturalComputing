from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


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
    