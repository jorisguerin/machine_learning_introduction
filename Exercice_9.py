# -*- coding: utf-8 -*-
import numpy as np
import sklearn.datasets as ds
import sklearn.svm as svm

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

### Linéaire ###
c = 100000
test_err = []
for i in range(10):
    algo = svm.LinearSVC(C = c)
    algo.fit(features_iris, target_iris)
    test_err.append(np.sum(algo.predict(features_iris) != target_iris))

print('linear : ', np.mean(np.array(test_err), axis = 0))

### RBF kernel ###
c = 1000
test_err = []
for i in range(10):
    algo = svm.SVC(C = c, kernel = 'rbf')
    algo.fit(features_iris, target_iris)
    test_err.append(np.sum(algo.predict(features_iris) != target_iris))

print('rbf : ', np.mean(np.array(test_err), axis = 0))

### polynomial kernel ###
c = 1000
test_err = []
for i in range(10):
    algo = svm.SVC(C = c, kernel = 'poly', degree = 3)
    algo.fit(features_iris, target_iris)
    test_err.append(np.sum(algo.predict(features_iris) != target_iris))

print('poly : ', np.mean(np.array(test_err), axis = 0))

