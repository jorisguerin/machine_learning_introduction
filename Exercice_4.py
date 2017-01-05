# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets as ds
import sklearn.neighbors as knn

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

N = len(features_iris)
N_train = int(0.6 * N)
N_valid = int(0.2 * N)
N_test = int(0.2 * N)

randomized_indices = np.arange(len(features_iris))
np.random.shuffle(randomized_indices)

shuffled_features_iris = features_iris[randomized_indices]
shuffled_target_iris = target_iris[randomized_indices]

training_features = shuffled_features_iris[:N_train]
validation_features = shuffled_features_iris[N_train:N_train + N_valid]
test_features = shuffled_features_iris[N_train + N_valid:]
training_target = shuffled_target_iris[:N_train]
validation_target = shuffled_target_iris[N_train:N_train + N_valid]
test_target = shuffled_target_iris[N_train + N_valid:]

### Choix paramètres ###
K = np.arange(1, 5)
val_err = []
for i in range(1000):
    validation_errors = []
    for k in K:
        algo = knn.KNeighborsClassifier(k)
        algo.fit(training_features, training_target)
        
        validation_errors.append(np.sum(algo.predict(validation_features) != validation_target))
    val_err.append(validation_errors)

print(np.mean(np.array(val_err), axis = 0))

### Test paramètre retenu ###
k = 1
test_err = []
for i in range(10000):
    algo = knn.KNeighborsClassifier(k)
    algo.fit(np.concatenate((training_features, validation_features), axis = 0),
             np.concatenate((training_target, validation_target), axis = 0))
    test_err.append(np.sum(algo.predict(test_features) != test_target))

print(np.mean(np.array(test_err), axis = 0))
