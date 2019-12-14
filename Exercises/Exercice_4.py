# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets as ds
import sklearn.neighbors as knn

### Load dataset ###

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

### Split dataset ###

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

### Parameters Selection phase ###

K = np.arange(1, 6) #Different values of k to be tested
validation_errors = []
for k in K:
    algo = knn.KNeighborsClassifier(k)
    algo.fit(training_features, training_target)
    error = np.sum(algo.predict(validation_features) != validation_target)#Find misclassifications
    validation_errors.append(error)
print 'Errors for different values of k:'
print 'k :     ', K
print 'error : ', validation_errors

### Test chosen parameters ###

k = K[np.argmin(validation_errors)]
print '\nChosen k :', k

algo = knn.KNeighborsClassifier(k)
algo.fit(np.concatenate((training_features, validation_features), axis = 0),
         np.concatenate((training_target, validation_target), axis = 0))
test_error = np.sum(algo.predict(test_features) != test_target)

print 'Test error rate:', float(test_error) / N_test
