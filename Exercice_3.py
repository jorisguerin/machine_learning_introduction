# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets as ds
import sklearn.neighbors as knn

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

N = len(features_iris)

randomized_indices = np.arange(len(features_iris))
np.random.shuffle(randomized_indices)

shuffled_features_iris = features_iris[randomized_indices]
shuffled_target_iris = target_iris[randomized_indices]

features = [shuffled_features_iris[:N / 3], shuffled_features_iris[N / 3 : 2 * N / 3], shuffled_features_iris[2 * N / 3:]]
targets = [shuffled_target_iris[:N / 3], shuffled_target_iris[N / 3 : 2 * N / 3], shuffled_target_iris[2 * N / 3:]]

model = knn.KNeighborsClassifier(3)

for i in range(3):
    model.fit(np.concatenate((features[i%3], features[(i+1)%3]), axis = 0), np.concatenate((targets[i%3], targets[(i+1)%3]), axis = 0))

    predictions = model.predict(features[(i+2)%3])
    erreurs = np.abs(predictions - targets[(i+2)%3])

    print(np.sum(erreurs))

