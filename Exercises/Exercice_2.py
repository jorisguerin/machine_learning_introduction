# -*- coding: utf-8 -*-

import time
import numpy as np
import sklearn.datasets as ds
import sklearn.neighbors as knn

### Import dataset ###

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

### Get the function from Exercise 1 ###

def distance(point1, point2):
    return np.linalg.norm(point2 - point1)
    
def k_closest_points(point, dataset, k):
    distances = [distance(point, data) for data in dataset]
    max_dist = np.max(distances)
    indices_knn = []
    for i in range(k):
        min_index = np.argmin(distances)
        indices_knn.append(min_index)
        distances[min_index] += max_dist
    return indices_knn

def get_class(point, dataset, target_set, k):
    closest_points = k_closest_points(point, dataset, k)
    classes_list = list(set(target_set))
    knn_classes = target_iris[closest_points]
    occurences = [np.count_nonzero(knn_classes == classes_list[0]),
                  np.count_nonzero(knn_classes == classes_list[1]),
                  np.count_nonzero(knn_classes == classes_list[2])]
    return classes_list[np.argmax(occurences)]
    
### Test the two algorithms ###

testing_point = features_iris[141] + 0.01

start = time.time()
for i in range(1000):
    predicted_class = get_class(testing_point, features_iris, target_iris, 3)
end = time.time()
print 'Prediction hand coded :', predicted_class
print 'Time hand coded :' + str(end - start) + ' sec'

model = knn.KNeighborsClassifier(3)
model.fit(features_iris, target_iris)

start = time.time()
for i in range(1000):
    predicted_class = model.predict(testing_point.reshape(1, -1))
end = time.time()
print 'Prediction scikit implementation :', predicted_class
print 'Time scikit-learn :' + str(end - start) + ' sec'

