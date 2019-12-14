# -*- coding: utf-8 -*-

##Useful functions:
#numpy.random.shuffle
#numpy.concatenate
#modulo operator : %

import numpy as np
import sklearn.datasets as ds
import sklearn.neighbors as knn

### Load the data ###

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

### Shuffle the dataset ###

N = len(features_iris)

randomized_indices = np.arange(len(features_iris))
np.random.shuffle(randomized_indices)

shuffled_features_iris = features_iris[randomized_indices] #Iris dataset after randomization
shuffled_target_iris = target_iris[randomized_indices] #Corresponding targets

### Split the dataset into three equal subsets ###

features = [shuffled_features_iris[:N / 3], shuffled_features_iris[N / 3 : 2 * N / 3], shuffled_features_iris[2 * N / 3:]]
targets = [shuffled_target_iris[:N / 3], shuffled_target_iris[N / 3 : 2 * N / 3], shuffled_target_iris[2 * N / 3:]]

### Build the model ###

model = knn.KNeighborsClassifier(3)

### Carry out 3-folds ###
total_errors = 0
for i in range(3):
    ## Build training set ##
    training_features = np.concatenate((features[i%3], features[(i+1)%3]), axis = 0)
    training_targets = np.concatenate((targets[i%3], targets[(i+1)%3]), axis = 0)
    
    ## Train model ##
    model.fit(training_features, training_targets)
    
    ## Build testing set ##
    testing_features = features[(i+2)%3]
    testing_targets = targets[(i+2)%3]
    
    ## Compute error rate ##
    predictions = model.predict(testing_features)
    errors = np.abs(predictions - testing_targets)
    total_errors += np.sum(errors)
    
    print 'Errors ' + str(i + 1) + ' : ' + str(np.sum(errors))

print 'Average error rate:', float(total_errors) / N

