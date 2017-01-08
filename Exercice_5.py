import numpy as np
import sklearn.datasets as ds
import sklearn.svm as svm

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


### Coarse parameter selection  ###
C = [10**i for i in range(-3, 3)]
validation_errors = []
for i in range(100): #Repeat the process several time because the algorithm does not perform always exactly the same
    errors = []
    for c in C:
        algo = svm.LinearSVC(C = c)
        algo.fit(training_features, training_target)
        
        errors.append(np.sum(algo.predict(validation_features) != validation_target))
    validation_errors.append(errors)

print 'Value of C :      ', C
print 'Validation error :', np.mean(np.array(validation_errors), axis = 0)

### Finner selection ###
C = np.arange(0.1, 1, 0.2)
validation_errors = []
for i in range(100): #Repeat the process several time because the algorithm does not perform always exactly the same
    errors = []
    for c in C:
        algo = svm.LinearSVC(C = c)
        algo.fit(training_features, training_target)
        
        errors.append(np.sum(algo.predict(validation_features) != validation_target))
    validation_errors.append(errors)

print '\nValue of C :      ', C
print 'Validation error :', np.mean(np.array(validation_errors), axis = 0)

### Test chosen parameters ###

c = 0.1
test_errors = []
for i in range(100):
    algo = svm.LinearSVC(C = c)
    algo.fit(np.concatenate((training_features, validation_features), axis = 0),
             np.concatenate((training_target, validation_target), axis = 0))
    test_errors.append(np.sum(algo.predict(test_features) != test_target))

print '\nTesting error rate :', np.mean(np.array(test_errors), axis = 0) / N_test
