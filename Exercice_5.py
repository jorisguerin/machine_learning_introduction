import numpy as np
import sklearn.datasets as ds
import sklearn.svm as svm

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

### Choix paramètres grossier ###
C = [10**i for i in range(-3, 3)]
val_err = []
for i in range(1000):
    validation_errors = []
    for c in C:
        algo = svm.LinearSVC(C = c)
        algo.fit(training_features, training_target)
        
        validation_errors.append(np.sum(algo.predict(validation_features) != validation_target))
    val_err.append(validation_errors)

print(np.mean(np.array(val_err), axis = 0))

### Choix paramètre plus fin ###
C = np.arange(0.1, 1, 0.2)
val_err = []
for i in range(1000):
    validation_errors = []
    for c in C:
        algo = svm.LinearSVC(C = c)
        algo.fit(training_features, training_target)
        
        validation_errors.append(np.sum(algo.predict(validation_features) != validation_target))
    val_err.append(validation_errors)

print np.mean(np.array(val_err), axis = 0)

### Test paramètre retenu ###
c = 0.1
test_err = []
for i in range(1000):
    algo = svm.LinearSVC(C = c)
    algo.fit(np.concatenate((training_features, validation_features), axis = 0),
             np.concatenate((training_target, validation_target), axis = 0))
    test_err.append(np.sum(algo.predict(test_features) != test_target))

print(np.mean(np.array(test_err), axis = 0))
