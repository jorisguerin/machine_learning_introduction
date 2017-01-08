# -*- coding: utf-8 -*-

##Useful functions
#pickle.load

import numpy as np
import pickle
import sklearn.neural_network as nn

### Load dataset ###
with open('training_set.p', 'rb') as fichier:
    training_set = pickle.load(fichier)
    th_train = training_set[0]
    xy_train = training_set[1]
    
with open('testing_set.p', 'rb') as fichier:
    testing_set = pickle.load(fichier)
    th_test = testing_set[0]
    xy_test = testing_set[1]
    
### Neural network training ###
regressor = nn.MLPRegressor(hidden_layer_sizes = (7), activation = 'relu', solver = 'lbfgs',  max_iter = 10000000)
regressor.fit(xy_train, th_train)

### Neural network testing ###
score_train = regressor.score(xy_train, th_train)
score_test = regressor.score(xy_test, th_test)
print 'Score on the training set : ', score_train
print 'Score on the testing set : ', score_test

### Save network ###
## If better than the one currently stored ##
try:
    with open('NN_IK.p', 'rb') as fichier:
        prev_reg = pickle.load(fichier)
    prev_score_test = prev_reg.score(xy_test, th_test)
except IOError:
    prev_score_test = - np.inf
    
if score_test > prev_score_test:
    with open('NN_IK.p', 'wb') as fichier:
        pickle.dump(regressor, fichier)