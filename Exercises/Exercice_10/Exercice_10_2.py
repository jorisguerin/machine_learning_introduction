# -*- coding: utf-8 -*-

import numpy as np
import sklearn.neural_network as nn

### Function to approximate ###
def fonction(x):
    return [x[0]**3 * np.sin(2*x[1])**2, x[2]**2+x[0]]

### Synthetic dataset generation ###
inf_bound = 0
sup_bound = np.pi
N_train = 1000
N_test = 20

X = np.array([[np.random.uniform(inf_bound, sup_bound), np.random.uniform(inf_bound, sup_bound), np.random.uniform(inf_bound, sup_bound)] for i in range(N_train)])
Y = np.array([fonction(x) for x in X])

### Neural network regression ###
regressor = nn.MLPRegressor(hidden_layer_sizes = (10, 10, 10, 10), activation = 'tanh', solver = 'lbfgs',  max_iter = 100000)

regressor.fit(X, Y)

### Neural network validation ###
X2 = np.array([[np.random.uniform(inf_bound, sup_bound), np.random.uniform(inf_bound, sup_bound), np.random.uniform(inf_bound, sup_bound)] for i in range(N_test)])
Y2_ex = np.array([fonction(x) for x in X2])
Y2_pred = regressor.predict(X2)

errors = np.abs(Y2_ex - Y2_pred)

print 'Mean validation error :', np.mean(errors[:, 0]), np.mean(errors[:, 1])
print 'Max validation error :', np.max(errors[:, 0]), np.max(errors[:, 1])