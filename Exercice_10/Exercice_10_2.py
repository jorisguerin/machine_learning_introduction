# -*- coding: utf-8 -*-

import numpy as np
import sklearn.neural_network as nn

def fonction(x):
    return [x[0]**3 * np.sin(2*x[1])**2, x[2]**2+x[0]]
    
borne_inf = 0
borne_sup = np.pi
N_train = 1000
N_test = 20

X = np.array([[np.random.uniform(borne_inf, borne_sup), np.random.uniform(borne_inf, borne_sup), np.random.uniform(borne_inf, borne_sup)] for i in range(N_train)])
Y = np.array([fonction(x) for x in X])

regressor = nn.MLPRegressor(hidden_layer_sizes = (10, 10, 10, 10), activation = 'tanh', solver = 'lbfgs',  max_iter = 100000)

regressor.fit(X, Y)

X2 = np.array([[np.random.uniform(borne_inf, borne_sup), np.random.uniform(borne_inf, borne_sup), np.random.uniform(borne_inf, borne_sup)] for i in range(N_test)])
Y2_ex = np.array([fonction(x) for x in X2])
Y2_pred = regressor.predict(X2)

errors = np.abs(Y2_ex - Y2_pred)

print('\n')
print(np.mean(errors[:, 0]), np.mean(errors[:, 1]))
print(np.max(errors[:, 0]), np.max(errors[:, 1]))