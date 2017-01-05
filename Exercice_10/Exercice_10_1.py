# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn

plt.close('all')

N_train = 20

x = np.random.uniform(0, 2, N_train)
y = [x[i]**2 + np.sin(5*x[i]) for i in range(N_train)]

plt.scatter(x, y)

x_test = np.linspace(0, 2, 100)
y_test = [x_test[i]**2 + np.sin(5*x_test[i]) for i in range(len(x_test))]
          
plt.plot(x_test, y_test)

model = nn.MLPRegressor(hidden_layer_sizes = (7), activation = 'tanh', solver = 'lbfgs',  max_iter = 10000000)
model.fit(x.reshape(-1,1), np.array(y).reshape(-1,1))

y_nn = model.predict(x_test.reshape(-1,1))

plt.plot(x_test, y_nn)