# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
      
N_train = 15

def lin_reg(X, Y):
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))
  
x = np.random.uniform(0, 2, N_train)
y = [x[i]**2 + np.sin(5*x[i]) for i in range(N_train)]

plt.scatter(x, y)

x_test = np.linspace(0, 2, 100)
y_test = [x_test[i]**2 + np.sin(5*x_test[i]) for i in range(len(x_test))]
          
plt.plot(x_test, y_test)

X = np.array([[1, x[i], x[i]**2] for i in range(len(x))])
Y = np.array(y)

beta = lin_reg(X, Y)

y_reg = [np.dot(beta, np.array([1, x_test[i], x_test[i]**2])) for i in range(len(x_test))]

plt.plot(x_test, y_reg)
    