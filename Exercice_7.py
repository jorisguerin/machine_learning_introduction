# -*- coding: utf-8 -*-

##Useful functions
#numpy.linalg.inv
#numpy.random.uniform
#matplotlib.pyplot.scatter
#matplotlib.pyplot.plot

import numpy as np
import matplotlib.pyplot as plt

### Clear workspace ###
plt.close('all')
      
### Number of training points to generate ###
N_train = 15

### Linear regression function ###

def lin_reg(X, Y):
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))

### Synthetic training data generation ###

x = np.random.uniform(0, 2, N_train)
y = [x[i]**2 + np.sin(5*x[i]) for i in range(N_train)]

plt.scatter(x, y) #plot synthetic data

### Plot the real function ###

x_plot = np.linspace(0, 2, 100)
y_plot = [x_plot[i]**2 + np.sin(5*x_plot[i]) for i in range(len(x_plot))]
          
plt.plot(x_plot, y_plot) #plot function

### Regression ###
X = np.array([[1, x[i]] for i in range(len(x))])
Y = np.array(y)

beta = lin_reg(X, Y)

### Plot regression line ###
y_reg = [np.dot(beta, np.array([1, x_plot[i]])) for i in range(len(x_plot))]

plt.plot(x_plot, y_reg)
    