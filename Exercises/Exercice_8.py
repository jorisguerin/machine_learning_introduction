# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

### Create polynomial regression functions ###

def lin_reg(X, Y):#Linear regression
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))

def poly_X(x, p):#polynomial kernel construction
    return np.array([[x[i]**j for j in range(p+1)] for i in range(len(x))])

def poly_inference(beta, x, p):#inference for polynomial regression
    return [np.dot(beta,  np.array([x[i]**j for j in range(p+1)])) for i in range(len(x))]

### Build training and plotting datasets ###

N_train = 15

x = np.random.uniform(0, 2, N_train)
y = [x[i]**2 + np.sin(5*x[i]) for i in range(N_train)]

plt.scatter(x, y)

x_test = np.linspace(0, 2, 100)
y_test = [x_test[i]**2 + np.sin(5*x_test[i]) for i in range(len(x_test))]
          
plt.plot(x_test, y_test)

### Build design matrix and Y matrix
X = np.array([[1, x[i]] for i in range(len(x))])
Y = np.array(y)

### Plot polynomial regression results for different degrees ###
legende = ['true function']
for p in range(1, 5):
    X_polyreg = poly_X(x, p)
    beta = lin_reg(X_polyreg, Y)
    y_polyreg = poly_inference(beta, x_test, p)
    plt.plot(x_test, y_polyreg)
    legende.append('p = ' + str(p))
legende.append('points')
plt.legend(legende, loc='top left')
