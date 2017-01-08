##Useful functions
#pickle.dump

import numpy as np
import pickle

### Dataset parameter ###
robot_params = [2,2]

angular_range = [0, 180]
cartesian_range = [[-1,-3],[1,-1]]

N_train = 2000
N_test = 100

### Direct kinematic function ###
def DK(thetas, robot_params):
    l1, l2 = robot_params
    th1_r, th2_r = np.deg2rad(thetas[0]), np.deg2rad(thetas[1])
    x = l1 * np.sin(th1_r) + l2 * np.sin(th1_r + th2_r)
    y = l1 * np.cos(th1_r) + l2 * np.cos(th1_r + th2_r)
    return [x, y]

### Building the training set ###
list_th_train, list_xy_train = [], []
n = 0
while n < N_train:
    theta = [np.random.uniform(angular_range[0],angular_range[1]), 
             np.random.uniform(angular_range[0],angular_range[1])]
    xy = DK(theta, robot_params)
    if cartesian_range[0][0] <= xy[0] <= cartesian_range[1][0] and cartesian_range[0][1] <= xy[1] <= cartesian_range[1][1]:
        list_th_train.append(theta)
        list_xy_train.append(xy)
        n += 1
th_train = np.array(list_th_train)
xy_train = np.array(list_xy_train)

### Building the testing set ### 
list_th_test, list_xy_test = [], []
n = 0
while n < N_test:
    theta = [np.random.uniform(angular_range[0],angular_range[1]), 
             np.random.uniform(angular_range[0],angular_range[1])]
    xy = DK(theta, robot_params)
    if cartesian_range[0][0] <= xy[0] <= cartesian_range[1][0] and cartesian_range[0][1] <= xy[1] <= cartesian_range[1][1]:
        list_th_test.append(theta)
        list_xy_test.append(xy)
        n += 1
th_test = np.array(list_th_test)
xy_test = np.array(list_xy_test)

### Save dataset using pickle ### 
with open('training_set.p', 'wb') as fichier:
    pickle.dump([th_train, xy_train], fichier)
with open('testing_set.p', 'wb') as fichier:
    pickle.dump([th_test, xy_test], fichier)