# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

pl.close('all')

##robot params##
robot_params = [2,2]
angular_range = 180

def DK(thetas, robot_params):
    l1, l2 = robot_params
    th1_r, th2_r = np.deg2rad(thetas[0]), np.deg2rad(thetas[1])
    x = l1 * np.sin(th1_r) + l2 * np.sin(th1_r + th2_r)
    y = l1 * np.cos(th1_r) + l2 * np.cos(th1_r + th2_r)
    return [x, y]

N_train = 2000
rect = [[-5,-5], [5, 5]]
angular_range = [[180, 180]]#, [180, -180], [-180,180], [-180,-180], [360,360]]
for i in range(5):
    liste_th_train, liste_xyo_train = [], []
    n = 0
    while n < N_train:
        theta = [np.random.uniform(0,angular_range[i][0]), np.random.uniform(0,angular_range[i][1])]
        xyo = DK(theta, robot_params)
        if rect[0][0] <= xyo[0] <= rect[1][0] and rect[0][1] <= xyo[1] <= rect[1][1]:
            liste_th_train.append(theta)
            liste_xyo_train.append(xyo)
            n += 1
        
    th_train = np.array(liste_th_train)
    xyo_train = np.array(liste_xyo_train)
    
    fig, ax = pl.subplots()
    for i in range(N_train):
        ax.scatter(xyo_train[i, 0], xyo_train[i, 1])
    pl.show()