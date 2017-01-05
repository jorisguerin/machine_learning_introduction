import numpy as np
import pickle
import pylab as pl
from matplotlib import collections  as mc

### Configuration jeux de données ###
robot_params = [2,2]
cartesian_range = [[-1,-3],[1,-1]]

### affichage configuration ###    
def plot_config(thetas, robot_params, objectif = [0, 0]):
    l1, l2 = robot_params
    th1_r, th2_r = np.deg2rad(thetas[0]), np.deg2rad(thetas[1])
    
    lines = [[(0, 0), 
              (l1 * np.sin(th1_r), l1 * np.cos(th1_r))], 
             [(l1 * np.sin(th1_r), l1 * np.cos(th1_r)), 
              (l1 * np.sin(th1_r) + l2 * np.sin(th1_r + th2_r), l1 * np.cos(th1_r) + l2 * np.cos(th1_r + th2_r))]]
              
    c = np.array([(1, 0, 0, 1), (0, 0, 1, 1)])    
    lc = mc.LineCollection(lines, colors=c, linewidths=2)
    
    pl.close('all')
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.scatter(objectif[0], objectif[1], color='green')
    pl.axis([-1.1, 2.1, -3.1, 0.1])
    ax.margins(0.1)
    pl.show()

### Charger NN ### 
with open('NN_IK.p', 'rb') as fichier:
    regressor = pickle.load(fichier)

### Affichage ###
pl.close('all')
plot_config([0, 0], robot_params)

for i in range(10):
    point = pl.ginput(1)
    
    xy_target = np.array([[point[0][0], point[0][1]]])
    thetas = regressor.predict(xy_target) 
    
    plot_config(thetas[0], robot_params, xy_target[0])