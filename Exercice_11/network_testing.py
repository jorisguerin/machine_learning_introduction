import numpy as np
import pickle

robot_params = [2,2]

### cinématique directe ###
def DK(thetas, robot_params):
    l1, l2 = robot_params
    th1_r, th2_r = np.deg2rad(thetas[0]), np.deg2rad(thetas[1])
    x = l1 * np.sin(th1_r) + l2 * np.sin(th1_r + th2_r)
    y = l1 * np.cos(th1_r) + l2 * np.cos(th1_r + th2_r)
    return [x, y]
    
### Charger donnees tests ###
with open('testing_set.p', 'rb') as fichier:
    testing_set = pickle.load(fichier)
    th_test = testing_set[0]
    xy_test = testing_set[1]

### Charger NN ### 
with open('NN_IK.p', 'rb') as fichier:
    regressor = pickle.load(fichier)

### Predict ###
th_pred = regressor.predict(xy_test)

### calcul erreur ###
xy_pred = np.array([DK(th_pred[i], robot_params) for i in range(len(th_pred))])
errors = np.abs(xy_pred - xy_test)

mean_error = np.mean(errors)
max_error = np.max(errors)

print('mean error: ', mean_error)
print('max error: ', max_error)