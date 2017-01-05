# -*- coding: utf-8 -*-

import numpy as np
import sklearn.cluster as clstr
import sklearn.preprocessing as prprc
import urllib

### Importer données wine ###

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
rawData = urllib.urlopen(url)

wine = np.loadtxt(rawData, delimiter=",")
features_wine = wine[:, 1:]
targets_wine = wine[:, 0]

### Mélanger les données ###

randomized_indices = np.arange(len(features_wine))
np.random.shuffle(randomized_indices)
features_wine = features_wine[randomized_indices]
targets_wine = targets_wine[randomized_indices]

features_wine = prprc.scale(features_wine)
### Trier données ###
err = []
for j in range(100):
    model = clstr.KMeans(3)
    model.fit(features_wine)
    
    ### Tester clustering ###
    
    predictions = model.predict(features_wine)
    erreur = 0


    for i in range(len(predictions)):
        if predictions[i] != targets_wine[i] - 1:
            erreur += 1
    
    err.append(erreur)
    erreur = 0

print(100. * np.min(err) / targets_wine.shape[0])