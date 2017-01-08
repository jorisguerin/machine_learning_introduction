# -*- coding: utf-8 -*-

##Useful functions:
#urllib.urlopen
#numpy.loadtxt
#sklearn.preprocessing.scale

import numpy as np
import sklearn.cluster as clstr
import sklearn.preprocessing as prprc
import urllib

### Import Wine data ###

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
rawData = urllib.urlopen(url)

wine = np.loadtxt(rawData, delimiter=",")
features_wine = wine[:, 1:]
targets_wine = wine[:, 0]

### Shuffle dataset ###

randomized_indices = np.arange(len(features_wine))
np.random.shuffle(randomized_indices)
features_wine = features_wine[randomized_indices]
targets_wine = targets_wine[randomized_indices]

### Scale dataset ###
features_wine = prprc.scale(features_wine)

### Sort data ###
errors = []
for j in range(100): #Needs to be repeated because Kmeans produces different outputs
    model = clstr.KMeans(3)
    model.fit(features_wine)
    
    ### Test clustering ###
    
    predictions = model.predict(features_wine)
    err = 0

    for i in range(len(predictions)):
        if predictions[i] != targets_wine[i] - 1:
            err += 1
    
    errors.append(err)
    err = 0

print 'Error rate (%) :', 100. * np.min(errors) / targets_wine.shape[0] #We need to take the min of errors because there is no guarantee that the labels will match