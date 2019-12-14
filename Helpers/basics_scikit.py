# -*- coding: utf-8 -*-

import sklearn.datasets as ds
import sklearn.neighbors as knn

#Building a Machine learning model using scikit-learn is always the same logic
#From a dataset (which in some cases can be imported directly from sklearn), 
#we start by selecting a model (along with its hyperparameters),
#and we fit the data to this model, which basically means training the model.
#
#Then, we can use the model to predict outputs for new objects.

### Import data ###

cancer = ds.load_breast_cancer()

features_cancer = cancer.data
targets_cancer  = cancer.target

### Build a model ###

model = knn.KNeighborsClassifier(3)
model.fit(features_cancer, targets_cancer)

### Use a model ###

print model.predict(features_cancer[0:8, :])