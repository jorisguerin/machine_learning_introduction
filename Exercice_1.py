import numpy as np
import sklearn.datasets as ds

### Load the dataset ###

iris = ds.load_iris()

features_iris = iris.data
target_iris   = iris.target

### Define the functions needed ###

def distance(point1, point2): #Compute cartesian distance between points
    return np.linalg.norm(point2 - point1)

## Uncomment this code to test the function ##
#print "Test 'distance' function:"
#dist_test = distance(np.array([0,0,0]), np.array([1,1,1]))
#print 'true value :', np.sqrt(3)
#print 'Function output :', dist_test

def closest_point(point, dataset): #Compute the closest point from a given instance within the dataset
    distances = [distance(point, data) for data in dataset] #This is a list built directly with a for loop. Specific to python.
    return np.argmin(distances)

## Uncomment this code to test the function ##
#print "Test 'closest_point' function:"
#point_test = features_iris[41] + 0.01
#point_proche_test = closest_point(point_test, features_iris)
#print 'true value :', 41
#print 'Function output :', point_proche_test
    
def k_closest_points(point, dataset, k): #Compute the list of k closest point from a given instance within the dataset
    distances = [distance(point, data) for data in dataset]
    max_dist = np.max(distances)
    indices_knn = []
    for i in range(k):
        min_index = np.argmin(distances) #Get the min
        indices_knn.append(min_index)
        distances[min_index] += max_dist #Add the max distance to make sure it's not the min anymore
    return indices_knn

## Uncomment this code to test the function ##
#print "Test 'k_closest_point' function:"
#point_test = features_iris[141] + 0.01
#points_proche_test = k_closest_points(point_test, features_iris, 10)
#print 'Function output :', points_proche_test

def get_class(point, dataset, target_set, k):
    closest_points = k_closest_points(point, dataset, k)
    classes_list = list(set(target_set)) #Get the list of all possible classes
    knn_classes = target_iris[closest_points] #Select only the values corresponding to the closest points
    occurences = [np.count_nonzero(knn_classes == classes_list[0]),
                  np.count_nonzero(knn_classes == classes_list[1]),
                  np.count_nonzero(knn_classes == classes_list[2])] #Count the different occurences of each class among the nearest neighbors
    return classes_list[np.argmax(occurences)]

## Uncomment this code to test the function ##
print "\nTest The K-nearest neighbors program:"
point_test = features_iris[141] + 0.01
print 'Parameters: k = 3 // Dataset : Iris'
print 'Expected class : 2'
predicted_class = get_class(point_test, features_iris, target_iris, 3)
print 'Prediction :', predicted_class