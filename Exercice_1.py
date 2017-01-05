import numpy as np
import sklearn.datasets as ds

iris = ds.load_iris()

features_iris = iris.data
target_iris = iris.target

def distance(point1, point2):
    return np.linalg.norm(point2 - point1)

#dist_test = distance(np.array([0,0,0]), np.array([1,1,1]))
#print np.sqrt(3)
#print(dist_test)

def closest_point(point, dataset):
    distances = [distance(point, data) for data in dataset]
    return np.argmin(distances)
    
#point_test = features_iris[41] + 0.01
#point_proche_test = closest_point(point_test, features_iris)
#print(point_proche_test)
    
def k_closest_points(point, dataset, k):
    distances = [distance(point, data) for data in dataset]
    max_dist = np.max(distances)
    indices_knn = []
    for i in range(k):
        min_index = np.argmin(distances)
        indices_knn.append(min_index)
        distances[min_index] += max_dist
    return indices_knn

#point_test = features_iris[141] + 0.01
#points_proche_test = k_closest_points(point_test, features_iris, 10)
#print(points_proche_test)

def get_class(point, dataset, target_set, k):
    closest_points = k_closest_points(point, dataset, k)
    classes_list = list(set(target_set))
    knn_classes = target_iris[closest_points]
    occurences = [np.count_nonzero(knn_classes == classes_list[0]),
                  np.count_nonzero(knn_classes == classes_list[1]),
                  np.count_nonzero(knn_classes == classes_list[2])]
    return classes_list[np.argmax(occurences)]
                        
point_test = features_iris[141] + 0.01
classe = get_class(point_test, features_iris, target_iris, 3)
print(classe)