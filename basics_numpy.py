

import numpy as np
import time

### Useful tips ###
# To comment a block of lines, just highlight them and press 'ctrl + 1'
# When you don't know how to do something in programming, very often Google knows!

### Lists ###

print 'Lists:'
my_list_of_integers = [1, 2, 3] #Create a list, items can be of any type
my_list_of_strings  = ['abc', 'def']
my_mixed_type_list  = ['abc', 2, 21.3, 'def']

print 'first item of string list :', my_list_of_strings[0] #In python, starting index for a list is 0
print 'last item of string list :', my_list_of_strings[-1] #A list can be indexed backward
print 'Middle items of mixed list:', my_mixed_type_list[1:3] #By doing 1:3, & is taken but three is not included in the interval

### Loops ###

print '\nLoops:'
print 'types of mixed list items:'
for i in range(len(my_mixed_type_list)):#range(n) means that i loops from 0 to n-1
    print 'type of item ' + str(i), type(my_mixed_type_list[i])

print '\nprint all integers less than 3:'
k = 0
while k < 3:
    print(k)
    k += 1

### If statement ###

print '\nIf statement:'
i = 3
if i == 2:
    print str(i) + ' is equal to 2'
else:
    print str(i) + ' is not equal to 2'

### Fonctions ###

print '\nFunctions:'
def addition(x, y):
    return x + y
print '5 + 6 is equal to :', addition(5, 6)

### Vecteurs et matrices ###

print '\nVectors and Matrices:'
my_vector = np.array([0, 1, 2, 4])
print 'my vector is :', my_vector
print 'vector addition :', my_vector + my_vector
print 'term by term multiplication :', my_vector * my_vector
print 'dot product :', np.dot(my_vector, my_vector)

my_matrix = np.array([[1,2,3,5], [5,6,7,4], [5,4,2,3]])
print 'my matrix is :\n', my_matrix
print 'term to term multiplication :\n', my_matrix * my_vector
print 'dot product :\n',np.dot(my_matrix, my_vector)

### Index a matrix ###

print '\nMatrix indexing:'
print 'my matrix is :\n', my_matrix
print 'Lower rows, left columns:\n', my_matrix[1:3, 1:]

### Numpy functions ###

print '\nNumpy functions:'
a = np.sqrt(3) / 2
result = np.rad2deg(np.arcsin(a))
print 'Combine numpy functions :', result

### Linear algebra functions ###

print '\nLinear algebra functions:'
mat = np.array([[1, 2, 3], [2, 3, 4], [5, 9, 2]])
print 'Eigen values :', np.linalg.eigvals(mat)

### Time measurement ###

print '\nTime measurement:'
start = time.time()
for i in range(100000):
    2 * i
end = time.time()
print 'time elapsed ' + str(end-start) + ' secondes'