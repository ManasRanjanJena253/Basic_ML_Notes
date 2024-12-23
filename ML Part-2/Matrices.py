import numpy as np

#Creating a Matrix using numpy


matrix_1 = np.array([[2,3],[6,7]])  # Here [2,3] will be the first row and [6,7] will be the second row.
print(matrix_1)


print(matrix_1.shape)


matrix_2 = np.array([[10,35,45],[50,64,80],[20,15,90]])
print(matrix_2)


print(matrix_2.shape)

#Creating Matrices with Random Values


random_matrix = np.random.rand(3,3)
print(random_matrix)


# creating matrix with random integers
random_integer_matrix = np.random.randint(100, size=(4,5))  # Will create random values inside the matrix less than 100. The size of the matrix will be 4X5.
print(random_integer_matrix)

#Matrix with all the values as 1


matrix_3 = np.ones((2,3))  # Here 2 is the number of rows and 3 is the number of columns.
# This function will have all the elements as 1. By default, the elements will be in the form of float . So we have to specify its datatype as int.
print(matrix_3)


matrix_3 = np.ones((2,3), dtype=int)
print(matrix_3)



matrix_3 = np.ones((10,10), dtype=int)
print(matrix_3)


#Null Matrix or Zero Matrix


null_matrix = np.zeros((4,4)) # Will create a matrix of size 4X4 having all of its values as 0.
print(null_matrix)


null_matrix = np.zeros((7,7))
print(null_matrix)

#Identity Matrix


identity_matrix = np.eye(3,3) # Will create an identity matrix.
print(identity_matrix)


identity_matrix = np.eye(5,5)
print(identity_matrix)

#Transpose of a Matrix


# matrix with random integer values
a = np.random.randint(100, size=(4,5))
print(a)


transpose_of_a = np.transpose(a) # Function for transposing a matrix.
print(transpose_of_a)


#Matrix Addition

#Two Matrices can be added only if they have the same shape


# creating two matrices

A = np.array([[2,3],[4,5]])

B = np.array([[6,7],[8,9]])


print(A)


print(B)


print(A.shape)


print(B.shape)

#Adding two Matrices


sum = A + B


print(sum)


# create two matrices with random values
matrix_1 = np.random.randint(10, size=(3,3))
matrix_2 = np.random.randint(20, size=(3,3))


print(matrix_1)


print(matrix_2)


sum = np.add(matrix_1, matrix_2)


print(sum)


# create two matrices with random values
matrix_3 = np.random.randint(10, size=(3,3))
matrix_4 = np.random.randint(20, size=(3,3))


sum_2 = np.add(matrix_3, matrix_4)


#Matrix Subtraction


# creating two matrices

A = np.array([[2,3],[4,5]])

B = np.array([[6,7],[8,9]])


print(A)


print(B)


difference = A - B


print(difference)


# create two matrices with random values
matrix_1 = np.random.randint(10, size=(3,3))
matrix_2 = np.random.randint(20, size=(3,3))


print(matrix_1)


print(matrix_2)


diff = np.subtract(matrix_1, matrix_2)


print(diff)

#Multiplying a matrix by a scalar


x = 5
y = np.random.randint(10, size=(4,4))

print(y)


product = np.multiply(x,y)


print(product)

#Multiplying 2 Matrices


# create two matrices with random values
matrix_3 = np.random.randint(5, size=(3,3))
matrix_4 = np.random.randint(5, size=(3,4))


print(matrix_3)

print(matrix_4)


product = np.dot(matrix_3, matrix_4)


print(product)


print(product.shape)


# create two matrices with random values
matrix_3 = np.random.randint(5, size=(3,3))
matrix_4 = np.random.randint(5, size=(4,4))


product = np.dot(matrix_3, matrix_4)



# create two matrices with random values
matrix_3 = np.random.randint(5, size=(3,3))
matrix_4 = np.random.randint(5, size=(3,3))


print(matrix_3)

print(matrix_4)


product = np.multiply(matrix_3, matrix_4)


print(product)


# create two matrices with random values
matrix_3 = np.random.randint(5, size=(3,3))
matrix_4 = np.random.randint(5, size=(3,4))


product = np.multiply(matrix_3, matrix_4)







