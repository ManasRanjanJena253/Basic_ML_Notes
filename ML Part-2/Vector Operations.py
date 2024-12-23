import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()  # Use to get a grid on the graph otherwise the graph will be plotted on a plain background.

#Plotting a vector


plt.quiver(0,0,4,5) #This function is used to create vector arrows . It takes 4 arguments as inputs , (x-coordinate of the arrow heads,y-coordinate of the arrow heads , x-component of the vector , y-component of the vector).
plt.show()



plt.quiver(0,0,4,5, scale_units='xy', angles='xy', scale=1)  #This will take the scale of the arguments we have passed inside the function as 1 for correct plotting.
plt.xlim(-8,8)  # The right and left limit of the x-axis.
plt.ylim(-8,8)  # The down and upper limit of the y-axis.
plt.show()



plt.quiver(0,0,4,5, scale_units='xy', angles='xy', scale=1, color='b')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


# To plot two vectors in one graph.

plt.quiver(1,1,4,5, scale_units='xy', angles='xy', scale=1, color='b') # b tells the function to create blue colour arrow.
plt.quiver(0,0,-3,-6, scale_units='xy', angles='xy', scale=1, color='y') # y tells the function to create yellow colour arrow.
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


#Addition of two Vectors


vector_1 = np.asarray([0,0,2,3])

vector_2 = np.asarray([0,0,3,-2])

Sum = vector_1 + vector_2
print(Sum)


plt.quiver(0,0,2,3, scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(0,0,3,-2, scale_units='xy', angles='xy', scale=1, color='y')
plt.quiver(0,0,5,1, scale_units='xy', angles='xy', scale=1, color='r')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


#Subtraction of 2 vectors


vector_1 = np.asarray([0,0,2,3])

vector_2 = np.asarray([0,0,3,-2])

difference = vector_1 - vector_2
print(difference)


plt.quiver(0,0,2,3, scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(0,0,3,-2, scale_units='xy', angles='xy', scale=1, color='y')
plt.quiver(0,0,-1,5, scale_units='xy', angles='xy', scale=1, color='r')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


#Multiplying a vector by a scalar


vector_1 = np.asarray([0,0,2,3])

vector_2 = 2*vector_1

print(vector_2)


plt.quiver(0,0,2,3, scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(0,0,4,6, scale_units='xy', angles='xy', scale=1, color='y')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()



vector_1 = np.asarray([0,0,2,3])

vector_2 = -0.5*vector_1

print(vector_2)


plt.quiver(0,0,2,3, scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(0,0,-1,-1.5, scale_units='xy', angles='xy', scale=1, color='y')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


#Dot Product of 2 Vectors


a = np.array([2, 3])

b = np.array([4, 4])

a_dot_b = np.dot(a, b)  #Function to find the dot product of two vectors i.e a,b.


print(a_dot_b)


c = np.array([40, 20, 35])

d = np.array([53, 24, 68])

c_dot_d = np.dot(c, d)


print(c_dot_d)

#Cross Product of 2 Vectors


a = np.array([2, 3])

b = np.array([4, 4])

a_cross_b = np.cross(a, b)


print(a_cross_b)


c = np.array([5, 10, 20])

d = np.array([18, 32, 50])

c_cross_d = np.cross(c, d)


print(c_cross_d)

#Projection of "a" vector on "v" vector
# There is no direct function to find the projection so we need to apply the formula directly.

a = np.array([2, 5])

v = np.array([8, -6])

# magnitude of "v" vector
magnitude_of_v = np.sqrt(sum(v**2))

proj_of_a_on_v = (np.dot(a,v)/magnitude_of_v**2)*v

print('Projection of a vector on v vector = ', proj_of_a_on_v)


a = np.array([23, 45, 62])

v = np.array([45, 82, 67])

# magnitude of "v" vector
magnitude_of_v = np.sqrt(sum(v**2))

proj_of_a_on_v = (np.dot(a,v)/magnitude_of_v**2)*v

print('Projection of a vector on v vector = ', proj_of_a_on_v)

