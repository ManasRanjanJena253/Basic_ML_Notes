#Matplotlib:

#Useful for making Plots

# importing matplotlib library
import matplotlib.pyplot as plt


# import numpy to get data for our plots
import numpy as np


x = np.linspace(0,10,100)
y = np.sin(x) #Gives all the sin values for the elements present in x . For eg :: if element in x is 1 then it will have its
#corresponding element in y as sin(1) .

z = np.cos(x)


print(x)
print(y)


#Plotting the data


# sine wave
plt.plot(x,y)  # Will create a sin graph. By default, it will create a line plot if type of plot not specified.
plt.show() # Will show the created sin graph .



# cosine wave
plt.plot(x,z)
plt.show()



# adding title, x-axis & y-axis labels
plt.plot(x,y)
plt.xlabel('angle')  #Gives a name to the x-axis .
plt.ylabel('sine value') #Gives a name to the y-axis .
plt.title('sine wave') #Gives a title to our graph .
plt.show()



# parabola
x = np.linspace(-10,10,20) #Gives 20 equally spaced values between -10 and 10.
y = x**2
plt.plot(x,y)
plt.show()



plt.plot(x, y, 'r+')  #This will plot a graph of red colour (due to r) and will plot the points as '+' signs .
plt.show()



plt.plot(x, y, 'g.') #This will plot a graph of green colour (due to g) and will plot the points as '.' signs .
plt.show()



plt.plot(x, y, 'rx') #This will plot a graph of red colour (due to r) and will plot the points as 'x' signs .
plt.show()



x = np.linspace(-5,5,50)
plt.plot(x, np.sin(x), 'g-') #This will create a simple green line graph.
plt.plot(x, np.cos(x), 'r--') #This will create a red graph with dashed lines.
plt.show()  #This will show a graph as a combination of both the  above plotted graphs .


#Bar Plot


fig = plt.figure()  #This will create an empty plot .
ax = fig.add_axes([0,0,1,1]) #In this (0,0) represents the origin and 1,1 represents the dimensions of the rectangle.
languages = ['English','French','Spanish','Latin','German']
people = [100, 50, 150, 40, 70]
ax.bar(languages, people) #This will create a bar graph by plotting languages on x-axis and plotting people on y-axis.
plt.xlabel('LANGUAGES')
plt.ylabel('NUMBER OF PEOPLE')
plt.show()


#Pie Chart


fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
languages = ['English','French','Spanish','Latin','German']
people = [100, 50, 150, 40, 70]
ax.pie(people, labels=languages, autopct='%1.1f%%') #Will create a pie chart . The autopct is used for telling the function upto what decimal point to calculate the data before plotting.
plt.show()


#Scatter Plot


x = np.linspace(0,10,30)
y = np.sin(x)
z = np.cos(x)
fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.scatter(x,y,color='g') # Will build a scatter plot in green colour.
ax.scatter(x,z,color='b') # Will build a scatter plot in blue colour.
plt.show()


#3D Scatter Plot


fig3 = plt.figure()
ax = plt.axes(projection='3d') #This will create a 3D plot.
z = 20 * np.random.random(100) #This will give 100 random values .
x = np.sin(z)
y = np.cos(z)
ax.scatter(x,y,z,c=z,cmap='Blues') #This will create Scatter graph in 3D plot.
plt.show()

