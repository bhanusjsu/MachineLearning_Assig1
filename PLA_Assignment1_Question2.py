# Load the libraries you will need
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.linear_model import perceptron
from pandas import *
 
# You only need this if using Notebook
#%matplotlib inline

# Put some data into a dataframe
inputs = DataFrame({
'A' :       [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
'B' :       [1,2,3,4,1,2,3.5,4,1,1.5,3,4,1,2.5,3,4,1,2,3,4],
'Targets' : [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1]
})

# Set an array of colours, we could call it
# anything but here we call is colormap
# It sounds more awesome
colormap = np.array(['r','b', 'k'])
 
# Plot the data, A is x axis, B is y axis
# and the colormap is applied based on the Targets
plt.scatter(inputs.A, inputs.B, c=colormap[inputs.Targets], s=40)


# Create the perceptron object (net)
net = perceptron.Perceptron(n_iter=60, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
 
# Train the perceptron object (net)
net.fit(inputs[['A', 'B']],inputs['Targets'])

# Output the coefficints
# print "Coefficient 0 " + str(net.coef_[0,0])
# print "Coefficient 1 " + str(net.coef_[0,1])
# print "Bias " + str(net.intercept_)

# Plot the original data
plt.scatter(inputs.A, inputs.B, c=colormap[inputs.Targets], s=40)
 
# Calc the hyperplane (decision boundary)
ymin, ymax = plt.ylim()
print (net.coef_)
w = net.coef_[0]
print (w)
a = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = a * xx - (net.intercept_[0]) / w[1]
 
# Plot the hyperplane
plt.plot(xx,yy, 'k-')
plt.ylim([0,8]) # Limit the y axis size
plt.show()