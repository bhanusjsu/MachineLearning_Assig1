import numpy as np
import matplotlib.pyplot as mpl
from sklearn.linear_model import perceptron
from pandas import *

data_input = DataFrame({
'A' :       [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
'B' :       [1,2,3,4,1,2,3.5,4,1,1.5,3,4,1,2.5,3,4,1,2,3,4],
'Targets' : [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1]
})

colormap = np.array(['r','b', 'k'])
net = perceptron.Perceptron(n_iter=60, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
net.fit(data_input[['A', 'B']],data_input['Targets'])
mpl.scatter(data_input.A, data_input.B, c=colormap[data_input.Targets], s=40)

ymin, ymax = mpl.ylim()
print (net.coef_)
w = net.coef_[0]
print (w)
a = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = a * xx - (net.intercept_[0]) / w[1]
 
mpl.plot(xx,yy, 'k-')
mpl.ylim([0,8]) 
mpl.show()