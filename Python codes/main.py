# modules
import numpy as np

# functions
''' sigmoid function'''
def nonlin (x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([
                [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]
            ])

# output dataset
y = np.array([
                [0,0,1,1]
            ]).T

# seed random number to make calculation ''' deterministic'''
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1))-1

# training a layer
for iter in range(10000):

    # forward propagation
    L0 = X
    L1 = nonlin(np.dot(L0,syn0))

    # how much did we miss?
    L1_error = y - L1

    # multiply how much we missed by the slope of the sigmod at the values in L1
    L1_delta = L1_error * nonlin(L1, True)

    # update weights
    syn0 += np.dot(L0.T, L1_delta)

# prints
print "Output after training"
print L1

A = nonlin(1, False)
print A