""" ENGG*6500 A1
    Train a MLP on the StumbleUpon Evergreen dataset
"""

import time
import numpy as np
import pickle
import pylab
import pandas as pd
import xlrd

from numpy.random import uniform
from autodiff import function        
from autodiff import Gradient
from scipy.linalg import norm

# Global variables
dtype = 'float32'
eps = np.finfo(np.double).eps  # -- a small number

"""
def load_data(dtype=dtype):
    with open('evergreen.pkl') as f:
        train_set, val_set, test_set = pickle.load(f)
        train_X, train_y = train_set
        val_X, val_y = val_set

    return (train_X.astype(dtype), train_y.astype('int8'),
            val_X.astype(dtype), val_y.astype('int8'))
"""
            
def parse_file(datafile,worksheet=0):
    workbook = xlrd.open_workbook(datafile)
    sheet = workbook.sheet_by_index(worksheet)

    data = [[sheet.cell_value(r, col) 
                for col in range(sheet.ncols)] 
                    for r in range(sheet.nrows)]
    return data
    
#Call data from Excel file, sheet 1 and convert to pandas dataframe   
train_X = pd.DataFrame(parse_file("FAH1000.xlsx",0)).values
train_Y = pd.DataFrame(parse_file("FAH1000.xlsx",1)).values 

val_X = pd.DataFrame(parse_file("FAH1000.xlsx",0)).values
val_Y = pd.DataFrame(parse_file("FAH1000.xlsx",1)).values 


#@function
def logistic(z):
    #return logistic sigmoid of float or ndarray `z`
    return 1.0 / (1.0 + np.exp(-z))

#@function
def p_y_given_x(W, b, x):
    return logistic(np.dot(x, W) + b)

def logreg_prediction(W, b, x):
    return p_y_given_x(W, b, x) > 0.5

#@function
def cross_entropy(x, z):
    # note we add a small epsilon for numerical stability
    return -(x * np.log(z + eps) + (1 - x) * np.log(1 - z + eps))

def logreg_cost(W, b, x, y):
    z = p_y_given_x(W, b, x)
    l = cross_entropy(y, z).mean(axis=0)
    return l

def accuracy(y, y_pred):
    return 1.0 * np.sum(y == y_pred) / y.shape[0]

#@function
def mlp_cost(X, y, W_hid, b_hid, W_out, b_out):
    ###### forward pass
    # hidden activations
    act_hid = p_y_given_x(W_hid, b_hid, X)
    # output activation
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return cross_entropy(y, act_out).mean(axis=0)

def mlp_predict(X, W_hid, b_hid, W_out, b_out):
    act_hid = p_y_given_x(W_hid, b_hid, X)
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return act_out > 0.5

def initialize_model(n_inputs, n_hidden, dtype=dtype):
    W_hid = uniform(low=-4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    high=4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    size=(n_inputs, n_hidden)).astype(dtype)
    b_hid = np.zeros(n_hidden, dtype=dtype)

    # now allocate the logistic regression model at the top
    W_out = uniform(low=-4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    high=4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    size=(n_hidden,)).astype(dtype)
    b_out = np.array(0.0)

    return W_hid, b_hid, W_out, b_out

##########################################################################
def my_grads(X, y, W_hid, b_hid, W_out, b_out):
    # Fill this in
    pass

def check_gradients(cost_fn, grad_fn, X, y, W_hid, b_hid, W_out, b_out,
                    min_diff=0.0001):
    # generate pyautodiff gradient function
    wrt = ('W_hid', 'b_hid', 'W_out', 'b_out')
    auto_grad_fn = Gradient(cost_fn, wrt=wrt)
    weights = [W_hid, b_hid, W_out, b_out]

    grads_auto = auto_grad_fn(X, y, *weights)
    grads_manual = grad_fn(X, y, *weights)
    norms = [norm(auto_g - manual_g) for auto_g, manual_g in zip(grads_auto,
                                                                 grads_manual)]
    ret = True
    print "Checking gradients:"
    for name, n in zip(wrt, norms):
        if n < min_diff:
            msg = 'OK  '
        else:
            msg = 'FAIL'
            ret = False
        print '%s: %s - norm of gradient difference with pyautodiff: %s' % \
              (name, msg, n)

    if ret:
        print "All gradients OK!"
    else:
        print "Some gradients failed."

    return ret


def train_model(train_X, train_y, test_X, test_y, W_hid, b_hid, W_out, b_out,
                learning_rate, epochs, mlp_grads=None, dtype=dtype):
    train_costs = np.zeros(epochs, dtype=dtype)
    test_costs = np.zeros(epochs, dtype=dtype)

    # Make a list of the weights
    weights = [W_hid, b_hid, W_out, b_out]

    if mlp_grads is None:
        mlp_grads = Gradient(mlp_cost, wrt=weights) 

    for epoch in xrange(epochs):
        print "Epoch", epoch

        train_cost = mlp_cost(train_X, train_y, *weights)
        test_cost = mlp_cost(test_X, test_y, *weights)
        train_costs[epoch] = train_cost
        test_costs[epoch] = test_cost

        grads = mlp_grads(train_X, train_y, *weights)

        # returns a CudaNDarray when running on GPU
        # creating a np.array copies the data back from the GPU
        if not isinstance(grads[0], np.ndarray):
            grads = [np.array(g) for g in grads]

        # update the weights
            for W, dW in zip(weights, grads):
                W -= learning_rate * dW

            print "Training set cost:", train_cost
            print "Test set cost:    ", test_cost

    return train_costs, test_costs

def run_training(n_hidden, learning_rate, epochs, data=None, model=None,
                 grad_fn=None, show_plot=True):
    t0 = time.time()
    train_X, train_y, test_X, test_y = data

    # initialize input layer parameters
    n_inputs = train_X.shape[1]  # -- aka D_0
    print "NUM input dimensions:", n_inputs

    if model is None:
        model = initialize_model(n_inputs, n_hidden)
    W_hid, b_hid, W_out, b_out = model

    if grad_fn is not None:
        check = check_gradients(mlp_cost, grad_fn, train_X, train_y,
                                W_hid, b_hid, W_out, b_out)
        if not check:
            print "Failed gradient check. Aborting training"
            return [], []

    print "Before training"
    print 'train accuracy: %6.4f' % \
          accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    print 'train cross entropy: %6.4f' % \
          mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    print 'test accuracy: %6.4f' % \
          accuracy(test_y, mlp_predict(test_X, W_hid, b_hid, W_out, b_out))
    print 'test cross entropy: %6.4f' % mlp_cost(test_X, test_y, W_hid, b_hid,
                                                 W_out, b_out)

    train_costs, test_costs = train_model(train_X, train_y, test_X, test_y,
                                          W_hid, b_hid, W_out, b_out,
                                          learning_rate, epochs, grad_fn)

    print "After training, n_hidden: %s, learning_rate: %s" % (n_hidden,
                                                               learning_rate)
    print 'train accuracy: %6.4f' % \
          accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    print 'train cross entropy: %6.4f' % \
          mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    print 'test accuracy: %6.4f' % \
          accuracy(test_y, mlp_predict(test_X, W_hid, b_hid, W_out, b_out))
    print 'test cross entropy: %6.4f' % mlp_cost(test_X, test_y, W_hid, b_hid,
                                                 W_out, b_out)

    print "training took: %s sec" % (time.time() - t0)

    if show_plot:
        pylab.plot(train_costs, '-b', label="Training data")
        pylab.plot(test_costs, '-r', label="Test data")
        pylab.legend(loc='upper right')
        pylab.xlabel("Epoch")
        pylab.ylabel("Cross entropy")
        pylab.show()

    return train_costs, test_costs

if __name__ == '__main__':

    # swap the lines below once you have implemented my_grads
    #grad_fn = my_grads
    grad_fn = None

    epochs = 250
    learning_rate = 0.01
    n_hidden = 100  # -- aka D_1

    #data = load_data()
    n_inputs = len(train_X.T)  # -- 
    n_outputs = len(train_Y.T)

    model = initialize_model(n_inputs, n_hidden)
    train_costs, test_costs = run_training(n_hidden, learning_rate, epochs,
                                           data, model, grad_fn,
                                           show_plot=True)

########## End of code