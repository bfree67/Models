import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras import regularizers as R
from keras import backend as K

#---------------- Put training and test data file in the same folder as py code
#Training data file name
data_file_name = "output.xlsx"
#Test/validation data file name (toggle comment if different from data file name)
test_filename = data_file_name
#test_file_name = "FAH1002.xlsx"

alpha = 0.001  # momentum rate (Default = 0.1)
min_gradient  = .0001 #minimum gradient updates allowed. if updates are too small, stop training
hidden_factor = 2 #scaling factor for number of hidden layer nodes
L1_reg = 0. # L1 norm (absolute sum of all weights) regulation (Default = 0.01)
L2_reg = 0.0001 # L2 norm (square sum of all weights) regulation (Default = 0.01)
alpha_lr = .01 #learning rate
n_epochs = 400
drop_out = 0.1
n_batch = 2


### - Load data from excel file function
def load_file(datafile,worksheet=0,type_data="Input Training"):
    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0,         #assumes 1st row are header names
                skiprows=None, 
                skip_footer=0, 
                index_col=None, 
                parse_cols=None, #default = None
                parse_dates=False, 
                date_parser=None, 
                na_values=None, 
                thousands=None, 
                convert_float=True, 
                has_index_names=None, 
                converters=None, 
                engine=None)
    # stop clock
    end = time.clock() 
    
    if (end-start > 60):
        print type_data, "data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print type_data, "data loaded in {0:.2f} seconds".format((end-start)/1.)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
    return data 

print "Loading training data.... \n"
#Sheet 0 = input X data
#Sheet 1 = input Y data

x_train = load_file(data_file_name,0,"Input Training")
y_train = load_file(data_file_name,1,"Output Training")

D = (x_train, y_train) #makes a training tuple

# convert to lists
x_train_list = x_train.tolist()
y_train_list = y_train.tolist()

print "\n Building model...\n"

n_inputs = len(x_train.T) # number of inputs
n_outputs = len(y_train.T) # of outputs
n_hidden = int((n_inputs + n_outputs ) * hidden_factor)  # of hidden units - use 2 x # of outputs
N = len(x_train) # Number of samples minus header

print"Inputs = ", n_inputs
print"Hidden layer nodes = ", n_hidden
print"Outputs = ",n_outputs, "\n"

########### Build model
model = Sequential()

#input to first hidden layer
model.add(Dense(n_hidden, input_dim=n_inputs, 
                kernel_initializer='random_uniform',
                use_bias = True,
                bias_initializer='zeros',
                kernel_regularizer = R.l2(L2_reg),
                activity_regularizer = R.l1(L1_reg),
                activation='relu'))
#first hidden layer to second hidden layer with dropout
model.add(Dropout(drop_out))
model.add(Dense(n_hidden, 
                kernel_initializer='random_uniform',
                use_bias = True,
                bias_initializer='zeros',
                kernel_regularizer = R.l2(L2_reg),
                activity_regularizer = R.l1(L1_reg),
                activation='relu'))

#2nd hidden layer to output layer with dropout
model.add(Dropout(drop_out))
model.add(Dense(n_outputs, 
                kernel_initializer='random_uniform',
                use_bias = True,
                bias_initializer='zeros',
                kernel_regularizer = R.l2(L2_reg),
                activity_regularizer = R.l1(L1_reg),
                activation='sigmoid'))

#prepare optimizer
sgd = SGD(lr = alpha_lr, decay = 0., momentum = alpha, nesterov=False)
#sgd = SGD(lr = alpha_lr)
            #lr: float >= 0. Learning rate.
            #momentum: float >= 0. Parameter updates momentum.
            #decay: float >= 0. Learning rate decay over each update.
            #nesterov: boolean. Whether to apply Nesterov momentum.
#model.summary()
#build model

start_train = time.clock()
model.compile(loss='binary_crossentropy',  optimizer=sgd)

#train model using training data set
model.fit(x_train, y_train,
          epochs=n_epochs,
          batch_size= n_batch)

'''
print "Model trained....testing model.."

#load test data
#Remember to check the sheet number of the data if using the same file
#Normal file is Xtest = 0 and Ytest =1, otherwise Xtest = 2, Ytest = 3
x_test = load_file(test_file_name,2, "Input Test")
y_test = load_file(test_file_name,3, "Output Test")

N_test = len(x_test) # Number of test samples minus header

Xtest_list = x_test.tolist()

score = model.evaluate(x_test, y_test, batch_size=128)
'''
out = (model.predict_proba(x_train) > .5) + 0.
tot_err = np.absolute(out - y_train)
tot_err_per = np.sum(tot_err)/(N*n_outputs)

end_train = time.clock()    
if (end_train-start_train > 60):
    print"\n Trained in {0:.2f} minutes with {1:6i} data points".format((end_train-start_train)/60., N)
else:
    print"\n Trained in {0:.2f} seconds with {1} data points".format((end_train-start_train)/1.,N)
    #print" with", N, "data points"

print " Error = {0:.2f}%".format(tot_err_per * 100)
