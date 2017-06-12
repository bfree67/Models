from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import Adagrad
import numpy as np
import time
import pandas as pd
from pandas import ExcelWriter
import copy

#Training data file name
data_files = ["outputDates.xlsx", "outputDatesO3.xlsx", "outputDatesFAH.xlsx",
              "outputDatesJAR.xlsx","outputDatesMAN.xlsx"]
#data_file_name = data_files[1]
timesteps = 12
#num_classes = 2
newpath = 'C:\\\TARS\Phd\Keras\\'   #example output path
#optimizer_type = 'nadam'
activation_type = 'relu'
drop_out = .15

### - Load data from excel file function
def load_file(datafile,worksheet=0,type_data="Input Training"):
    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0,         #assumes 1st row are header names
                skiprows=None, 
                skip_footer=0, 
                index_col=0,    #first column is index
                parse_cols=None, #default = None
                parse_dates=False, 
                date_parser=None, 
                na_values=None, 
                thousands=None, 
                convert_float=True, 
                has_index_names=None, 
                converters=None, 
                engine=None)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
    # stop clock
    end = time.clock()    
    if (end-start > 60):
        print type_data, "data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print type_data, "data loaded in {0:.2f} seconds".format((end-start)/1.)
    return data

def ThreeDdata(data):
    '''#reshape a 2D dataset into a super 2D with time steps and then into a 3D matrix
    #where (total sample rows, delay, feature size)
    #a data set of 500 samples with 6 features (500 x 6 = 3,000 elements) that has 3 time inputs
    #t = 0, t = -1, t = -2 will first be transformed into a 497 x 9 matrix
    #and have a dimension of (497, 3, 6) or 8,946 elements - have to remove the last 
    #rows to keep dimensionality as you shift columns up
    '''
    n,p = np.shape(data)    # get dimensions of data matrix 
    a = np.zeros((n,p*timesteps))  # initialize a matrix with multiples based on total delay
    dataX = copy.copy(data[0:(n-timesteps),:])  #deep copy the data
    for i in range(1,timesteps):
        a = data[i:(n-timesteps+i), :]       # create delayed matrix for each increment
        dataX = np.concatenate((dataX, a), axis = 1)
    dataXre=np.reshape(dataX,((n-timesteps),timesteps,p))   
    return dataXre

#########Select data source - Use random data for architecture development
data_select = raw_input('Use random data (y) or imported data (n)? ')
if data_select == 'y':
    ######### Generate dummy training data if yes
    data_dim = 13  #set dimensions for data
    num_classes = 5
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = (np.random.random((1000, num_classes)) > .5) + 0.

    x_test = np.random.random((100, timesteps, data_dim))
    y_test = (np.random.random((100, num_classes)) > .5) + 0.
else:
    #### import data from file if No. Convert input data into 3D matrices
    print"\nWhich data file would you like to use?"  # print list of files
    for data_i in range (len(data_files)):
        print data_i, data_files[data_i]
    d_num = raw_input('Choose a number: ')
    if d_num > len(data_files): #if out of range, use a default
        d_num = 0
        print"\nValue out of range. Using {0} for data".format(data_files[0])
    data_file_name = data_files[d_num]
    
    x_train = ThreeDdata(load_file(data_file_name,0,"Input Training"))
    n_train = len(x_train)
    y_train = load_file(data_file_name,1,"Output Training")
    y_train = y_train[0:n_train,:] 

    x_test = ThreeDdata(load_file(data_file_name,4,"Input Test"))
    n_test = len(x_test)
    y_test = load_file(data_file_name,5,"Output Test") 
    y_test = y_test[0:n_test,:]

data_dim = len(x_test.T) 
num_classes = len(y_test.T)

############### Build RNN model
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
layer_out = len(x_train.T) * 2

######### create RNN model with 2 LSTM layers and 1 FF layer with dropout between layers
# setup LSTM layers
model.add(LSTM(layer_out, return_sequences=True,
               input_shape=(timesteps, data_dim),
               bias_initializer="zeros",
               activation=activation_type))  # returns a sequence of vectors of dimension 32
model.add(Dropout(drop_out))
'''
model.add(LSTM(layer_out, return_sequences=True,
               activation=activation_type))  # returns a sequence of vectors of dimension 32
model.add(Dropout(drop_out))
'''
model.add(LSTM(layer_out, activation=activation_type)) # return a single vector of dimension 32
model.add(Dropout(drop_out))

#add feed forward layer
model.add(Dense(num_classes,
                use_bias = True, bias_initializer="zeros",
                activation='sigmoid'))

optimizer_type = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer_type,
              metrics=['accuracy'])

print"\nModel prepared using {0} samples with {1} features from file {2}".format(len(x_train),data_dim,data_file_name)
print"\nThere are {0} outputs.\n".format(num_classes)
########## function to fit model with data based on epochs
def mod_fit(x_train, y_train, n_epochs):
    model.fit(x_train, y_train, batch_size=24, epochs=n_epochs, shuffle = False)
    out = (model.predict_proba(x_test) > .5) + 0.
    tot_err = np.absolute(out - y_test)
    return tot_err.sum(axis=0)

#n_epochs = int(raw_input('Number of epochs: '))
n_outputs = len(y_train.T)
N = len(x_train)
################## Record error at different parameters
cErr = []
for n_epochs in range (1,20,2):  #set number of epochs (1 to n) to cycle through
    b = []
    tot_err_mod = mod_fit(x_train, y_train, n_epochs)    
    tot_err_per = np.sum(tot_err_mod)/(N*n_outputs)
    print "\n Composite Error = {0:.2f}% after {1} epochs".format(tot_err_per * 100, n_epochs)
    b.append(n_epochs)
    b.append(round(tot_err_per,5))
    b.append(tot_err_mod)
    cErr.append(b)
    
### make extension for output file
fname=data_file_name[-9:-1]
fname = fname[1:4]
filename = 'RNNFinalError'+fname+'_2.xlsx'    #example output file
writer = ExcelWriter(filename)
pd.DataFrame(cErr).to_excel(writer,'FinalError')

print'File saved in ', newpath + '\\' + filename
