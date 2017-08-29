import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import copy
import easygui 
from seasonal import fit_seasons, adjust_seasons
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Put training and test data file in the same folder as py code
#Training data file name
########### Set name of data file
title = 'Choose file with data table to format...'
data_file_name = easygui.fileopenbox(title)
test_file_name = data_file_name
print('Loading data file ' + data_file_name)
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
    # stop clock
    end = time.clock() 
    
    if (end-start > 60):
        print type_data, "data loaded in {0:.1f} minutes".format((end-start)/60.)
    else:
        print type_data, "data loaded in {0:.1f} seconds".format((end-start)/1.)
    
    Data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
       
    return Data 

def cleanzero(X):
#convert low values to zero
    
    limit = 10e-4  #set limit where less than = 0
    
    Clean = (np.absolute(X) > limit) + 0.  #create vector of 1's and 0's
    Xclean = np.multiply(X,Clean) #elementwise multiply to convert low limits to 0
    
    return Xclean

##################################################
#########Execute

print "Loading training data...."
#Sheet 0 = input X data
#Sheet 1 = input Y data

##############ask if there are any initial columns to ignore
try:
    d = int(raw_input("Which column does data start (Default = 0)? "))
except ValueError:
    d = 0

#################SELECT Worksheets    
x_train = load_file(data_file_name,0,"Input Training")
x_train = x_train[:,d:]

y_train = load_file(data_file_name,1,"Output Training")

#load test data
#Remember to check the sheet number of the data if using the same file
#Normal file is Xtest = 0 and Ytest =1, otherwise Xtest = 2, Ytest = 3
x_test = load_file(test_file_name,4, "Input Test")
x_test = x_test[:,d:]

y_test = load_file(test_file_name,5, "Output Test")

print "\nBuilding model...\n"
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 10., C=1.0)
ranforest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
tree = DecisionTreeClassifier(criterion="entropy")
ada = AdaBoostClassifier(base_estimator = tree, 
                         n_estimators = 500, 
                         learning_rate = 0.1, 
                         random_state=0)

classifier = tree

rows_tot, n_inputs = np.shape(x_train) # number of input rows and columns
n_outputs = np.shape(y_train)[1] # of output columns

count = 0
cum_acc = 0
train_length = 4380 # of hours of input training data to continuously train on
test_length = 24

for traincount in range(train_length,rows_tot-test_length,test_length): ##full training set
#for traincount in range(0,100,test_length):   ## partial training set
    train_start = traincount - train_length
    train_end = traincount 
    test_start = train_end + 1
    test_end = test_start + test_length
    
    ### set up training pairs for moving training
    x_trainpart = x_train[train_start:train_end,:]
    y_trainpart = y_train[train_start:train_end,0]
    
    x_testpart = x_train[test_start:test_end,:]
    y_testpart = y_train[test_start:test_end,0]
    
    if y_testpart.sum()>0:  
        classifier.fit(x_trainpart,y_trainpart)
        y_classifier = classifier.predict(x_testpart)
        
        ada.fit(x_trainpart,y_trainpart)
        y_classifier = classifier.predict(x_testpart)
        
        print accuracy_score(y_classifier, y_testpart), traincount
        cum_acc += accuracy_score(y_classifier, y_testpart)
        count += 1
    else:
        pass
print'\nAverage accuracy',cum_acc/count
    
    
    
        