#!/usr/bin/python
import numpy
import pandas as pd
import pickle
import sys
dir_path = '.'
sys.path.insert(0, dir_path+"/pyrenn-master/python/")
import pyrenn

# *********** main
csv_train = pd.read_csv(dir_path+'/small_train_data.csv', delimiter=",", low_memory=False)
print('Read train data')
train = csv_train.drop(csv_train.columns[-1], axis=1)
train_result = csv_train.drop(csv_train.columns[0:-1], axis=1)

num_features = train.shape[1]
num_data = train.shape[0]
num_genre = len((train.iloc[:,-1]).unique())
print('Read '+ str(num_data)+' values with '+str(num_features) + 'features')
num_features -=1 
nn = [num_features, (3*num_features), (3*num_features), 10]
net = pyrenn.CreateNN(nn, dIn=[0],dIntern=[ ], dOut=[1])

train = train.drop(train.columns[0], axis=1)
P1 = numpy.array(train.values)
P = (P1 - P1.min(0))/P1.ptp(0)#
#joined_new_norm = (joined_new - joined_new.min(0)) / joined_new.ptp(0)
P = P.T
train_result = pd.get_dummies(train_result)
Y = numpy.array(train_result.values)
Y = Y.T
print(Y.shape)
print(num_features)
print(P.shape)
pyrenn.train_LM(P, Y, net, k_max=1000, E_stop=1e-10, dampfac=3.0, dampconst=10.0, verbose = True)

