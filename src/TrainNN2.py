#!/usr/bin/python
import numpy as np
import pandas as pd
import pickle
import sys
dir_path = '.'
# sys.path.insert(0, dir_path+"/pyrenn-master/python/")
# import pyrenn

# *********** sigmoid
def sigmoid(x):
	return 1/(1+np.exp(-x))

# *********** derivative of sigmoid
def sigmoid_der(x):
	return x*(1-x)

# *********** to find weights in the neural network
def train_neural_network(train_set, train_result_set, num_steps, num_neurons):
	num_hidden_layers = len(num_neurons)
	np.random.seed(4)
	weights0 = 0*np.random.random((num_neurons[0], num_neurons[1])) 
	weights1 = 0*np.random.random((num_neurons[1], num_neurons[2])) 
	weights2 = 0*np.random.random((num_neurons[2], num_neurons[3])) 
	layer0 = train_set
	# print(len(train_set))
	# print(len(train_result_set))
	for i in range(num_steps):
		# print(i)
		# print('weights0')
		# print(weights0)
		# print('weights1')
		# print(weights1)
		# print('weights2')
		# print(weights2)
		layer1 = sigmoid(np.dot(layer0, weights0))
		# print('layer1')
		layer2 = sigmoid(np.dot(layer1, weights1))
		# print('layer2')
		# print(layer2)
		layer3 = sigmoid(np.dot(layer2, weights2))
		# print('layer3')
		# print(layer3)
		layer3_error  = (train_result_set - layer3)
		if((i %50) == 0):
			print(i)
			# print(weights0)
			print(np.mean(np.abs(layer3_error)))
		layer3_delta = layer3_error*sigmoid_der(layer3)
		layer2_error = layer3_delta.dot(weights2.T)
		layer2_delta = layer2_error*sigmoid_der(layer2)
		layer1_error = layer2_delta.dot(weights1.T)
		layer1_delta = layer1_error*sigmoid_der(layer1)
		# update weights
		weights2 = weights2 + 0.01*layer2.T.dot(layer3_delta)
		weights1 = weights1 + 0.01*layer1.T.dot(layer2_delta)
		weights0 = weights0 + 0.01*layer0.T.dot(layer1_delta)
	return weights0, weights1, weights2


# *********** main
csv_train = pd.read_csv(dir_path+'/small_train_data.csv', delimiter=",", low_memory=False)
print('Read train data')
train = csv_train.drop(csv_train.columns[-1], axis=1)
train = train.drop(train.columns[0], axis=1)
train_result_str = csv_train.drop(csv_train.columns[0:-1], axis=1)
# print(train_result_str)
# exit()
# print(train_result_str.iloc[0].genre)
# exit()
num_features = train.shape[1] 
num_data = train.shape[0]
num_genre = 10

train_result = [ [0] * 10 for _ in range(num_data)]
for i in range(num_data):
	if(train_result_str.iloc[i].genre == 'classic pop and rock'):
		train_result[i][0] = 1
	elif(train_result_str.iloc[i].genre == 'dance and electronica'):
		train_result[i][1] = 1
	elif(train_result_str.iloc[i].genre == 'jazz and blues'):
		train_result[i][2] = 1
	elif(train_result_str.iloc[i].genre == 'folk'):
		train_result[i][3] = 1
	elif(train_result_str.iloc[i].genre == 'soul and reggae'):
		train_result[i][4] = 1
	elif(train_result_str.iloc[i].genre == 'classical'):
		train_result[i][5] = 1
	elif(train_result_str.iloc[i].genre == 'metal'):
		train_result[i][6] = 1
	elif(train_result_str.iloc[i].genre == 'punk'):
		train_result[i][7] = 1
	elif(train_result_str.iloc[i].genre == 'pop'):
		train_result[i][8] = 1
	elif(train_result_str.iloc[i].genre == 'hip-hop'):
		train_result[i][9] = 1

print('Read '+ str(num_data)+' values with '+str(num_features) + ' features')

train_result = np.array(train_result)
# print(train.values)

train = np.array(train.values)

# exit()

train_norm = (train - train.min(0)) / train.ptp(0)

num_neurons = []
num_neurons.append(num_features)
num_neurons.append(2*num_features)
num_neurons.append(2*num_features)
num_neurons.append(num_genre)

num_steps = 1000
print('Training neural network for '+str(num_steps)+' iterations')
weights0, weights1, weights2 = train_neural_network(train_norm, train_result, num_steps, num_neurons)
with open('weights2.txt', 'wb') as f:
	pickle.dump([weights0, weights1, weights2, train.min(0), train.ptp(0)], f)
print('Saved weights and normalizing factors for features in the file weights2.txt in binary format')


# print(num_genre)
# exit()
# nn = [num_features, (3*num_features), (3*num_features), num_genre]
# net = pyrenn.CreateNN(nn, dIn=[0],dIntern=[ ], dOut=[1])

# train = train.drop(train.columns[0], axis=1)
# P = np.array(train.values)
# P = P.T
# train_result = pd.get_dummies(train_result)
# Y = np.array(train_result.values)
# Y = Y.T
# pyrenn.train_LM(P, Y, net, k_max=2, E_stop=1e-10, dampfac=3.0, dampconst=10.0, verbose = True)

