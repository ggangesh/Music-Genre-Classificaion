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

# *********** to find output corresponding to the test data set
def find_neural_output(test_set, weights0, weights1, weights2, num_neurons, num_data):
	# print(len(test_set))
	layer0 = test_set
	layer1 = sigmoid(np.dot(layer0, weights0))
	layer2 = sigmoid(np.dot(layer1, weights1))
	layer3 = sigmoid(np.dot(layer2, weights2))
	layer3_str = []
	
	for i in range(num_data):
		print(layer3)
		max_index = layer3[i].argmax(axis=0)
		print(max_index)
		if(max_index == 0):
			layer3_str.append(['classic pop and rock'])
		elif(max_index == 1):
			layer3_str.append(['dance and electronica'])
		elif(max_index == 2):
			layer3_str.append(['jazz and blues'])
		elif(max_index == 3):
			layer3_str.append(['folk'])
		elif(max_index == 4):
			layer3_str.append(['soul and reggae'])
		elif(max_index == 5):
			layer3_str.append(['classical'])
		elif(max_index == 6):
			layer3_str.append(['metal'])
		elif(max_index == 7):
			layer3_str.append(['punk'])
		elif(max_index == 8):
			layer3_str.append(['pop'])
		elif(max_index == 9):
			layer3_str.append(['hip-hop'])
		else:
			layer3_str.append(['not found'])
	return layer3_str

with open('weights2.txt', 'rb') as f:  # Python 3: open(..., 'rb')
    weights0, weights1, weights2, nf1, nf2 = pickle.load(f)
print('Read weights and normalizing factors')
dir_path = '.'
# csv = pd.read_csv(dir_path+'/train.csv', delimiter=",")
csvtest = pd.read_csv(dir_path+'/small_test_data.csv', delimiter=",", low_memory=False)
print('Read test data')
test = csvtest.drop(csvtest.columns[-1], axis=1)
test = test.drop(test.columns[0], axis=1)
test_result_str = csvtest.drop(csvtest.columns[0:-1], axis=1)

num_features = test.shape[1] 
num_data = test.shape[0]
num_genre = 10

test_result = [ [0] * 10 for _ in range(num_data)]
for i in range(num_data):
	if(test_result_str.iloc[i].genre == 'classic pop and rock'):
		test_result[i][0] = 1
	elif(test_result_str.iloc[i].genre == 'dance and electronica'):
		test_result[i][1] = 1
	elif(test_result_str.iloc[i].genre == 'jazz and blues'):
		test_result[i][2] = 1
	elif(test_result_str.iloc[i].genre == 'folk'):
		test_result[i][3] = 1
	elif(test_result_str.iloc[i].genre == 'soul and reggae'):
		test_result[i][4] = 1
	elif(test_result_str.iloc[i].genre == 'classical'):
		test_result[i][5] = 1
	elif(test_result_str.iloc[i].genre == 'metal'):
		test_result[i][6] = 1
	elif(test_result_str.iloc[i].genre == 'punk'):
		test_result[i][7] = 1
	elif(test_result_str.iloc[i].genre == 'pop'):
		test_result[i][8] = 1
	elif(test_result_str.iloc[i].genre == 'hip-hop'):
		test_result[i][9] = 1

print('Read '+ str(num_data)+' values with '+str(num_features) + ' features')

test_result = np.array(test_result)
test = np.array(test.values)

test_norm = (test - nf1) / nf2

num_neurons = []
num_neurons.append(num_features)
num_neurons.append(2*num_features)
num_neurons.append(2*num_features)
num_neurons.append(num_genre)

print('Finding final results')
final_result = find_neural_output(test_norm, weights0, weights1, weights2, num_neurons, num_data)


test_indices = csvtest.drop(csvtest.columns[1:], axis=1)
indices = test_indices.values

# print(len(indices))
# print(len(final_result))
result_output = np.concatenate((indices, final_result), axis=1)
# print(result_output)
fmt='%s,%s'
np.savetxt('predictions2.csv', result_output, fmt=fmt, header="id,salary", comments='')

exit()