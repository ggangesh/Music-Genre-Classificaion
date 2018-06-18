#!/usr/bin/python
import numpy as np
import pandas as pd
import pickle


# *********** main
dir_path = '../data'
csv = pd.read_csv(dir_path+'/SongCSV.csv', delimiter=",", low_memory=False)
print('Read all features')
csv = csv.sample(frac=1)
num_rows, num_cols = csv.shape
num_rows_train = int((num_rows*4)/5)
csv_train = csv.copy()
csv_train.drop(csv_train.index[num_rows_train:], inplace=True)
csv_train.to_csv('train_data.csv', sep=',', encoding='utf-8', index=False)
print('Extracted first 80 percent data to train_data.csv')
csv_test = csv.copy()
csv_test.drop(csv_test.index[0:num_rows_train], inplace=True)
csv_test.to_csv('test_data.csv', sep=',', encoding='utf-8', index = False)
print('Extracted last 20 percent data to test_data.csv')
