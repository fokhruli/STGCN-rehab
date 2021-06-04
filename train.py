import numpy as np
random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from GCN.sgcn_lstm import Sgcn_Lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""import the whole dataset"""
data_loader = Data_Loader("Kimore ex3")  # folder name -> Train.csv, Test.csv

"""import the graph data structure"""
graph = Graph(len(data_loader.body_part))

"""Split the data into training and validation sets while preserving the distribution"""
train_x, test_x, train_y, test_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)

"""Train the algorithm"""
algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr = 0.0001, epoach = 1000, batch_size=10)
history = algorithm.train()

"""Test the model"""
y_pred = algorithm.prediction(test_x)
y_pred = data_loader.sc2.inverse_transform(y_pred)
test_y = data_loader.sc2.inverse_transform(test_y) 

"""Performance matrix"""
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
test_dev = abs(test_y-y_pred)
mean_abs_dev = np.mean(test_dev)
mae = mean_absolute_error(test_y, y_pred)
rms_dev = sqrt(mean_squared_error(y_pred, test_y))
mse = mean_squared_error(test_y,y_pred) 
mape = mean_absolute_percentage_error(test_y, y_pred)
print('Mean absolute deviation:', mae)
print('RMS deviation:', rms_dev)
print('MSE:', mse)
print('MAPE: ', mape)
