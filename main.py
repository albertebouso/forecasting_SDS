import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import datetime as dt
import io
import os
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from data_manager import load_data
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop, SGD


def getTimeSeriesTraining(x, lag, day_divisons):
    n_inputs = int(len(x)/day_divisons)
    input_feature = np.empty([n_inputs, lag])
    cnt = 0

    for i in range(n_inputs - 7):
        a = i*cnt
        b = lag + i*cnt
        day_7 = x[a:b]
        input_feature[i, :] = day_7
        cnt = + day_divisons

    return input_feature


# Initial conditions
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2017, 12, 31)
start_week = dt.datetime(2018, 1, 1)
end_week = dt.datetime(2018, 1, 8)

# Import data
training, test_1, test_2, test_3 = load_data(start_date, end_date, start_week, end_week)

# Input and output data
n_hours = 24
lag = 168

start = dt.datetime(2017, 1, 1, 0, 0)
end = dt.datetime(2017, 12, 23, 23, 45)
X = training['belpex'][start:end].resample('1H').mean().values
print(X.shape)

X = getTimeSeriesTraining(X, lag, 24)

start = dt.datetime(2017, 1, 8, 0, 0)
end = dt.datetime(2017, 12, 30, 23, 45)
Y = training['belpex'][start:end].resample('1H').mean().values.reshape(-1, n_hours)

print(training['belpex'][start:end].resample('1H').mean().shape)
print('input features ' + str(X.shape))
print('target dimensions ' + str(Y.shape))
