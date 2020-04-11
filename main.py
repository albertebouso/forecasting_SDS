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

# Initial conditions
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2017, 12, 31)
start_week = dt.datetime(2018, 1, 1)
end_week = dt.datetime(2018, 1, 8)

# Import data
training, test_1, test_2, test_3 = load_data(start_date, end_date, start_week, end_week)



