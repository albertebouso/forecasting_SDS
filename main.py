import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import datetime as dt
import io
from sklearn.model_selection import train_test_split, cross_val_predict, KFold


def _get_dataframe(wind, solar, price):
    """
    Function which returns a unique dataframe. Usefull commands
    data.head(10) prints the first 10 samples
    data.tail(10) prints the last 10 samples
    data.wind prints the column with label wind
    data.wind.plot plots the wind timeseries
    :param wind: MW produced of wind power
    :param solar: MW produced of PV power
    :param price: Eur/MWh spot price
    :return: datafrarme with 3 columns and datatime index, based on price index
    """
    info = {'belpex': price.values.flatten(), 'solar': solar.values, 'wind': wind.values}
    data = pd.DataFrame(index=price.index, data=info)

    return data


# Data treatment
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime(2017, 12, 31)

# Import solar, wind and prices data into pandas

df_solar = pd.read_csv('data/solar_20162017.csv', header=0)
df_solar = df_solar.rename(columns={'Unnamed: 0': 'time'})
df_solar['time'] = pd.to_datetime(df_solar['time'])
df_solar.set_index('time', inplace=True)
df_solar = df_solar.LoadFactor

df_wind = pd.read_csv('data/wind_20162017.csv', header=0)
df_wind = df_wind.rename(columns={'Unnamed: 0': 'time'})
df_wind['time'] = pd.to_datetime(df_wind['time'])
df_wind.set_index('time', inplace=True)
df_wind = df_wind.LoadFactor

df_belpex = pd.read_csv('data/belpex_20162017.csv', header=0)
df_belpex = df_belpex.rename(columns={'Unnamed: 0':'time'})
df_belpex['time'] = pd.to_datetime(df_belpex['time'])
df_belpex.set_index('time', inplace=True)

# Correct prices error by assuming they are the same as previous hour
dates = pd.date_range(start=start_date, end=end_date, freq='1H')
for d in dates:
    try:
        p = df_belpex.loc[d]
    except KeyError:
        df_belpex.loc[d] = df_belpex.loc[d-dt.timedelta(hours=1)]
df_belpex = df_belpex.sort_index()
df_belpex = df_belpex[start_date:end_date]


