import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import datetime as dt
import io
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_predict, KFold


def _get_dataframe(wind, solar, price):
    """
    Function which returns a unique dataframe. Useful commands
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


def _normalised_data(data, column):
    """
    Function to remove outliers, based on a genereic method from statistics independent of timeseries

    :param data:

    :return:
    """
    col = data.loc[column]
    mean = col.mean()
    std = col.std()
    n_std = 5
    data[column][(col >= mean + n_std * std)] = mean + n_std * std
    data[column][(col <= mean + n_std * std)] = mean + n_std * std

    data_normalised = data
    return data_normalised


def _create_bloxplots(data, column, attribute):
    """
    this function returns a plot of the dataset in a box plot form, grouped by different attributes

    :param data: dataframe with with 3 columns and datatime index, based on price index
    :param column: string with the data to be analysed
    :param attribute: one of the following strings: week_days, month_days, hours, months and years

    :return:
    """
    data[attribute] = data.index.weekday
    data.boxplot(column=column, by=attribute)


def _get_accuracy(x, y):
    """
    metric to define the quality of the forecast

    :param x:
    :param y:

    :return:
    """
    return np.mean(np.abs(x - y)) / np.mean(x)


def _csv_to_data(path, energy):
    """
    Function for transforming CSV files into pandas dataframe, with two columns: time and value

    :param path: string, relative path to the file
    :param energy: boolean, True if set is energy, False if else

    :return: table in pandas dataframe format
    """
    df = pd.read_csv(path, header=0)
    if not df.columns[0]:
        df = df.rename(columns={'Unnamed: 0': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    if energy:
        df = df.LoadFactor

    return df


def _correct_hours(df, start, end):
    """
    returns dataframe with price the same as previous hour when data is missing

    :param df: dataframe that needs to be corrected
    :param start: start date of the dataset
    :param end: end date of the dataset

    :return:
    """
    if df.columns[0] == 'time':
        dates = pd.date_range(start=start, end=end, freq='1H')
        for d in dates:
            try:
                p = df.loc[d]
            except KeyError:
                df.loc[d] = df.loc[d - dt.timedelta(hours=1)]
        df = df.sort_index()
        df = df[start:end]

    return df


def _correct_NaN(df):
    """
    Overcomes the problem of Not a Number values by filling the previous valid observation until the next valid one

    :param df: dataframe that needs to be corrected

    :return:
    """
    df = df.fillna(method='pad')

    return df


def _resample_granularity(df):
    """
    Resample the dataframe on a 15 minutes basis
    :param df:
    :return:
    """
    if df.columns[0] == 'time':
        df = df.resample('15T').pad()
    elif df.columns[0] == 0:
        df_temp = pd.DataFrame()
        for r in range(len(df)):
            df1 = df.iloc[r]
            for n in range(3):
                df_temp = df_temp.append(df1)

            df = df_temp
    return df


def _csv_to_dataframe(solar_path, wind_path, price_path, start, end):
    """
    Function for import, treat and create unique dataframe

    :param solar_path: Relative path towards solar data
    :param wind_path: Relative path towards wind data
    :param price_path: Relative path towards price data
    :param start: Start data
    :param end: End data

    :return:
    """
    df_solar = _csv_to_data(solar_path, True)
    df_wind = _csv_to_data(wind_path, True)
    df_belpex = _csv_to_data(price_path, False)

    df_belpex = _correct_hours(df_belpex, start, end)
    df_belpex = _resample_granularity(df_belpex)
    df_wind = _correct_NaN(df_wind)

    data = _get_dataframe(df_wind, df_solar, df_belpex)

    return data


# Initial Conditions
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2017, 12, 31)

data = _csv_to_dataframe('data/solar_20162017.csv', 'data/wind_20162017.csv', 'data/belpex_20162017.csv', start_date, end_date)
# week_1 = _csv_to_dataframe('data/solar_week_1.csv', 'data/wind_week_1.csv', 'data/belpex_week_1.csv', start_date, end_date)
# week_2 = _csv_to_dataframe('data/solar_week_2.csv', 'data/wind_week_2.csv', 'data/belpex_week_2.csv', start_date, end_date)
# week_3 = _csv_to_dataframe('data/solar_week_3.csv', 'data/wind_week_3.csv', 'data/belpex_week_3.csv', start_date, end_date)

print(week_1.head(10))
print(week_2.tail(10))
