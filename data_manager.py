import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def _csv_to_data(path, energy, week, start, end):
    """

    :param path: Relative path to the data
    :param energy: Boolean. True if the variable is energy, False if other
    :param week: Boolean. True if its week data, False if other
    :param start: start date of the dataset
    :param end: ned date of the dataset
    :return:
    """
    if week:
        df1 = pd.read_csv(path, header=0)

        if energy:
            df1 = df1.iloc[0:672]
            df1 = df1.LoadFactor
            df1 = df1.to_frame()
            df = pd.date_range(start, end, freq='15T', name='time')
            df = df[:-1]
            df = df1.set_index(df, 'time')
        else:
            df1 = df1.iloc[0:169]
            df = pd.date_range(start, end, freq='1H', name='time')
            df = df1.set_index(df, 'time')
            df = df.rename(columns={'0': 'price'})
    else:
        df = pd.read_csv(path, header=0)
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


def _resample_granularity(df, week):
    """
    resample the dataframe on a 15 minutes basis

    :param df: dataframe that needs to be granulated
    :return:
    """

    df = df.resample('15T').pad()
    if week:
        df = df[:-1]

    return df


def _correct_NaN(df):
    """
    overcomes the problem of Not a Number values by filling the previous valid observation until the next valid one

    :param df: dataframe that needs to be corrected

    :return:
    """
    df = df.fillna(method='pad')

    return df


def _get_dataframe(wind, solar, price):
    """
    function which combines into an unique dataframe.
    Useful commands:
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


def csv_to_dataframe(week, solar_path, wind_path, price_path, start, end):
    """
    function which converts 3 csv files, type week or other, into a unique variable

    :param week: Boolean. True if csv is a week, False if not
    :param solar_path: Relative path to solar data
    :param wind_path: Relative path to wind data
    :param price_path: Relative path to Belpex price
    :param start: Start of the period
    :param end: End of the period

    :return: datafrarme with 3 columns and datatime index, based on price index
    """
    if week:
        df_solar = _csv_to_data(solar_path, True, True, start, end)
        df_wind = _csv_to_data(wind_path, True, True, start, end)
        df_belpex = _csv_to_data(price_path, False, True, start, end)

        df_belpex = _correct_hours(df_belpex, start, end)
        df_belpex = _resample_granularity(df_belpex, True)
        df_wind = _correct_NaN(df_wind)

        df_wind = df_wind['LoadFactor']
        df_solar = df_solar['LoadFactor']

        data_compact = _get_dataframe(df_wind, df_solar, df_belpex)


    else:
        df_solar = _csv_to_data(solar_path, True, False, [], [])
        df_wind = _csv_to_data(wind_path, True, False, [], [])
        df_belpex = _csv_to_data(price_path, False, False, [], [])

        df_belpex = _correct_hours(df_belpex, start, end)
        df_belpex = _resample_granularity(df_belpex, False)
        df_wind = _correct_NaN(df_wind)

        data_compact = _get_dataframe(df_wind, df_solar, df_belpex)

    data_compact = _normalise_data(data_compact, 'belpex')

    plt.figure()
    data_compact.belpex.plot(grid=True)
    plt.show()

    return data_compact


def _normalise_data(data, column):
    """
    function for removing outliers, based on a generic method from statistics, independent of timeseries.

    :param data: dataframe with with 3 columns and datatime index, based on price index
    :param data: String with column name

    :return:
    """
    col = data[column]
    mean = col.mean()
    std = col.std()
    n_std = 5
    data[column][(col >= mean + n_std * std)] = mean + n_std * std
    data[column][(col <= mean - n_std * std)] = mean + n_std * std

    data_normalised = data
    return data_normalised


def _create_bloxplots(data, column, attribute):
    """
    Function for plotting the dataset in a box plot form, grouped by different attributes

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


def load_data(start_date, end_date, start_week, end_week):
    """
    global function for the file, runs all the functions to obtain the dataframe for each training and test

    :param start_date: Start of the training set
    :param end_date: End of the training set
    :param start_week: Start of the test set
    :param end_week: End of the test set

    :return: Four DataFrames: one training set and three test sets
    """
    # Obtain data in df format
    data = csv_to_dataframe(False, 'data/solar_20162017.csv', 'data/wind_20162017.csv', 'data/belpex_20162017.csv',
                            start_date, end_date)

    week_1 = csv_to_dataframe(True, 'data/solar_week_1.csv', 'data/wind_week_1.csv', 'data/belpex_week_1.csv',
                              start_week, end_week)
    week_2 = csv_to_dataframe(True, 'data/solar_week_2.csv', 'data/wind_week_2.csv', 'data/belpex_week_2.csv',
                              start_week, end_week)
    week_3 = csv_to_dataframe(True, 'data/solar_week_3.csv', 'data/wind_week_3.csv', 'data/belpex_week_3.csv',
                              start_week, end_week)

    return data, week_1, week_2, week_3




