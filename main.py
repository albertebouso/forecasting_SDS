from neural_network import *
from keras.models import load_model
np.random.seed(seed=0)


def getTimeSeriesTraining(x, lag_days, day_divisions, is_week):
    """

    :param x: data
    :param lag_days: number of previous days used for training in order to predict next step
    :param day_divisions: number of division one day has in the selected timefreme, e.g. (24 if hourly, 96 if 15m)
    :return:
    """
    n_inputs = int((len(x) - lag_days * day_divisions) / day_divisions)
    # if is_week:
    #     n_inputs = 1
    input_feature = np.empty([n_inputs, lag_days * day_divisions])
    a = 0
    b = lag_days * day_divisions

    for i in range(n_inputs):
        input_feature[i, :] = x[a:b]
        a += day_divisions
        b += day_divisions

    return input_feature


def generate_input(data_imported, start_time, end_time, lag_days, n_div_per_day):
    start = start_time
    end = end_time
    X = data_imported['belpex'][start:end].values

    is_week = len(X) <= n_div_per_day*lag_days
    if is_week:
        X1 = getTimeSeriesTraining(X, lag_days, n_div_per_day, is_week)
        start += dt.timedelta(days=(lag_days))
        # end -= dt.timedelta(days=1)
        X2 = data_imported['solar'][start:end].values.reshape(-1, n_div_per_day)
        X3 = data_imported['wind'][start:end].values.reshape(-1, n_div_per_day)
    else:
        X1 = getTimeSeriesTraining(X, lag_days, n_div_per_day, is_week)
        start += dt.timedelta(days=(lag_days))
        # end -= dt.timedelta(days=1)
        X2 = data_imported['solar'][start:end].values.reshape(-1, n_div_per_day)
        X3 = data_imported['wind'][start:end].values.reshape(-1, n_div_per_day)

    X = np.hstack([X1, X2, X3])
    return X


def forecast_from_week(model, test_set, lag_days, n_div_per_day, plot_title, path):
    start = dt.datetime(2018, 1, 1, 0, 0)
    end = dt.datetime(2018, 1, 7, 23, 45)
    t_week = np.linspace(1, 24*7, 24*7)
    t_pred = np.linspace(24*7+1, 24*8, 24)

    week = generate_input(test_set, start, end, lag_days, n_div_per_day)
    prediction = predict_next_day(model, week)
    belpex = test_set['belpex'][start:end].resample('1H').mean().values.reshape(-1, 24*7)

    full_path = '{}/results/{}'.format(path, plot_title)
    np.savetxt(full_path, prediction, delimiter=",")

    plt.figure()
    plt.title(plot_title)
    plt.plot(t_week, belpex.flatten(), color='blue', label='Real')
    plt.plot(t_pred, prediction, '--', color='red', label='Forecast')
    plt.legend(frameon=False)
    plt.grid(True)


# Initial conditions
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2017, 12, 31)
start_week = dt.datetime(2018, 1, 1)
end_week = dt.datetime(2018, 1, 8)

# Import data
training, test_1, test_2, test_3 = load_data(start_date, end_date, start_week, end_week)

# Input and output data
n_div_per_day = 24*4
lag_days = 7           # days

# Inputs
start = dt.datetime(2016, 1, 1, 0, 0)
end = dt.datetime(2017, 12, 30, 23, 45)
X = generate_input(training, start, end, lag_days, n_div_per_day)

# Outputs
print(end-start)
start += dt.timedelta(days=lag_days)
# Y = training['belpex'][start:end].resample('1H').mean().values.reshape(-1, n_div_per_day)
Y = training['belpex'][start:end].values.reshape(-1, n_div_per_day)

model, path = create_nn_model(n_div_per_day, X, Y)
# model = load_model('model/model_0419_1341__7.73')
test_model(model, X, Y)

# Predict data for weeks
forecast_from_week(model, test_1, lag_days, n_div_per_day, 'Week 1', path)
forecast_from_week(model, test_2, lag_days, n_div_per_day, 'Week 2', path)
forecast_from_week(model, test_3, lag_days, n_div_per_day, 'Week 3', path)

plt.show()
