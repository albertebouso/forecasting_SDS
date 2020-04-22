from neural_network import *
from keras.models import load_model
import os

model_list = os.listdir('model')
# selected = [model_list[x] for x in [1,2,3,4,5]]
selected = model_list[-3]
print(selected)

np.random.seed(seed=0)
config = {'do_train': False,
          'path_to_model': 'model/' + selected + '/',
          'start_date_training': dt.datetime(2016, 1, 1, 0, 0),
          'stop_date_training': dt.datetime(2017, 12, 30, 23, 45),
          'plot_specific_time': False
}


def getTimeSeriesTraining(x, lag_days, day_divisions, is_week):
    """

    :param x: data
    :param lag_days: number of previous days used for training in order to predict next step
    :param day_divisions: number of division one day has in the selected timefreme, e.g. (24 if hourly, 96 if 15m)
    :return:
    """
    n_inputs = int((len(x) - lag_days * day_divisions) / day_divisions)

    if is_week:
        n_inputs = 1

    input_feature = np.empty([n_inputs, lag_days * day_divisions])*0
    a = 0
    b = lag_days * day_divisions

    for i in range(n_inputs):
        input_feature[i, :] = x[a:b]
        a += day_divisions
        b += day_divisions

    return input_feature


def normalize(data, data_max, data_min):

    data_n = (data - data_min) / (data_max - data_min)

    return data #data_n


def generate_input(data_imported, start_time, end_time, lag_days, n_div_per_day):
    start = start_time
    end = end_time
    X = data_imported['belpex'][start:end].values
    solar_max = 2380
    solar_min = 0
    wind_max = 2320
    wind_min = 0
    belpex_max = 156
    belpex_min = -42
    X = normalize(X, belpex_max, belpex_min)

    is_week = len(X) <= n_div_per_day*lag_days
    if is_week:
        X1 = getTimeSeriesTraining(X, lag_days, n_div_per_day, is_week)
        start += dt.timedelta(days=(lag_days-1))
        # end -= dt.timedelta(days=1)
        X2 = data_imported['solar'][start:end].values.reshape(-1, n_div_per_day)
        X3 = data_imported['wind'][start:end].values.reshape(-1, n_div_per_day)
    else:
        X1 = getTimeSeriesTraining(X, lag_days, n_div_per_day, is_week)
        start += dt.timedelta(days=(lag_days))
        # end -= dt.timedelta(days=1)
        X2 = data_imported['solar'][start:end].values.reshape(-1, n_div_per_day)
        X3 = data_imported['wind'][start:end].values.reshape(-1, n_div_per_day)

    X2 = normalize(X2, solar_max, solar_min)
    X3 = normalize(X3, wind_max, wind_min)

    X = np.hstack([X1, X2, X3])
    return X


def forecast_from_week(model, test_set, lag_days, n_div_per_day, plot_title, path):
    start = dt.datetime(2018, 1, 1, 0, 0)
    end = dt.datetime(2018, 1, 7, 23, 45)
    t_week = np.linspace(1, 24*8, 24*8)
    t_pred = np.linspace(24*7+1, 24*8, 24)

    week = generate_input(test_set, start, end, lag_days, n_div_per_day)
    prediction = predict_next_day(model, week)
    end = dt.datetime(2018, 1, 8, 23, 45) # NEW
    belpex = test_set['belpex'][start:end].resample('1H').mean().values.reshape(-1, 24*8)

    if not os.path.exists('{}results/'.format(path)):
        os.mkdir('{}results/'.format(path))

    full_path = '{}results/{}'.format(path, plot_title)
    np.savetxt(full_path, prediction, delimiter=",")

    t_8th = t_week[167:]
    t_week = t_week[0:168]
    # end_week = dt.datetime(2018, 1, 7, 23, 45)

    # connect last real point with prediction
    t_pred = np.append(t_week[-1], t_pred)
    prediction = np.append(belpex[0][167], prediction)

    plt.figure()
    plt.title(plot_title)
    plt.plot(t_week, belpex[0][0:168].flatten(), color='blue', label='Real')
    plt.plot(t_8th, belpex[0][167:].flatten(), '--', color='blue', label='Real', alpha=0.5)
    plt.plot(t_pred, prediction, '-', color='red', label='Forecast')
    plt.legend(frameon=False)
    plt.grid(True)

    prediction = prediction[1:]
    sum_e2 = sum(((belpex[0][168:]-prediction)**2))

    return prediction, sum_e2


# Initial conditions
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2017, 12, 31)
start_week = dt.datetime(2018, 1, 1)
end_week = dt.datetime(2018, 1, 9)

# Import data
training, test_1, test_2, test_3 = load_data(start_date, end_date, start_week, end_week)

# Input and output data
n_div_per_day = 24*4
lag_days = 7

# Inputs
start = config['start_date_training']
end = config['stop_date_training']
X = generate_input(training, start, end, lag_days, n_div_per_day)

# Outputs
start += dt.timedelta(days=lag_days)
# Y = training['belpex'][start:end].resample('1H').mean().values.reshape(-1, n_div_per_day)
Y = training['belpex'][start:end].values.reshape(-1, n_div_per_day)

if config['do_train']:
    model, path = create_nn_model(n_div_per_day, X, Y)
    test_model(model, X, Y)
else:
    path = config['path_to_model']
    model = load_model('{}model'.format(path))

if config['plot_specific_time']:
    start = dt.datetime(2017, 1, 1, 0, 0)
    end = dt.datetime(2017, 12, 30, 23, 45)
    x_val = generate_input(training, start, end, lag_days, n_div_per_day)

    start += dt.timedelta(days=lag_days)
    y_val = training['belpex'][start:end].values.reshape(-1, n_div_per_day)

    test_model(model, x_val, y_val)

# Predict data for weeks
pred_w1, sum_e2_w1 = forecast_from_week(model, test_1, lag_days, n_div_per_day, 'Week 1', path)
pred_w2, sum_e2_w2 = forecast_from_week(model, test_2, lag_days, n_div_per_day, 'Week 2', path)
pred_w3, sum_e2_w3 = forecast_from_week(model, test_3, lag_days, n_div_per_day, 'Week 3', path)

rmse = (sum([sum_e2_w1, sum_e2_w2, sum_e2_w3])/24/3)**(0.5)
print(rmse)

with open('{}info.txt'.format(path), 'a') as file:
    file.write('\nrmse in test weeks: {}'.format(rmse))
    file.close()

forecasts = np.concatenate((pred_w1, pred_w2, pred_w3))
full_path = '{}results/{}'.format(path, 'Forecast_3_weeks.csv')
np.savetxt(full_path, forecasts, delimiter=",")

plt.show()
