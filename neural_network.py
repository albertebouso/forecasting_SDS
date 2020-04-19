import time
import os
from sklearn.model_selection import KFold
from data_manager import *
from keras.models import Sequential, save_model
from keras.layers.core import Dense
from keras.optimizers import RMSprop

seed = 0
np.random.seed(seed=seed)
n_splits = 2 #10
neurons = [60, 60] #[168, 108]
activation_functions = ['relu', 'linear']
shuffle = False
config = 'random seed = {} \nn_splits ={} \nneurons = {} \nactivation functions = {} \nshuffle = {}'.format(
    seed, n_splits, neurons, activation_functions, shuffle)


def create_nn_model(n_div_per_day, X, Y):
    '''
    Creates and trains the neural network model that is trained with the data given. It also plots the results of the
    training.
    :param n_div_per_day: number of divisions considered per day (e.g. 24 for hourly data or 96 for 15 minutes data)
    :param X: Dataset for training/test. It must be given as an array containing prices for seven days (accordingly
              to the n_div_per_day) appended with the forecast of wind and solar power of the day of interest. For
              hourly data a set of vectors of 216 values must be given (168 values for prices, 24 for wind power
              forecast and 24 for solar power forecast)
    :param Y: A set of vectors with the real belPEX prices for the day right after the week given in X
    :return:
    '''
    # Neural Network creation and training
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=X.shape[1], activation=activation_functions[0]))
    for i, num in enumerate(neurons):
        if i != 0:
            model.add(Dense(num, activation=activation_functions[0]))
    model.add(Dense(n_div_per_day, activation=activation_functions[1]))

    rprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-6)
    model.compile(loss='mean_squared_error', optimizer=rprop)

    i = 0
    mse = 10000

    # Cross validation training using KFold
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=shuffle).split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_training = model.fit(x_train, y_train, epochs=500, verbose=0)
        mse = y_training.history['loss'][-1]
        print('KFold counter:{}\nWith n_splits={}, training_set_length={}'.format(i+1, n_splits, len(train_index)))
        print('- mse is %.4f' % mse + ' @ ' + str(len(y_training.history['loss'])))
        i += 1

    path = 'model/model_{}_mse{:.2f}/'.format(time.strftime('%m%d-%H%M'),mse)
    os.mkdir(path)
    save_model(model, '{}model'.format(path), overwrite=False)
    file1 = open('{}info.txt'.format(path), 'x')
    file1.write(config)
    file1.close()

    return model, path


def test_model(model, X, Y):
    predict_nn = model.predict(X)

    yearY = np.array([])
    yearNN = np.array([])

    for y, pred in zip(Y, predict_nn):
        yearY = np.append(yearY, y)
        yearNN = np.append(yearNN, pred)

    plt.figure()
    plt.plot(yearY, color='blue', label='actual price')
    plt.plot(yearNN, color='red', label='forecast NN')
    plt.legend(frameon=False)
    plt.grid(True)

    error = yearY - yearNN
    cummulative_avg_error = np.array([])
    for i, value in enumerate(error):
        if i==0:
            cummulative_avg_error = np.append(cummulative_avg_error, value)
        else:
            cummulative_avg_error = np.append(cummulative_avg_error,
                                              (value+cummulative_avg_error[-1]*(len(cummulative_avg_error)-1)) /
                                              len(cummulative_avg_error))

    plt.figure()
    plt.plot(error, color='red', label='error (actual-forecast)')
    plt.plot(cummulative_avg_error, color='blue', label='avg error (actual-forecast)')
    plt.legend(frameon=False)
    plt.grid(True)

    plt.show()


def predict_next_day(model, x):
    y = model.predict(x)
    if len(y) == 24:
        prediction = y
    else:
        prediction = np.empty((24,))
        n_sections = y.shape[1]/24
        for i, value in enumerate(y[0]):
            index = int(np.floor(i/4))
            prediction[index] += value/n_sections

    return prediction