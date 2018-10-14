from enum import Enum

import keras as k
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# General parameters
model_name = 'eurpln_model'
data_name = 'eurpln_d'
save_file = str.format('resources/models/{}.h5', model_name)
data_file = str.format('resources/data/{}.csv', data_name)
forecast_time_lags = 14

# Training parameters
test_size = 0.3
number_of_time_lags = 14
gru_units = 5
optimizer = k.optimizers.RMSprop(0.0001)
batch_size = 16
epochs = 2000
early_stopping_patience = 15

scaler = MinMaxScaler()


class Variable(Enum):
    MIN = 1
    MAX = 2
    OPEN = 3
    CLOSE = 4


def get_parsed_data(path, scaler, variable):
    data = p.read_csv(path)

    if variable == Variable.MIN:
        data['Value'] = data['Najwyzszy']

    if variable == Variable.MAX:
        data['Value'] = data['Najnizszy']

    if variable == Variable.OPEN:
        data['Value'] = data['Otwarcie']

    if variable == Variable.CLOSE:
        data['Value'] = data['Zamkniecie']

    data['Value'] = scaler.fit_transform(data['Value'].values.reshape(-1, 1))

    new_data = p.DataFrame({'Date': p.to_datetime(data['Data']), 'Value': data['Value']})
    return new_data


def get_time_lagged_data(data, number_of_time_lags):
    for selected_number_of_time_lag in range(number_of_time_lags):
        column_name = str.format('Value_{}', selected_number_of_time_lag + 1)
        data[column_name] = data['Value'].shift(selected_number_of_time_lag + 1)

    data = data.dropna()

    return data


def get_observations(data):
    observations_x = data.drop(columns=['Date', 'Value']).values
    observations_x_shape = observations_x.shape
    observations_x = observations_x.reshape(-1, 1, observations_x_shape[1])

    observations_y = data['Value'].values.reshape(-1, 1)

    return observations_x, observations_y


parsed_data = get_parsed_data(data_file, scaler, Variable.CLOSE)
time_lagged_data = get_time_lagged_data(parsed_data, number_of_time_lags)
observations_x, observations_y = get_observations(time_lagged_data)

observations_x_train, observations_x_test, observations_y_train, observations_y_test = train_test_split(observations_x,
                                                                                                        observations_y,
                                                                                                        shuffle=False,
                                                                                                        test_size=test_size)
