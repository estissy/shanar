# Shanar
Keras GRU neural network to forecast currency prices

## In general
This is little project that allows you to forecast  prices of currencies. Forecasting is made by recurrent neural network model. One trained model is supplied with project but you can train your own.
    
Project contains 4 Python scripts:
* `core.py` 

    This file have parameters for neural network, training and forecasting. It also parse, scale, and shift data in time (create time lags).

* `shanar_train.py`

    This script allows you to create new model based on parameters of neural network, and training.
    
* `shanar_test.py`
    
    This script evaluates model performance with test data.
    
* `shanar_predict.py`
    
    This script forecasts next prices starting from the last observations in data. The result is Python list with forecasted float values for currency.

## The idea
So, we have historical prices for currency. We want to feed our network with "time lags" for every selected price. Time lags are just previous prices or in general some number of historical values. Number of time lags is selected (in this project it is 14). Time lags can be in context of hour, day, year, etc. (in this project it is day, it depend of data you have). The input to the network is 14 ordered prices (prices from last 14 days) and the output is next price (price in day 15). Used network is recurrent. This type of network is specified for forecasting, trend finding and processing sequential data because they remember some of previous states and use it to calculate current output.

## Parameters
Parameters for network are stored in `core.py` file.

__General parameters__
* `model_name`
   
   Name of the model used when saving (after train), and when model is loaded (to evaluate or predict). Stored in `resources/model`.
   
* `data_name` 

    Name of data file used for training, evaluation and prediction. Stored in `resources/data`.
    
* `forecast_time_lags`

    How many next time lags you want to predict from model. Used in `shanar_predict.py`.

__Model parameters__
* `test_size`

    Fraction of data to go for testing, te rest goes for training. Value from 0 to 1.

* `number_of_time_lags`

    How many time lags you want. For trained model it is 14. The more time lags you use the bigger capacity of network should be.
    
* `gru_units`
    
    Units in Keras GRU layer in the network. It specifies capacity of the network.
    
* `optimizer`

    Optimizer used during training process.
    
* `batch_size`

    Size of batch size used in training, evaluation and prediction.
    
* `epochs`

    Number of epoch to train the model.
    
* `early_stopping_patience`

    Determine when to stop training process. Prevents overfitting. If values of loss function starts to increase in validation data for selected number of epochs the learning process will stop and the best model will be saved.

## How to use this
To use this scripts you must have Keras running on your machine. You should install this project requirements and use Keras with Theano backend.

[How to use Keras with Theano backend](https://keras.io/backend/)
    
__Requirements__
    
    h5py==2.8.0
    Keras==2.2.4
    Keras-Applications==1.0.6
    Keras-Preprocessing==1.0.5
    numpy==1.15.2
    pandas==0.23.4
    python-dateutil==2.7.3
    pytz==2018.5
    PyYAML==3.13
    scikit-learn==0.20.0
    scipy==1.1.0
    six==1.11.0
    sklearn==0.0
    Theano==1.0.3
