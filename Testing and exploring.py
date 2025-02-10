import pandas as pd
import numpy as np
import itertools
import calendar
from datetime import datetime
import time
# Standard plotly imports
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
# stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
from statsmodels.tsa.stattools import adfuller
#Prophet
#from fbprophet import Prophet
# SKLEARN
from sklearn.metrics import mean_squared_error

# Read in the raw temperature dataset
raw_global = pd.read_csv('GLB.Ts+dSST.csv', skiprows=1)
raw_global = raw_global.iloc[:,:13]

#Look at the data
#print(raw_global.head())
#print(raw_global.tail())
#print(raw_global.shape[0]) # Greater than 140
#num_months = raw_global.iloc[:, 1:].to_numpy().flatten().shape[0]
#print(f"Number of months in data: {num_months}")

#Cleaning and preprocessing data
def clean_value(raw_value):
    try:
        return float(raw_value)
    except:
        return np.NaN

def preprocess_data(raw):
    # Extract and flatten temperature data
    temp_list = raw.iloc[:, 1:].to_numpy().flatten()

    # Dynamically create a date range that matches the number of data points
    data_horizon = pd.date_range(start='1/1/1880', periods=len(temp_list), freq='M')
    data = pd.DataFrame(data_horizon, columns=['Date'])

    # Assign extracted data
    data['Temp'] = temp_list

    # Clean values
    data['Temp'] = data['Temp'].apply(lambda x: clean_value(x))
    data.fillna(method='ffill', inplace=True)

    return data

global_t = preprocess_data(raw_global)
print(global_t.head())
print(global_t.tail())

#Data visualisation
fig = px.line(global_t, x = "Date", y = "Temp", title='Global-mean monthly Combined Land-Surface Air and Sea-Surface Water Temperature Anomalies')
fig.show()
fig = px.line(global_t.resample('A', on='Date').mean().reset_index(), x="Date", y="Temp", title='Global-mean yearly Combined Land-Surface Air and Sea-Surface Water Temperature Anomalies')
fig.show()

#Test Stationarity
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window = 30).mean()
    rolstd = timeseries.rolling(window = 30).std()
    
    plt.figure(figsize=(14,5))
    sns.despine(left=True)
    #Plot Data values
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std') 
    
    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    print ('')
    
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
        index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)    
#test_stationarity(global_t.Temp.dropna())

#Time Series Prediction - SARIMA
def auto_sarima(train):
    """
    Automatically finds the best SARIMA model based on AIC.
    Uses parallel processing to speed up the search.
    """
    p = d = q = range(0, 3)  # ARIMA parameters (0 to 2)
    seasonal_p = seasonal_d = seasonal_q = range(0, 2)  # Seasonal components (0 to 1)
    seasonal_period = [12]  # Assume yearly seasonality

    param_combinations = list(itertools.product(p, d, q))
    seasonal_combinations = list(itertools.product(seasonal_p, seasonal_d, seasonal_q, seasonal_period))

    def evaluate_model(param, seasonal_param):
        try:
            model = SARIMAX(train, order=param, seasonal_order=seasonal_param)
            model_fit = model.fit(disp=False)
            return (param, seasonal_param, model_fit.aic)
        except:
            return None

    # Parallel processing to speed up the search
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(p, s) 
                for p in param_combinations for s in seasonal_combinations)
    
    best_model = min([res for res in results if res is not None], key=lambda x: x[2])
    print(f"Best SARIMA Order: {best_model[0]}, Seasonal Order: {best_model[1]}, AIC={best_model[2]}")
    
    return best_model[0], best_model[1]

def plot_predictions(y_true, y_pred, dates):
    """
    Plots the actual vs. predicted temperature values.
    """
    plt.figure(figsize=(14,6))
    plt.plot(dates, y_true, label='Actual', color='blue', linestyle='dashed')
    plt.plot(dates, y_pred, label='Predicted', color='red')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature Prediction vs. Actual')
    plt.show()

def sarima_prediction(temp_data):
    """
    Optimized SARIMA prediction function:
    - Automatically selects the best hyperparameters.
    - Trains the model once and forecasts efficiently.
    - Uses parallel computing for faster tuning.
    - Plots actual vs. predicted values.
    """
    # Convert temperature data to list
    temperature = temp_data['Temp'].tolist()
    
    # Split data: 80% train, 20% test
    train = temperature[:-336]
    test = temperature[len(train):]
    
    # Find best SARIMA parameters
    best_order, best_seasonal_order = auto_sarima(train)

    # Train SARIMA model
    model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
    model_fit = model.fit(disp=False, low_memory=True)

    # Forecast test set
    y_pred = model_fit.forecast(steps=len(test))
    y_true = test

    # Compute and display error
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Plot results
    dates = temp_data['Date'][len(train):]  # Get corresponding dates for test set
    plot_predictions(y_true, y_pred, dates)

# Run the optimized function
start_time = time.time()
#sarima_prediction(global_t)
print("--- %s seconds ---" % (time.time() - start_time))

#Time series prediction - LSTM
from keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=5)
     
#Data preparation
temp_raw = np.array(global_t.Temp.astype("float32")).reshape(-1,1)
# Apply the MinMax scaler from sklearn to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range = (0, 1))
temp_LSTM = scaler.fit_transform(temp_raw)
     
# Train test split - Using 80% of data for training, 20% for validation.
ratio = 0.6
train_size = int(len(temp_LSTM) * ratio)
val_size = int(len(temp_LSTM) * 0.2)
test_size = len(temp_LSTM) - train_size - val_size
train, val, test = temp_LSTM[0:train_size, :], temp_LSTM[train_size:train_size+val_size, :], temp_LSTM[train_size+val_size:len(temp_LSTM), :]
print("Number of entries (training set, val set, test set): " + str((len(train), len(val), len(test))))
     
def create_dataset(dataset):
    window_size = 1
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

# Create test and training sets for one-step-ahead regression.
train_X, train_Y = create_dataset(train)
val_X, val_Y = create_dataset(val)
test_X, test_Y = create_dataset(test)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1,train_X.shape[1]))
val_X = np.reshape(val_X, (val_X.shape[0], 1,val_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1,test_X.shape[1]))
print("Training data for Keras shape:")
print(train_X.shape)


#LSTM model
def LSTM_modelone(train_X, train_Y, window_size):
    model = Sequential()
    model.add(LSTM(4, 
            input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
            optimizer = "adam")
    model.fit(train_X, 
            train_Y, 
            epochs = 100, 
            batch_size = 10, 
            verbose = 2,
            validation_data=(val_X,val_Y),callbacks=[earlyStop])
    
    return model

start_time = time.time()
LSTM_model1 = LSTM_modelone(train_X, train_Y, window_size=1)
print("--- %s seconds ---" % (time.time() - start_time))

def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = mean_squared_error(orig_data[0], pred[:, 0])
    return score
     

print("Test data score: %.3f MSE" % predict_and_score(LSTM_model1,test_X, test_Y))
     

#LSTM model 2
def LSTM_modeltwo(train_X, train_Y):
  model = Sequential()
  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  print(model.summary())

  model.fit(train_X, train_Y, epochs=50, batch_size=5, verbose=2, shuffle=False, validation_data=(val_X,val_Y),callbacks=[earlyStop])

  return model
     

start_time = time.time()
LSTM_model2 = LSTM_modeltwo(train_X, train_Y)
print("--- %s seconds ---" % (time.time() - start_time))
print("Test data score: %.3f MSE" % predict_and_score(LSTM_model2,test_X, test_Y))
     
def predict_and_plot(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Plot
    fig = go.Figure()
    x = global_t['Date'][global_t.shape[0]-len(orig_data[0]):]
    fig.add_trace(go.Scatter(x=x, y=orig_data[0], mode='lines', name='actual'))
    fig.add_trace(go.Scatter(x=x, y=pred[:, 0], mode='lines', name='predicted'))
    # Edit the layout
    fig.update_layout(title='Global Temperature: Predicted v.s. Actual',
                   xaxis_title='Month',
                   yaxis_title='Temperature')
    fig.show()

predict_and_plot(LSTM_model2,test_X, test_Y)

#MLP model
def MLP_model(train_X, train_Y):
  model = Sequential()
  model.add(Dense(100,  input_shape=(1,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.25))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.add(Activation('linear'))
  model.compile(optimizer='adam', loss='mse')
  print(model.summary())
  model.fit(train_X, train_Y, epochs=50, batch_size=10, verbose=2, shuffle=False, validation_data=(val_X,val_Y),callbacks=[earlyStop])
  return model
     

start_time = time.time()
MLP_model_result = MLP_model(train_X, train_Y)
print("--- %s seconds ---" % (time.time() - start_time))
print("Test data score: %.3f MSE" % predict_and_score(MLP_model_result,test_X, test_Y))


quit()
#Time Series Prediction - Prophet
def prophet_prediction(temp_data):
    """
    Optimized Prophet Model:
    - Uses log transformation for trend stabilization.
    - Removes irrelevant weekly seasonality.
    - Trains in parallel to speed up performance.
    - Forecasts and evaluates performance.
    """
    # Remove last 336 months for prediction
    df = temp_data.iloc[:-336].copy()
    
    # Rename columns for Prophet
    df = df.rename(columns={'Date': 'ds', 'Temp': 'y'})
    
    # Apply log transformation
    df['y'] = np.log1p(df['y'])

    # Initialize and train the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(df)

    # Create future dates
    future = model.make_future_dataframe(periods=336, freq='M')

    # Make predictions
    forecast = model.predict(future)
    
    # Reverse log transformation
    forecast['yhat'] = np.expm1(forecast['yhat'])

    # Plot forecast
    model.plot(forecast)
    plt.title("Climate Temperature Prediction with Prophet")
    plt.show()

    return forecast

# Measure execution time
start_time = time.time()
prophet_forecast = prophet_prediction(global_t)
print("--- %s seconds ---" % (time.time() - start_time))

# Evaluate performance
prophet_forecast_last = prophet_forecast.iloc[-336:]
global_t_last = global_t.iloc[-336:]
mse = mean_squared_error(global_t_last.Temp, prophet_forecast_last.yhat)
print(f"Mean Squared Error: {mse}")

