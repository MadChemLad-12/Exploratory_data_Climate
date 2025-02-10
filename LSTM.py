
from keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=5)
from data_preprocessing import preprocess_data 

raw_global = pd.read_csv('GLB.Ts+dSST.csv', skiprows=1)
global_t = preprocess_data(raw_global)
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

def LSTM_modelone(train_X, train_Y, window_size):
    model = Sequential()
    model.add(LSTM(4, 
            input_shape = (1, window_size)))
    #one LSTM layer four blocks and an input layer
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", # loss function
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
