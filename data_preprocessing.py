import pandas as pd
import numpy as np

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

