import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

def prophet_prediction(temp_data):
    df = temp_data.iloc[:-336].copy()
    df = df.rename(columns={'Date': 'ds', 'Temp': 'y'})
    
    df['y'] = np.log1p(df['y'])

    model = Prophet(yearly_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=336, freq='M')

    forecast = model.predict(future)
    
    forecast['yhat'] = np.expm1(forecast['yhat'])
    model.plot(forecast)
    plt.title("Climate Temperature Prediction with Prophet")
    plt.show()

    return forecast