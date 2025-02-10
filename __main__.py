import time
import pandas as pd
from data_preprocessing import preprocess_data
from visualization import plot_global_temperature, plot_predictions
from model_sarima import sarima_prediction
from model_prophet import prophet_prediction

def main():
    # Read in the raw temperature dataset
    raw_global = pd.read_csv('GLB.Ts+dSST.csv', skiprows=1)
    raw_global = raw_global.iloc[:,:13]
    
    # Preprocess data
    global_t = preprocess_data(raw_global)
    print(global_t.head())
    print(global_t.tail())

    # Plot temperature data
    plot_global_temperature(global_t)

    # SARIMA Model Prediction
    start_time = time.time()
    y_true, y_pred, dates = sarima_prediction(global_t)
    plot_predictions(y_true, y_pred, dates)
    print("--- SARIMA Prediction Time: %s seconds ---" % (time.time() - start_time))

    # Prophet Model Prediction
    start_time = time.time()
    prophet_forecast = prophet_prediction(global_t)
    print("--- Prophet Prediction Time: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()