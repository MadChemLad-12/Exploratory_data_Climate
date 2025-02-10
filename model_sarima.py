import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

def auto_sarima(train):
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

    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(p, s) for p in param_combinations for s in seasonal_combinations)
    
    best_model = min([res for res in results if res is not None], key=lambda x: x[2])
    print(f"Best SARIMA Order: {best_model[0]}, Seasonal Order: {best_model[1]}, AIC={best_model[2]}")
    
    return best_model[0], best_model[1]

def sarima_prediction(temp_data):
    temperature = temp_data['Temp'].tolist()
    train = temperature[:-336]
    test = temperature[len(train):]

    best_order, best_seasonal_order = auto_sarima(train)

    model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
    model_fit = model.fit(disp=False, low_memory=True)

    y_pred = model_fit.forecast(steps=len(test))
    y_true = test

    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")

    dates = temp_data['Date'][len(train):]
    return y_true, y_pred, dates