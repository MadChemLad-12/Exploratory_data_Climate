import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def plot_global_temperature(global_t):
    fig = px.line(global_t, x = "Date", y = "Temp", title='Global-mean monthly Combined Land-Surface Air and Sea-Surface Water Temperature Anomalies')
    fig.show()
    fig = px.line(global_t.resample('A', on='Date').mean().reset_index(), x="Date", y="Temp", title='Global-mean yearly Combined Land-Surface Air and Sea-Surface Water Temperature Anomalies')
    fig.show()

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