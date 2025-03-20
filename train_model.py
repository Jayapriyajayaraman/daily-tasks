# train_model.py

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Load your time series data (replace this with your own dataset)
# Example: Loading a CSV file
data = pd.read_csv('your_timeseries_data.csv', parse_dates=['Date'], index_col='Date')

# Check the structure of your data (ensure it's a time series with a Date column)
print(data.head())

# Train a SARIMA model (you can adjust the parameters based on your data)
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Save the trained model to a file (this will be used in the FastAPI app)
results.save('forecasting_model.pkl')

print("Model trained and saved as forecasting_model.pkl")
