'''import pandas as pd
from prophet import Prophet

# Load the data (assume it's in CSV format)
data = pd.read_csv('merged_company_data.csv')

# Ensure Date is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Prepare a list to hold individual DataFrames for each company
company_forecasts = []

# Get unique companies from the dataset
companies = data['Symbol'].unique()

for company in companies:
    # Filter the data for the current company
    company_data = data[data['Symbol'] == company]
    
    # Prepare the DataFrame for Prophet
    company_data = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Add a column to identify the company (optional)
    company_data['Symbol'] = company
    
    # Fit the model
    model = Prophet()
    model.fit(company_data)
    
    # Create future DataFrame for predictions (e.g., 5 days into the future)
    future = model.make_future_dataframe(periods=5)
    
    # Predict future stock prices
    forecast = model.predict(future)
    
    # Append the forecast results
    forecast['Symbol'] = company  # Add company column for identification
    company_forecasts.append(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Symbol']])

    # Concatenate all company forecasts into a single DataFrame
final_forecast = pd.concat(company_forecasts, ignore_index=True)

import matplotlib.pyplot as plt

# Function to plot forecasts for each company
def plot_forecast(forecast_df):
    for company in forecast_df['Symbol'].unique():
        company_forecast = forecast_df[forecast_df['Symbol'] == company]
        plt.figure(figsize=(10, 5))
        plt.plot(company_forecast['ds'], company_forecast['yhat'], label='Predicted', color='blue')
        plt.fill_between(company_forecast['ds'], 
                         company_forecast['yhat_lower'], 
                         company_forecast['yhat_upper'], 
                         color='lightblue', alpha=0.5, label='Confidence Interval')
        plt.title(f'Stock Price Forecast for {company}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid()
        plt.show()

plot_forecast(final_forecast)'''

'''import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of important companies for analysis
chosen_companies = ['MMM', 'AAPL', 'GOOGL']  # Example company tickers, modify as needed

# Filter data for only important companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast in a dictionary
forecasts = {}

for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Prepare the data for Prophet
    df = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,  # Disable default yearly seasonality
        weekly_seasonality=False,  # Disable default weekly seasonality
        holidays=holidays,         # Add holidays
        seasonality_mode='multiplicative',  # Adjust trend-season interaction
        changepoint_prior_scale=0.05  # Increase sensitivity to changes in trend
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Fit the model on company data
    model.fit(df)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # Generate forecast
    forecast = model.predict(future)
    forecasts[company] = forecast

    # Plot forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company}')
    plt.legend()
    plt.show()'''

'''import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of important companies for analysis
chosen_companies = ['MMM', 'AAPL', 'GOOGL']  # Example company tickers, modify as needed

# Filter data for only important companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast in a dictionary
forecasts = {}

for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Prepare the data for Prophet
    df = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,  # Disable default yearly seasonality
        weekly_seasonality=False,  # Disable default weekly seasonality
        holidays=holidays,         # Add holidays
        seasonality_mode='multiplicative',  # Adjust trend-season interaction
        changepoint_prior_scale=0.05  # Increase sensitivity to changes in trend
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Fit the model on company data
    model.fit(df)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # Generate forecast
    forecast = model.predict(future)
    forecasts[company] = forecast

    # Plot forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company}')
    plt.legend()

    # Save each plot with the company name
    plt.savefig(f'{company}_forecast.png')
    plt.close()  # Close the plot to free up memory

print("Forecasts and plots saved for each important company.")'''

'''import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# Calculate average volume for each company and determine the top 20% threshold
company_avg_volume = data.groupby('Symbol')['Volume'].mean().reset_index()
threshold_volume = company_avg_volume['Volume'].quantile(0.8)
chosen_companies = company_avg_volume[company_avg_volume['Volume'] >= threshold_volume]['Symbol']

# Filter data for only important companies based on the threshold
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast in a dictionary
forecasts = {}

for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Prepare the data for Prophet
    df = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,  # Disable default yearly seasonality
        weekly_seasonality=False,  # Disable default weekly seasonality
        holidays=holidays,         # Add holidays
        seasonality_mode='multiplicative',  # Adjust trend-season interaction
        changepoint_prior_scale=0.05  # Increase sensitivity to changes in trend
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Fit the model on company data
    model.fit(df)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # Generate forecast
    forecast = model.predict(future)
    forecasts[company] = forecast

    # Plot forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company}')
    plt.legend()

    # Save each plot with the company name
    plt.savefig(f'{company}_forecast.png')
    plt.close()  # Close the plot to free up memory

print("Forecasts and plots saved for each important company.")'''

''' model before i included additional regressors
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of companies for analysis
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the selected companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast and MAPE in dictionaries
forecasts = {}
mape_values = []

for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Prepare the data for Prophet
    df = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Fit the model on company data
    model.fit(df)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # Generate forecast
    forecast = model.predict(future)
    forecasts[company] = forecast

    # Plot forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company}')
    plt.legend()

    # Save each plot with the company name
    plt.savefig(f'{company}_forecast.png')
    plt.close()  # Close the plot to free up memory

# Calculate MAPE for each company
company_mape = {}

for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    df = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Ensure the lengths match by merging on dates
    actual_vs_pred = df.merge(forecasts[company][['ds', 'yhat']], on='ds')
    
    # Compute MAPE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    company_mape[company] = mape
    print(f"MAPE for {company}: {mape:.2%}")

# Calculate and print the average MAPE
average_mape = sum(company_mape.values()) / len(company_mape)
print(f"Average MAPE across companies: {average_mape:.2%}") '''

'''ruined model after trying to include regressors and had the values being fitted messed up again
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# Define the list of selected companies
selected_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for selected companies
filtered_data = data[data['Symbol'].isin(selected_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast and MAPE in dictionaries
forecasts = {}
mape_results = {}

for company in selected_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Prepare the data for Prophet
    df = company_data[['Date', 'Close', 'News - Positive Sentiment', 'News - Negative Sentiment']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Handle missing values by filling with zero (or use forward fill if more suitable)
    df[['News - Positive Sentiment', 'News - Negative Sentiment']] = df[['News - Positive Sentiment', 'News - Negative Sentiment']].fillna(0)
    
    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])
    
    # Add the news sentiment regressors
    model.add_regressor('News - Positive Sentiment')
    model.add_regressor('News - Negative Sentiment')

    # Fit the model on company data
    model.fit(df)

    # Create a future dataframe for predictions, including regressors
    future = model.make_future_dataframe(periods=future_periods)
    future[['News - Positive Sentiment', 'News - Negative Sentiment']] = df[['News - Positive Sentiment', 'News - Negative Sentiment']].iloc[-future_periods:].values

    # Generate forecast
    forecast = model.predict(future)
    forecasts[company] = forecast

    # Calculate MAPE for the fitted model (training period)
    mape = mean_absolute_percentage_error(df['y'], forecast['yhat'][:len(df)])
    mape_results[company] = mape

    # Plot forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company} (with News Sentiment Regressors)')
    plt.legend()

    # Save each plot with the company name
    plt.savefig(f'{company}_forecast_with_regressors.png')
    plt.close()  # Close the plot to free up memory

# Calculate the average MAPE across all selected companies
average_mape = sum(mape_results.values()) / len(mape_results)

print("MAPE for each company:", mape_results)
print("Average MAPE across companies:", average_mape)
print("Forecasts and plots saved for each selected company.")'''

''' pretty good model that gave mape before and after adding regressors
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of companies for analysis
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the selected companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast and MAPE in dictionaries
forecasts = {}
mape_values_without_regressors = {}
mape_values_with_regressors = []

# Define a function to create forecasts
def create_forecast(df, company, use_regressors=False):
    # Prepare the data for Prophet
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Add regressors if specified
    if use_regressors:
        df_prophet['News - Positive Sentiment'] = df['News - Positive Sentiment'].fillna(0)  # Fill NaN values
        df_prophet['News - Negative Sentiment'] = df['News - Negative Sentiment'].fillna(0)  # Fill NaN values
        model.add_regressor('News - Positive Sentiment')
        model.add_regressor('News - Negative Sentiment')

    # Fit the model on company data
    model.fit(df_prophet)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # If regressors are used, merge them into future
    if use_regressors:
        future = future.merge(df_prophet[['ds', 'News - Positive Sentiment', 'News - Negative Sentiment']], on='ds', how='left')
        future['News - Positive Sentiment'] = future['News - Positive Sentiment'].fillna(0)
        future['News - Negative Sentiment'] = future['News - Negative Sentiment'].fillna(0)

    # Generate forecast
    forecast = model.predict(future)
    return forecast

# First pass: Without regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast without regressors
    forecast = create_forecast(company_data, company, use_regressors=False)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    mape_values_without_regressors[company] = mape
    print(f"MAPE for {company} without regressors: {mape:.2%}")

# Second pass: With regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast with regressors
    forecast = create_forecast(company_data, company, use_regressors=True)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    mape_values_with_regressors.append(mape)
    print(f"MAPE for {company} with regressors: {mape:.2%}")

# Calculate and print the average MAPEs
average_mape_without_regressors = sum(mape_values_without_regressors.values()) / len(mape_values_without_regressors)
print(f"Average MAPE across companies without regressors: {average_mape_without_regressors:.2%}")

average_mape_with_regressors = sum(mape_values_with_regressors) / len(mape_values_with_regressors)
print(f"Average MAPE across companies with regressors: {average_mape_with_regressors:.2%}")

# Optionally plot the forecasts
for company, forecast in forecasts.items():
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data[filtered_data['Symbol'] == company]['Date'], 
             filtered_data[filtered_data['Symbol'] == company]['Close'], label='Actual', color='black')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='skyblue',
        alpha=0.4,
        label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company}')
    plt.legend()
    plt.savefig(f'{company}_forecast.png')
    plt.close()  # Close the plot to free up memory'''

'''code that plots 2 x 5 companies
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of companies for analysis
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the selected companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast, MAPE, and RMSE in dictionaries
forecasts = {}
metrics_without_regressors = {}
metrics_with_regressors = {}

# Define a function to create forecasts and compute evaluation metrics
def create_forecast(df, company, use_regressors=False):
    # Prepare the data for Prophet
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Add regressors if specified
    if use_regressors:
        df_prophet['News - Positive Sentiment'] = df['News - Positive Sentiment'].fillna(0)
        df_prophet['News - Negative Sentiment'] = df['News - Negative Sentiment'].fillna(0)
        model.add_regressor('News - Positive Sentiment')
        model.add_regressor('News - Negative Sentiment')

    # Fit the model on company data
    model.fit(df_prophet)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # If regressors are used, merge them into future
    if use_regressors:
        future = future.merge(df_prophet[['ds', 'News - Positive Sentiment', 'News - Negative Sentiment']], on='ds', how='left')
        future['News - Positive Sentiment'] = future['News - Positive Sentiment'].fillna(0)
        future['News - Negative Sentiment'] = future['News - Negative Sentiment'].fillna(0)

    # Generate forecast
    forecast = model.predict(future)
    return forecast

# Helper function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# First pass: Without regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast without regressors
    forecast = create_forecast(company_data, company, use_regressors=False)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE and RMSE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    rmse = calculate_rmse(actual_vs_pred['y'], actual_vs_pred['yhat'])
    metrics_without_regressors[company] = {'MAPE': mape, 'RMSE': rmse}
    print(f"MAPE for {company} without regressors: {mape:.2%}, RMSE: {rmse:.2f}")

# Second pass: With regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast with regressors
    forecast = create_forecast(company_data, company, use_regressors=True)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE and RMSE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    rmse = calculate_rmse(actual_vs_pred['y'], actual_vs_pred['yhat'])
    metrics_with_regressors[company] = {'MAPE': mape, 'RMSE': rmse}
    print(f"MAPE for {company} with regressors: {mape:.2%}, RMSE: {rmse:.2f}")

# Calculate and print the average metrics across companies
average_mape_without_regressors = np.mean([v['MAPE'] for v in metrics_without_regressors.values()])
average_rmse_without_regressors = np.mean([v['RMSE'] for v in metrics_without_regressors.values()])
print(f"Average MAPE without regressors: {average_mape_without_regressors:.2%}, Average RMSE without regressors: {average_rmse_without_regressors:.2f}")

average_mape_with_regressors = np.mean([v['MAPE'] for v in metrics_with_regressors.values()])
average_rmse_with_regressors = np.mean([v['RMSE'] for v in metrics_with_regressors.values()])
print(f"Average MAPE with regressors: {average_mape_with_regressors:.2%}, Average RMSE with regressors: {average_rmse_with_regressors:.2f}")

# Set up the number of companies to display per graph
num_companies_per_graph = 5
num_graphs = 2  # Total number of graphs you want

# Create a list of the chosen companies
companies_list = list(chosen_companies)

# Plot the forecasts for each company in two separate graphs
for i in range(num_graphs):
    plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
    for j in range(num_companies_per_graph):
        index = i * num_companies_per_graph + j
        if index < len(companies_list):  # Check if the index is within bounds
            company = companies_list[index]
            forecast = forecasts[company]
            
            # Plot actual data
            plt.plot(filtered_data[filtered_data['Symbol'] == company]['Date'], 
                     filtered_data[filtered_data['Symbol'] == company]['Close'], 
                     label=f'Actual {company}', color='black', linewidth=2)
            # Plot predicted data
            plt.plot(forecast['ds'], forecast['yhat'], label=f'Predicted {company}', linewidth=2)
            # Fill confidence interval
            plt.fill_between(
                forecast['ds'],
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                alpha=0.3
            )

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for Companies {i * num_companies_per_graph + 1} to {(i + 1) * num_companies_per_graph}')
    plt.legend()
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f'forecast_graph_{i + 1}.png')
    plt.close()  # Close the plot to free up memory'''

''' good code that outputs the confidence intervals widths and the graphs
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Ensure correct date parsing

# List of companies for analysis
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the selected companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store each forecast, MAPE, and RMSE in dictionaries
forecasts = {}
metrics_without_regressors = {}
metrics_with_regressors = {}
confidence_intervals = {}  # To store confidence interval widths

# Define a function to create forecasts and compute evaluation metrics
def create_forecast(df, company, use_regressors=False):
    # Prepare the data for Prophet
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize Prophet model with custom settings
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Add custom seasonalities
    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    # Add regressors if specified
    if use_regressors:
        df_prophet['News - Positive Sentiment'] = df['News - Positive Sentiment'].fillna(0)
        df_prophet['News - Negative Sentiment'] = df['News - Negative Sentiment'].fillna(0)
        model.add_regressor('News - Positive Sentiment')
        model.add_regressor('News - Negative Sentiment')

    # Fit the model on company data
    model.fit(df_prophet)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=future_periods)

    # If regressors are used, merge them into future
    if use_regressors:
        future = future.merge(df_prophet[['ds', 'News - Positive Sentiment', 'News - Negative Sentiment']], on='ds', how='left')
        future['News - Positive Sentiment'] = future['News - Positive Sentiment'].fillna(0)
        future['News - Negative Sentiment'] = future['News - Negative Sentiment'].fillna(0)

    # Generate forecast
    forecast = model.predict(future)
    
    return forecast

# Helper function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# First pass: Without regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast without regressors
    forecast = create_forecast(company_data, company, use_regressors=False)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE and RMSE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    rmse = calculate_rmse(actual_vs_pred['y'], actual_vs_pred['yhat'])
    metrics_without_regressors[company] = {'MAPE': mape, 'RMSE': rmse}
    print(f"MAPE for {company} without regressors: {mape:.2%}, RMSE: {rmse:.2f}")

# Second pass: With regressors
for company in chosen_companies:
    # Filter for a single company's data
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    # Create forecast with regressors
    forecast = create_forecast(company_data, company, use_regressors=True)
    forecasts[company] = forecast

    # Ensure the lengths match by merging on dates
    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    # Compute MAPE and RMSE for each company
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    rmse = calculate_rmse(actual_vs_pred['y'], actual_vs_pred['yhat'])
    metrics_with_regressors[company] = {'MAPE': mape, 'RMSE': rmse}
    print(f"MAPE for {company} with regressors: {mape:.2%}, RMSE: {rmse:.2f}")

# Calculate and print the average metrics across companies
average_mape_without_regressors = np.mean([v['MAPE'] for v in metrics_without_regressors.values()])
average_rmse_without_regressors = np.mean([v['RMSE'] for v in metrics_without_regressors.values()])
print(f"Average MAPE without regressors: {average_mape_without_regressors:.2%}, Average RMSE without regressors: {average_rmse_without_regressors:.2f}")

average_mape_with_regressors = np.mean([v['MAPE'] for v in metrics_with_regressors.values()])
average_rmse_with_regressors = np.mean([v['RMSE'] for v in metrics_with_regressors.values()])
print(f"Average MAPE with regressors: {average_mape_with_regressors:.2%}, Average RMSE with regressors: {average_rmse_with_regressors:.2f}")

# Calculate confidence interval widths and store for comparison
for company, forecast in forecasts.items():
    confidence_interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
    confidence_intervals[company] = confidence_interval_width.mean()  # Store average width for each company
    print(f"Average confidence interval width for {company}: {confidence_interval_width.mean():.2f}")

# Plotting forecasts in two separate figures
num_companies = len(chosen_companies)  # Get the number of companies

# Create the first figure for the first half of companies
fig1, axs1 = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))  # Adjust the number of rows and columns
axs1 = axs1.flatten()  # Flatten the 2D array of axes for easy indexing

# Plot first five companies
for i, company in enumerate(chosen_companies[:5]):
    forecast = forecasts[company]  # Get the forecast for the company

    # Plot actual stock prices
    axs1[i].plot(filtered_data[filtered_data['Symbol'] == company]['Date'], 
                  filtered_data[filtered_data['Symbol'] == company]['Close'], 
                  label='Actual Prices', color='blue')  # Actual stock prices in blue
    
    # Plot predicted stock prices
    axs1[i].plot(forecast['ds'], forecast['yhat'], label='Predicted Prices', color='orange')  # Predicted stock prices in orange
    
    # Plot the uncertainty intervals
    axs1[i].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                         color='gray', alpha=0.2, label='Uncertainty Interval')  # Fill between the lower and upper bounds of predictions
    
    # Set plot title and labels
    axs1[i].set_title(company)
    axs1[i].set_xlabel('Date')
    axs1[i].set_ylabel('Stock Price')
    axs1[i].legend()  # Add legend to the plot

# Hide any empty subplots if they exist
for j in range(i + 1, len(axs1)):
    axs1[j].axis('off')

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.savefig('stock_price_forecasts_first_5.png')  # Save the first figure
plt.show()  # Display the first figure

# Create the second figure for the second half of companies
fig2, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))  # Adjust the number of rows and columns
axs2 = axs2.flatten()  # Flatten the 2D array of axes for easy indexing

# Plot the remaining five companies
for i, company in enumerate(chosen_companies[5:]):
    forecast = forecasts[company]  # Get the forecast for the company

    # Plot actual stock prices
    axs2[i].plot(filtered_data[filtered_data['Symbol'] == company]['Date'], 
                  filtered_data[filtered_data['Symbol'] == company]['Close'], 
                  label='Actual Prices', color='blue')  # Actual stock prices in blue
    
    # Plot predicted stock prices
    axs2[i].plot(forecast['ds'], forecast['yhat'], label='Predicted Prices', color='orange')  # Predicted stock prices in orange
    
    # Plot the uncertainty intervals
    axs2[i].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                         color='gray', alpha=0.2, label='Uncertainty Interval')  # Fill between the lower and upper bounds of predictions
    
    # Set plot title and labels
    axs2[i].set_title(company)
    axs2[i].set_xlabel('Date')
    axs2[i].set_ylabel('Stock Price')
    axs2[i].legend()  # Add legend to the plot

# Hide any empty subplots if they exist
for j in range(i + 1, len(axs2)):
    axs2[j].axis('off')

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.savefig('stock_price_forecasts_second_5.png')  # Save the second figure
plt.show()  # Display the second figure'''


''' normal and rolling forecasts and allat but no graphs
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# List of companies for analysis
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the selected companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Store forecasts, metrics, and confidence intervals
forecasts = {}
metrics_without_regressors = {}
metrics_with_regressors = {}
confidence_intervals = {}
confidence_interval_percentages = {}

# Define a function to create forecasts and compute evaluation metrics
def create_forecast(df, company, use_regressors=False):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    if use_regressors:
        df_prophet['News - Positive Sentiment'] = df['News - Positive Sentiment'].fillna(0)
        df_prophet['News - Negative Sentiment'] = df['News - Negative Sentiment'].fillna(0)
        model.add_regressor('News - Positive Sentiment')
        model.add_regressor('News - Negative Sentiment')

    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=future_periods)

    if use_regressors:
        future = future.merge(df_prophet[['ds', 'News - Positive Sentiment', 'News - Negative Sentiment']], on='ds', how='left')
        future['News - Positive Sentiment'] = future['News - Positive Sentiment'].fillna(0)
        future['News - Negative Sentiment'] = future['News - Negative Sentiment'].fillna(0)

    forecast = model.predict(future)
    
    return forecast

# Helper function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Forecast without regressors
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]
    
    forecast = create_forecast(company_data, company, use_regressors=False)
    forecasts[company] = forecast

    actual_vs_pred = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast[['ds', 'yhat']], on='ds')
    
    mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])
    rmse = calculate_rmse(actual_vs_pred['y'], actual_vs_pred['yhat'])
    metrics_without_regressors[company] = {'MAPE': mape, 'RMSE': rmse}

# Calculate confidence interval metrics without regressors
for company, forecast in forecasts.items():
    confidence_interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
    confidence_intervals[company] = confidence_interval_width.mean()
    
    company_range = filtered_data[filtered_data['Symbol'] == company]['Close'].max() - filtered_data[filtered_data['Symbol'] == company]['Close'].min()
    
    if company_range > 0:
        ci_percentage = (confidence_intervals[company] / company_range) * 100
    else:
        ci_percentage = np.nan

    confidence_interval_percentages[company] = ci_percentage

# Print metrics for forecasts without regressors
print("Metrics without Regressors:")
for company, metrics in metrics_without_regressors.items():
    print(f"{company} - MAPE: {metrics['MAPE']:.2%}, RMSE: {metrics['RMSE']:.2f}, Average Confidence Interval Width: {confidence_intervals[company]:.2f}, CI % of Range: {confidence_interval_percentages[company]:.2f}%")

# Forecast with regressors
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Create forecast with regressors
    forecast_with_regressors = create_forecast(company_data, company, use_regressors=True)
    forecasts[company + "_with_regressors"] = forecast_with_regressors

    actual_vs_pred_with_regressors = company_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).merge(forecast_with_regressors[['ds', 'yhat']], on='ds')

    # Calculate metrics for forecasts with regressors
    mape_with_regressors = mean_absolute_percentage_error(actual_vs_pred_with_regressors['y'], actual_vs_pred_with_regressors['yhat'])
    rmse_with_regressors = calculate_rmse(actual_vs_pred_with_regressors['y'], actual_vs_pred_with_regressors['yhat'])
    metrics_with_regressors[company] = {'MAPE': mape_with_regressors, 'RMSE': rmse_with_regressors}

# Calculate confidence interval metrics with regressors
for company, forecast in forecasts.items():
    if "_with_regressors" in company:
        confidence_interval_width_with_regressors = forecast['yhat_upper'] - forecast['yhat_lower']
        confidence_intervals[company] = confidence_interval_width_with_regressors.mean()
        
        original_company_name = company.replace("_with_regressors", "")
        company_range = filtered_data[filtered_data['Symbol'] == original_company_name]['Close'].max() - filtered_data[filtered_data['Symbol'] == original_company_name]['Close'].min()
        
        if company_range > 0:
            ci_percentage = (confidence_intervals[company] / company_range) * 100
        else:
            ci_percentage = np.nan

        confidence_interval_percentages[company] = ci_percentage

# Print metrics for forecasts with regressors
print("\nMetrics with Regressors:")
for company, metrics in metrics_with_regressors.items():
    print(f"{company} - MAPE: {metrics['MAPE']:.2%}, RMSE: {metrics['RMSE']:.2f}, Average Confidence Interval Width: {confidence_intervals[company + '_with_regressors']:.2f}, CI % of Range: {confidence_interval_percentages[company + '_with_regressors']:.2f}%")


from prophet import Prophet
import numpy as np
import pandas as pd

# Function to calculate rolling forecast MAPE without regressors
def rolling_forecast_mape(df, company, periods=future_periods):
    # Handle NaN values in the 'Close' column
    df['Close'].fillna(df['Close'].mean(), inplace=True)
    
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    mape_list = []

    # Set initial training data
    initial_train_size = len(df_prophet) - periods

    for i in range(periods):
        # Create a model for the current training window
        train_data = df_prophet.iloc[:initial_train_size + i].copy()
        
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            holidays=holidays,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        model.fit(train_data)

        # Create future dataframe for the next period
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        # Extract the predicted value for the next period
        predicted_value = forecast['yhat'].iloc[-1]

        # Get the actual value for that next period
        actual_value = df_prophet['y'].iloc[initial_train_size + i]

        # Calculate MAPE for this prediction
        mape = np.abs((actual_value - predicted_value) / actual_value) * 100
        mape_list.append(mape)

    # Calculate mean MAPE for rolling forecast
    rolling_mape = np.mean(mape_list)
    return rolling_mape, mape_list

# Function to calculate rolling forecast MAPE with regressors
def rolling_forecast_mape_with_regressors(df, company, regressors, periods=future_periods):
    # Handle NaN values for the regressors and 'Close' column
    for regressor in regressors:
        df[regressor].fillna(df[regressor].mean(), inplace=True)
    df['Close'].fillna(df['Close'].mean(), inplace=True)

    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    mape_list = []

    # Set initial training data
    initial_train_size = len(df_prophet) - periods

    for i in range(periods):
        # Create a model for the current training window
        train_data = df_prophet.iloc[:initial_train_size + i].copy()

        # Add regressors to the training data
        for regressor in regressors:
            train_data[regressor] = df[regressor].iloc[:initial_train_size + i].values

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            holidays=holidays,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        # Add regressors to the Prophet model
        for regressor in regressors:
            model.add_regressor(regressor)

        model.fit(train_data)

        # Create future dataframe for the next period
        future = model.make_future_dataframe(periods=1)

        # Add regressor values for the future periods
        for regressor in regressors:
            future[regressor] = df[regressor].iloc[initial_train_size + i]  # Direct assignment

        forecast = model.predict(future)

        # Extract the predicted value for the next period
        predicted_value = forecast['yhat'].iloc[-1]

        # Get the actual value for that next period
        actual_value = df_prophet['y'].iloc[initial_train_size + i]

        # Calculate MAPE for this prediction
        mape = np.abs((actual_value - predicted_value) / actual_value) * 100
        mape_list.append(mape)

    # Calculate mean MAPE for rolling forecast
    rolling_mape = np.mean(mape_list)
    return rolling_mape, mape_list

# Specify your regressors here
regressors = ['News - Positive Sentiment', 'News - Negative Sentiment', 'News - New Products', 'Volume', 'News - Analyst Comments', 'News - Stocks']

# Calculate rolling MAPE for each company without regressors
rolling_mape_results = {}
rolling_mape_individuals_without = {}
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]
    rolling_mape, individual_mapes = rolling_forecast_mape(company_data, company)
    rolling_mape_results[company] = rolling_mape
    rolling_mape_individuals_without[company] = individual_mapes

# Calculate rolling MAPE for each company with regressors
rolling_mape_results_with_regressors = {}
rolling_mape_individuals_with = {}
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]
    rolling_mape, individual_mapes = rolling_forecast_mape_with_regressors(company_data, company, regressors)
    rolling_mape_results_with_regressors[company] = rolling_mape
    rolling_mape_individuals_with[company] = individual_mapes

# Output mean rolling forecast MAPE across all companies
mean_rolling_mape = np.mean(list(rolling_mape_results.values()))
mean_rolling_mape_with_regressors = np.mean(list(rolling_mape_results_with_regressors.values()))

# Summary of results
print("\nRolling Forecast MAPE Summary:")
print("--------------------------------------------------")
print(f"{'Company':<15} {'MAPE (without regressors)':<30} {'MAPE (with regressors)':<30}")
print("--------------------------------------------------")
for company in chosen_companies:
    mape_without = rolling_mape_results.get(company, 0)
    mape_with = rolling_mape_results_with_regressors.get(company, 0)
    print(f"{company:<15} {mape_without:.2f}%{' ' * 20} {mape_with:.2f}%")

print("--------------------------------------------------")
print(f"Mean Rolling Forecast MAPE across all companies without regressors: {mean_rolling_mape:.2f}%")
print(f"Mean Rolling Forecast MAPE across all companies with regressors: {mean_rolling_mape_with_regressors:.2f}%")'''


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

data = pd.read_csv('merged_company_data.csv', low_memory=False, dtype={'ColumnName': str})
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# List of companies we chose at random from each industry
chosen_companies = ['XOM', 'SHW', 'AMZN', 'PG', 'JNJ', 'JPM', 'AAPL', 'GOOGL', 'NEE', 'AMT', 'MMM']

# Filter data for only the chosen companies
filtered_data = data[data['Symbol'].isin(chosen_companies)]

# Forecasting parameters
future_periods = 365  # Number of days to forecast
custom_seasonalities = {
    'monthly': {'period': 30.5, 'fourier_order': 5},
    'quarterly': {'period': 91.25, 'fourier_order': 7}
}

# Holidays (Example: US stock market holidays)
holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# Function to create forecasts and compute evaluation metrics
def create_forecast(df, company, use_regressors=False):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    for name, params in custom_seasonalities.items():
        model.add_seasonality(name=name, period=params['period'], fourier_order=params['fourier_order'])

    if use_regressors:
        df_prophet['News - Positive Sentiment'] = df['News - Positive Sentiment'].fillna(0)
        df_prophet['News - Negative Sentiment'] = df['News - Negative Sentiment'].fillna(0)
        model.add_regressor('News - Positive Sentiment')
        model.add_regressor('News - Negative Sentiment')

    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=future_periods)

    if use_regressors:
        future = future.merge(df_prophet[['ds', 'News - Positive Sentiment', 'News - Negative Sentiment']], on='ds', how='left')
        future['News - Positive Sentiment'] = future['News - Positive Sentiment'].fillna(0)
        future['News - Negative Sentiment'] = future['News - Negative Sentiment'].fillna(0)

    forecast = model.predict(future)
    
    return forecast

# Helper function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate rolling forecast predictions WITHOUT regressors
def rolling_forecast_predictions(df, company, periods=future_periods):
    df['Close'].fillna(df['Close'].mean(), inplace=True)
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    predictions = []
    lower_bounds = []
    upper_bounds = []
    initial_train_size = len(df_prophet) - periods

    for i in range(periods):
        train_data = df_prophet.iloc[:initial_train_size + i].copy()
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, holidays=holidays,
                        seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
        model.fit(train_data)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        predicted_value = forecast['yhat'].iloc[-1]
        lower_bound = forecast['yhat_lower'].iloc[-1]
        upper_bound = forecast['yhat_upper'].iloc[-1]

        predictions.append(predicted_value)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return predictions, lower_bounds, upper_bounds

# Function to calculate rolling forecast predictions WITH regressors
def rolling_forecast_predictions_with_regressors(df, company, regressors, periods=future_periods):
    for regressor in regressors:
        df[regressor].fillna(df[regressor].mean(), inplace=True)
    df['Close'].fillna(df['Close'].mean(), inplace=True)
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    predictions = []
    lower_bounds = []
    upper_bounds = []
    initial_train_size = len(df_prophet) - periods

    for i in range(periods):
        train_data = df_prophet.iloc[:initial_train_size + i].copy()
        for regressor in regressors:
            train_data[regressor] = df[regressor].iloc[:initial_train_size + i].values

        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, holidays=holidays,
                        seasonality_mode='multiplicative', changepoint_prior_scale=0.05)

        for regressor in regressors:
            model.add_regressor(regressor)

        model.fit(train_data)
        future = model.make_future_dataframe(periods=1)

        for regressor in regressors:
            future[regressor] = df[regressor].iloc[initial_train_size + i]

        forecast = model.predict(future)

        predicted_value = forecast['yhat'].iloc[-1]
        lower_bound = forecast['yhat_lower'].iloc[-1]
        upper_bound = forecast['yhat_upper'].iloc[-1]

        predictions.append(predicted_value)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return predictions, lower_bounds, upper_bounds

regressors = ['News - Positive Sentiment', 'News - Negative Sentiment', 'News - New Products', 'Volume', 'News - Analyst Comments', 'News - Stocks']

# Calculate rolling predictions for each company WITHOUT regressors
rolling_predictions_results = {}
rolling_lower_bounds = {}
rolling_upper_bounds = {}
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]
    predictions, lower_bounds, upper_bounds = rolling_forecast_predictions(company_data, company)
    rolling_predictions_results[company] = predictions
    rolling_lower_bounds[company] = lower_bounds
    rolling_upper_bounds[company] = upper_bounds

# Calculate rolling predictions for each company WITH regressors
rolling_predictions_results_with_regressors = {}
rolling_lower_bounds_with_regressors = {}
rolling_upper_bounds_with_regressors = {}
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]
    predictions, lower_bounds, upper_bounds = rolling_forecast_predictions_with_regressors(company_data, company, regressors)
    rolling_predictions_results_with_regressors[company] = predictions
    rolling_lower_bounds_with_regressors[company] = lower_bounds
    rolling_upper_bounds_with_regressors[company] = upper_bounds

# Saving individual plots for each company
for company in chosen_companies:
    company_data = filtered_data[filtered_data['Symbol'] == company]

    # Normal Prophet Forecast without regressors
    forecast_without_regressors = create_forecast(company_data, company, use_regressors=False)

    plt.figure(figsize=(10, 6))
    plt.plot(company_data['Date'], company_data['Close'], label='Actual', color='blue')
    plt.plot(forecast_without_regressors['ds'], forecast_without_regressors['yhat'], label='Predicted', color='orange')
    plt.fill_between(forecast_without_regressors['ds'], 
                     forecast_without_regressors['yhat_lower'], 
                     forecast_without_regressors['yhat_upper'], 
                     color='gray', alpha=0.2, label='Uncertainty Interval')
    plt.title(f'{company} - Prophet Forecast Without Regressors')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f'prophet_forecast_without_regressors_{company}.png')
    plt.close()

    # Normal Prophet Forecast with regressors
    forecast_with_regressors = create_forecast(company_data, company, use_regressors=True)

    plt.figure(figsize=(10, 6))
    plt.plot(company_data['Date'], company_data['Close'], label='Actual', color='blue')
    plt.plot(forecast_with_regressors['ds'], forecast_with_regressors['yhat'], label='Predicted', color='orange')
    plt.fill_between(forecast_with_regressors['ds'], 
                     forecast_with_regressors['yhat_lower'], 
                     forecast_with_regressors['yhat_upper'], 
                     color='gray', alpha=0.2, label='Uncertainty Interval')
    plt.title(f'{company} - Prophet Forecast With Regressors')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f'prophet_forecast_with_regressors_{company}.png')
    plt.close()

    # Rolling forecast without regressors
    rolling_dates = pd.date_range(start=company_data['Date'].min(), 
                                   periods=len(rolling_predictions_results[company]), 
                                   freq='D')

    plt.figure(figsize=(10, 6))
    plt.plot(company_data['Date'], company_data['Close'], label='Actual', color='blue')
    plt.plot(rolling_dates, rolling_predictions_results[company], label='Predicted', color='orange')
    plt.fill_between(rolling_dates, 
                     rolling_lower_bounds[company], 
                     rolling_upper_bounds[company], 
                     color='gray', alpha=0.2, label='Uncertainty Interval')
    plt.title(f'{company} - Rolling Forecast Without Regressors')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.legend()
    plt.savefig(f'rolling_forecast_without_regressors_{company}.png')
    plt.close()

    # Rolling forecast with regressors
    rolling_dates_with_regressors = pd.date_range(start=company_data['Date'].min(), 
                                                   periods=len(rolling_predictions_results_with_regressors[company]), 
                                                   freq='D')

    plt.figure(figsize=(10, 6))
    plt.plot(company_data['Date'], company_data['Close'], label='Actual', color='blue')
    plt.plot(rolling_dates_with_regressors, rolling_predictions_results_with_regressors[company], label='Predicted', color='orange')
    plt.fill_between(rolling_dates_with_regressors, 
                     rolling_lower_bounds_with_regressors[company], 
                     rolling_upper_bounds_with_regressors[company], 
                     color='gray', alpha=0.2, label='Uncertainty Interval')
    plt.title(f'{company} - Rolling Forecast With Regressors')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.legend()
    plt.savefig(f'rolling_forecast_with_regressors_{company}.png')
    plt.close()

print("\nAll forecast graphs have been saved successfully.")





# /usr/local/bin/python3 /Users/aravindh/vscode/IT1244-Stock-Market/facebook_prophet.py
