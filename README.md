# Provide step-by-step instructions on how to run our project. This is on top of our comments in the code folder which explains the steps work.
# Principal Component Analysis (PCA):
- Load your training data
- Under creation of PCA dataframe and pca_components, you can change the number of PC you want to see

# Autoregressive Integrated Moving Average (ARIMA):
- Load Training Data: Import the dataset and filter it for the company of interest.
- Define Training Duration: Select the number of days to use from the training data.
- Parameter Selection: Set the ARIMA model parameters (p, d, q) based on the data's patterns and model requirements.
- Set Rolling-Window Cross-Validation Parameters: Define train_size, test_size, and roll_window for iterative forecasting.
- Select Data for Plotting: Choose the amount of historical data to visualize if not using the full dataset.
- Load Test Data: Import and filter the test dataset for the same company.
- Define Test Duration: Specify the number of days to use from the test data.
- Forecast: Generate predictions for the specified test set length to evaluate the model’s performance.

# Holt-Winter's Exponential Smoothing (HWES):
- Load Training Data: Import the dataset and filter it for the company of interest.
- Define Training Duration: Select the number of days to use from the training data.
- Set Rolling-Window Cross-Validation Parameters: Define train_size, test_size, and roll_window for iterative forecasting.
- Select Data for Plotting: Choose the amount of historical data to visualize if not using the full dataset.
- Load Test Data: Import and filter the test dataset for the same company.
- Define Test Duration: Specify the number of days to use from the test data.
- Forecast: Generate predictions for the specified test set length to evaluate the model’s performance.

# Long Short-Term Memory Neural Network (LSTM):
- Load Training Data: Import the dataset and filter it for the company of interest.
- Define Training Duration: Select the number of days to use from the training data.
- Data Preparation: Normalize the data and reshape it to the required LSTM format (samples, time_steps, features).
- Set Rolling-Window Cross-Validation Parameters: Define train_size, test_size, and roll_window for iterative forecasting.
- Select Data for Plotting: Choose the amount of historical data to visualize if not using the full dataset.
- Load Test Data: Import and filter the test dataset for the same company.
- Define Test Duration: Specify the number of days to use from the test data.
- Forecast: Generate predictions for the specified test set length to evaluate the model’s performance.

# Facebook Prophet (FP):
This section contains a Python implementation for forecasting stock prices using the Facebook Prophet library. The model incorporates additional regressors derived from sentiment analysis and other company-related data.
Requirements
 • Python 3.10
 • pandas
 • prophet
 • matplotlib
 • scikit-learn
 • numpy
Data
The model uses a CSV file named merged_company_data.csv that includes the following columns:
 • Date: Date of the stock price.
 • Close: Closing price of the stock.
 • Symbol: Stock ticker symbol.
 • News-related columns representing various sentiment metrics.
Selected Companies
The model focuses on the following randomly chosen companies:
 • XOM
 • SHW
 • AMZN
 • PG
 • JNJ
 • JPM
 • AAPL
 • GOOGL
 • NEE
 • AMT
 • MMM

Model Implementation

 1. Load Training Data
Load your training data by running the script that reads merged_company_data.csv.

 2. Select Company
To forecast stock prices, select your company of interest by typing their symbol from the chosen_companies list. Feel free to alter this list.

 3. Configure Forecasting Parameters
Specify the following parameters:
 • Future Periods: Number of days to forecast (default is 365).
 • Custom Seasonalities: Optionally define monthly and quarterly seasonalities.
 • Holidays: Define market holidays if applicable (I used US holidays as SNP 500 is very US based anyway)

 4. Training the Model
Train the model using the selected company’s historical stock price data. You can choose to include regressors (e.g., News - Positive Sentiment) to improve forecast accuracy.

 5. Perform Rolling Forecasting
 • Specify rolling-window cross-validation parameters:
 • train_size: Size of the training dataset.
 • test_size: Size of the testing dataset.
 • roll_window: Rolling window size for cross-validation.

 6. Load Test Data
Load your test dataset by selecting the same company. Optionally, define how many days you wish to test if not using the entire test set.

 7. Forecasting
Forecast against the length of the test set used. The script will output predictions for both models (with and without regressors).

 8. Visualization
The model generates and saves forecast plots for:
 • Normal forecasts without regressors.
 • Normal forecasts with regressors.
 • Rolling forecasts without regressors.
 • Rolling forecasts with regressors.
Plots are saved in the current directory with filenames formatted as prophet_forecast_without_regressors_{company}.png and rolling_forecast_with_regressors_{company}.png.

 9. Evaluation Metrics
The model calculates prediction errors using:
 • Mean Absolute Percentage Error (MAPE)
 • Root Mean Squared Error (RMSE)

 10. Running the Model
To execute the model, run the Python script:
stock_price_forecast.py
Do fine tune your own hyperparameters that is in tune with your desired outcome.