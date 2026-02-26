
A Python tool to predict stock trends using Linear Regression and yfinance
# Stock Price Predictor ( quant project )
This project uses **Machine Learning** to forecast the next 10 days of stock prices. ( the 10 day forcast will be inaccurate as we are using a linear regression analysis (error propagation))

## How it Works
1. Pulls historical data using `yfinance`.
2. Calculates **Moving Averages** as technical indicators.
3. Trains a **Linear Regression** model via `scikit-learn`.
4. Visualizes the trend using `matplotlib`.

## Results
The model successfully predicted the trend for **TSLA**, **AAPL**, and **NVDA**.
