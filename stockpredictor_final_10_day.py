import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Downloading data...")
ticker = "TSLA"

data = yf.download(ticker, start="2020-01-01")

data['MA_10'] = data['Close'].rolling(window=10).mean()

data['Target'] = data['Close'].shift(-1)

data = data.dropna()

X = data[['Close', 'MA_10']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train) 

predictions = model.predict(X_test)

actual_last = y_test.iloc[-1]
pred_last = predictions[-1]

print(f"\n--- Results for {ticker} ---")
print(f"Model's guess for the next day: ${pred_last:.2f}")
print(f"What actually happened: ${actual_last:.2f}")
print(f"Difference: ${abs(actual_last - pred_last):.2f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(y_test.values[-30:], label="Actual Closing Price", color="blue", linewidth=2, marker='o')

plt.plot(predictions[-30:], label="Model's Prediction", color="red", linestyle="--", linewidth=2, marker='x')

plt.title(f"Final Report: {ticker} Actual vs. Predicted Prices")
plt.xlabel("Days (Most Recent 30)")
plt.ylabel("Price in USD ($)")
plt.legend()
plt.grid(True)

plt.show()

# edit 10 day forecast ( margin of error is high due to error propogation and inaccurate recursive prediction
print("\nPredicting the next 10 days...")


current_batch = X.iloc[-1].values.reshape(1, -1)
future_predictions = []

for i in range(10):
    
    next_pred = model.predict(current_batch)[0]
    future_predictions.append(next_pred)
    
    
    
    current_batch = np.array([[next_pred, np.mean(future_predictions[-10:] + [next_pred])]])


print("Day | Predicted Price")
print("-" * 20)
for i, price in enumerate(future_predictions, 1):
    print(f"{i}   | ${price:.2f}")


plt.figure(figsize=(10,5))
plt.plot(range(1, 11), future_predictions, marker='o', color='green', linestyle='--')
plt.title(f"10-Day Forecast Trend for {ticker}")
plt.xlabel("Days into the Future")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
