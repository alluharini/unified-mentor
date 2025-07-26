import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("Coca-Cola_stock_history.csv")

data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
data['Date'] = data['Date'].dt.tz_localize(None)  
data.sort_values('Date', inplace=True)

data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)

data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.plot(data['Date'], data['MA_20'], label='MA 20', linestyle='--', color='orange')
plt.plot(data['Date'], data['MA_50'], label='MA 50', linestyle='--', color='green')
plt.title('Coca-Cola Stock Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 
            'Stock Splits', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Closing Price", color='blue')
plt.plot(y_pred, label="Predicted Closing Price", color='red', linestyle='--')
plt.title("Actual vs Predicted Coca-Cola Closing Prices")
plt.xlabel("Time (Test Data Index)")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()

errors = y_test.values - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.title("Prediction Error Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

latest_data = X.iloc[-1:]
predicted_price = model.predict(latest_data)[0]

last_30_days = data.tail(30)
plt.figure(figsize=(12, 6))
plt.plot(last_30_days['Date'], last_30_days['Close'], label='Last 30 Days Close', color='blue')
plt.scatter(last_30_days['Date'].iloc[-1] + pd.Timedelta(days=1), predicted_price, 
            label='Predicted Next Close', color='red', s=100)
plt.title('Last 30 Days Close Prices & Next Day Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
