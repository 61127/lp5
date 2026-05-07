# =========================================
# GOOGLE STOCK PRICE PREDICTION USING RNN
# =========================================

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("GOOGL.csv")

print(df.head())

# Use only Close price
data = df[['Close']]

# =========================================
# 3. NORMALIZE DATA
# =========================================
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

# =========================================
# 4. CREATE TRAINING DATA
# =========================================
X = []
y = []

time_step = 60

for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)    

# =========================================
# 5. TRAIN-TEST SPLIT
# =========================================
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# =========================================
# 6. BUILD RNN MODEL
# =========================================
model = Sequential([
    LSTM(50, return_sequences=True,
         input_shape=(X_train.shape[1], 1)),

    LSTM(50),

    Dense(1)
])
# =========================================
# 7. COMPILE MODEL
# =========================================
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# =========================================
# 8. TRAIN MODEL
# =========================================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

# =========================================
# 9. PREDICTIONS
# =========================================
predictions = model.predict(X_test)

# Convert back to original values
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1,1)
)

# =========================================
# 10. PLOT RESULTS
# =========================================
plt.figure(figsize=(12,6))

plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')

plt.title("Google Stock Price Prediction using RNN")
plt.xlabel("Time")
plt.ylabel("Stock Price")

plt.legend()
plt.show()