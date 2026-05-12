import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Generate Dummy Data (Oru wave pattern + konjam noise)
t = np.arange(0, 1000)
data = np.sin(0.02 * t) + np.random.uniform(-0.2, 0.2, 1000)
data = data.reshape(-1, 1)

# 2. Scaling (Data-vai 0-1 kulla mathurom - LSTM-ku ithu thaan romba mukkiyam)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create Training Sequences
# AI-kitta 50 naal data-vai kaati 51st day-ah predict panna solrom
X, y = [], []
for i in range(50, len(scaled_data)):
    X.append(scaled_data[i-50:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # Reshape for LSTM

# 4. Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the AI
print("AI is studying the patterns in time... ⏳")
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 6. Predict & Visualize
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

plt.plot(data[50:], color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='AI Prediction')
plt.title('Time Series Prediction - Day 16')
plt.legend()
plt.show()