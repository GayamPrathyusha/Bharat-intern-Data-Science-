import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

Load the data from the CSV file
df = pd.read_csv("HCLTECH.csv")

# Use the 'Close' price as the target variable
data = df["Close"].values.reshape(-1, 1)

Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Function to create sequences for time series data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences

# Define sequence length
seq_length = 10

# Create sequences for training and testing data
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Convert the sequences into NumPy arrays
X_train, y_train = zip(*train_sequences)
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = zip(*test_sequences)
X_test, y_test = np.array(X_test), np.array(y_test)

# Build an LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test data
predictions = model.predict(X_test)

# Inverse transform the predictions to get original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Visualize the results
plt.figure(figsize=(18, 6))
plt.plot(predictions, label='Predicted Price', linestyle='dashed')
plt.plot(y_test, label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
