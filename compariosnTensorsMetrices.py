import tensorflow as tf
import numpy as np
import time

# Parameters
batch_size = 64
time_steps = 10
features = 5

# Generate dummy 3D tensor data
data_tensor = np.random.random((batch_size, time_steps, features)).astype(np.float32)

# Flatten the data to create a matrix representation
data_matrix = data_tensor.reshape(batch_size, -1)

# RNN Model for Tensor Input
model_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=(time_steps, features)),
    tf.keras.layers.SimpleRNN(50),
    tf.keras.layers.Dense(1)
])

# Dense Model for Matrix Input
model_dense = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(time_steps * features,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Time the RNN model
start_time_rnn = time.time()
predictions_rnn = model_rnn.predict(data_tensor)
end_time_rnn = time.time()

# Time the Dense model
start_time_dense = time.time()
predictions_dense = model_dense.predict(data_matrix)
end_time_dense = time.time()

# Print results
print(f"RNN Model Time: {end_time_rnn - start_time_rnn} seconds")
print(f"Dense Model Time: {end_time_dense - start_time_dense} seconds")


