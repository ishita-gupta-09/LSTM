import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import io

# --- 1. Data Simulation & Loading ---
# In your project, you'll replace this with: pd.read_csv('your_sensor_data.csv')
# I'm creating a sample dataset that simulates a machine degrading over time.
# A real dataset would have thousands of points from multiple machine cycles.

csv_data = """
timestamp,temperature,voltage,vibration,current
1,40.1,219.8,0.11,5.0
2,40.3,220.0,0.13,5.1
3,40.2,220.1,0.12,5.0
...
90,45.5,221.0,0.35,4.9
91,46.1,221.2,0.38,4.8
92,46.8,221.5,0.45,4.7
93,47.5,221.8,0.55,4.5
94,48.9,222.1,0.70,4.2
95,50.2,222.5,0.95,3.9
96,52.0,223.0,1.35,3.5
97,54.1,223.4,1.80,3.1
98,56.5,223.9,2.40,2.5
99,59.0,224.5,3.10,2.1
100,62.0,225.0,4.00,1.5
"""
# This part is just to make the sample data longer for the simulation
# In a real scenario, you would have this data already
# We simulate a 'healthy' period followed by the degradation from the csv_data
healthy_data = pd.DataFrame({
    'timestamp': range(1, 81),
    'temperature': np.linspace(40, 42, 80) + np.random.normal(0, 0.2, 80),
    'voltage': np.linspace(220, 220.5, 80) + np.random.normal(0, 0.1, 80),
    'vibration': np.linspace(0.1, 0.15, 80) + np.random.normal(0, 0.02, 80),
    'current': np.linspace(5.0, 4.9, 80) + np.random.normal(0, 0.05, 80)
})
degrading_data = pd.read_csv(io.StringIO(csv_data.replace("...", "")))
df = pd.concat([healthy_data, degrading_data.iloc[9:]], ignore_index=True)

# --- 2. Data Preprocessing & Feature Engineering ---

# Define the features to be used by the model
features = ['temperature', 'voltage', 'vibration', 'current']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# ** CRUCIAL STEP: Create Remaining Useful Life (RUL) labels **
# Here, we assume the last entry is the point of failure.
# RUL is 0 at the end and increases as we go back in time.
df['RUL'] = len(df) - 1 - df.index


# --- 3. Create Sequences for the LSTM ---
# LSTMs learn from a sequence of past data to predict the future.
def create_sequences(data, labels, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)


SEQUENCE_LENGTH = 15  # Use the last 15 data points to make a prediction
X, y = create_sequences(df[features].values, df['RUL'].values, SEQUENCE_LENGTH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 4. Build the LSTM Model ---
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1))  # Output layer: predicts a single value (the RUL)

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 5. Train the Model ---
# In a real project, you would increase epochs for better training.
print("\n--- Training Model ---")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
print("--- Model Training Complete ---")


# --- 6. Prediction and Alert Generation ---
def predict_and_generate_alert(model, data_sequence, current_values):
    """
    Predicts RUL and generates an alert based on thresholds.

    Args:
        model: The trained Keras model.
        data_sequence (np.array): A sequence of scaled sensor data (shape: 1, sequence_length, num_features).
        current_values (pd.Series): The most recent UN-SCALED sensor values for rule-based checks.

    Returns:
        dict: A dictionary containing the alert level, predicted RUL, and a message.
    """
    # Thresholds for alerts (you should tune these based on your equipment)
    YELLOW_ALERT_RUL = 30  # Trigger yellow alert if RUL is less than 30 cycles/time units
    RED_ALERT_RUL = 10  # Trigger red alert if RUL is less than 10

    # Rule-based thresholds for immediate red alerts
    RED_ALERT_VIBRATION = 3.0  # Vibration level that is immediately dangerous
    RED_ALERT_TEMP = 60.0  # Temperature that is immediately dangerous

    # --- Rule-based check for immediate danger ---
    if current_values['vibration'] > RED_ALERT_VIBRATION or current_values['temperature'] > RED_ALERT_TEMP:
        return {
            "alert": "RED",
            "predicted_rul": 0,
            "message": f"Immediate danger detected! Vibration: {current_values['vibration']:.2f} or Temp: {current_values['temperature']:.2f} exceeded critical threshold."
        }

    # --- ML-based prediction ---
    predicted_rul = model.predict(data_sequence)[0][0]

    if predicted_rul < RED_ALERT_RUL:
        return {
            "alert": "RED",
            "predicted_rul": predicted_rul,
            "message": f"Critical failure predicted soon. Predicted RUL: {predicted_rul:.2f} time units."
        }
    elif predicted_rul < YELLOW_ALERT_RUL:
        return {
            "alert": "YELLOW",
            "predicted_rul": predicted_rul,
            "message": f"Maintenance required. Equipment degrading. Predicted RUL: {predicted_rul:.2f} time units."
        }
    else:
        return {
            "alert": "GREEN",
            "predicted_rul": predicted_rul,
            "message": f"Equipment is operating normally. Predicted RUL: {predicted_rul:.2f} time units."
        }


# --- Example Usage ---
print("\n--- Generating Example Alerts ---")

# Take the last sequence from our test data to simulate a new reading near failure
last_sequence_scaled = np.expand_dims(X_test[-1], axis=0)
last_sequence_unscaled = scaler.inverse_transform(X_test[-1])
# Get the most recent values from the unscaled sequence for the rule-based check
most_recent_unscaled_values = pd.Series(last_sequence_unscaled[-1], index=features)

alert_status = predict_and_generate_alert(model, last_sequence_scaled, most_recent_unscaled_values)
print(f"Alert Status: {alert_status['alert']} | Message: {alert_status['message']}")

# Take a sequence from the beginning (healthy part)
first_sequence_scaled = np.expand_dims(X_train[0], axis=0)
first_sequence_unscaled = scaler.inverse_transform(X_train[0])
most_recent_unscaled_values_healthy = pd.Series(first_sequence_unscaled[-1], index=features)

alert_status_healthy = predict_and_generate_alert(model, first_sequence_scaled, most_recent_unscaled_values_healthy)
print(f"Alert Status (Healthy): {alert_status_healthy['alert']} | Message: {alert_status_healthy['message']}")