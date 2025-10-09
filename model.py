
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import argparse

# --- Configuration Section ---
# Grouping parameters here makes them easier to adjust.
config = {
    "data_csv": "telemetry_data.csv",  # Path to your CSV data file. Leave empty to generate synthetic data.
    "seq_len": 32,
    "latent_dim": 16,
    "epochs": 50,
    "batch_size": 64,
    "out_dir": "out_model",
    "yellow_threshold_percentile": 90.0,  # Percentile for Yellow Signal
    "red_threshold_percentile": 98.0,    # Percentile for Red Signal
}

# --- Argument Parser (Optional, you can directly use the config dict) ---
parser = argparse.ArgumentParser()
parser.add_argument("--data-csv", type=str, default=config["data_csv"])
# ... add other arguments if you want to override config from command line
args = parser.parse_args()
config["data_csv"] = args.data_csv # Example of overriding

os.makedirs(config["out_dir"], exist_ok=True)

# --- Data Loading and Preprocessing (No changes needed here) ---
def load_or_generate_data(csv_path, n_samples=20000):
    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # arr = df[['temp','current']].values.astype(np.float32)
        arr = df[['temp_c','humidity_pct','current_mA','voltage_v']].values.astype(np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-6
        arr = (arr - mean) / std
        return arr, mean, std
    else:
        # Generate synthetic normal operating data
        t = np.linspace(0, 500*np.pi, n_samples)
        temp = 35.0 + 5.0*np.sin(t*0.01) + 0.5*np.random.randn(n_samples)
        current = 2.0 + 0.5*np.sin(t*0.007 + 0.5) + 0.05*np.random.randn(n_samples)
        arr = np.stack([temp, current], axis=1).astype(np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-6
        arr = (arr - mean) / std
        return arr, mean, std

arr, data_mean, data_std = load_or_generate_data(config["data_csv"])
print("Data shape:", arr.shape)
print(f"Data Mean (Temp, Current): {data_mean}")
print(f"Data Std Dev (Temp, Current): {data_std}")


def make_sequences(x, seq_len):
    seqs = np.stack([x[i:i+seq_len] for i in range(len(x) - seq_len + 1)], axis=0)
    return seqs

seqs = make_sequences(arr, config["seq_len"])
n_train = int(0.8 * len(seqs))
x_train, x_val = seqs[:n_train], seqs[n_train:]

# --- Model Building and Training (No changes needed here) ---
inputs = layers.Input(shape=(config["seq_len"], 4))
encoded = layers.LSTM(config["latent_dim"])(inputs)
repeat = layers.RepeatVector(config["seq_len"])(encoded)
decoded = layers.LSTM(config["latent_dim"], return_sequences=True)(repeat)
outputs = layers.TimeDistributed(layers.Dense(4))(decoded)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x_train, x_train,
          epochs=config["epochs"],
          batch_size=config["batch_size"],
          validation_data=(x_val, x_val))

h5_path = os.path.join(config["out_dir"], "model.h5")
model.save(h5_path)
print("Saved Keras model:", h5_path)


# --- TFLite Conversion (No changes needed here) ---
def representative_data_gen():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

# --- Fixed TFLite Conversion ---
def representative_data_gen():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TensorFlow Select ops (for LSTM TensorList)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Optional: Float16 quantization (safe for LSTM)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Enable TensorList lowering support
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

tflite_path = os.path.join(config["out_dir"], "model_float16.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Saved compatible TFLite model:", tflite_path)


tflite_path = os.path.join(config["out_dir"], "model_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite model:", tflite_path)


# --- IMPROVEMENT: Calculate and Save Thresholds and Normalization Parameters ---

print("\nCalculating alert thresholds...")
# Reconstruct the validation data to get the errors
recon = model.predict(x_val)
# Calculate the Mean Squared Error for each sequence
mse = np.mean((recon - x_val)**2, axis=(1,2))

# Calculate the two thresholds based on percentiles
yellow_threshold = np.percentile(mse, config["yellow_threshold_percentile"])
red_threshold = np.percentile(mse, config["red_threshold_percentile"])

print(f"🟩 -> 🟨 Yellow Threshold (p{config['yellow_threshold_percentile']}): {yellow_threshold:.6f}")
print(f"🟨 -> 🟥 Red Threshold (p{config['red_threshold_percentile']}): {red_threshold:.6f}")

# --- NEW SECTION: Determine a single alert level based on thresholds ---
# Compute the average reconstruction error for validation data
avg_mse = float(np.mean(mse))

# Decide final alert level
if avg_mse >= red_threshold:
    alert_level = "RED"      # Critical
elif avg_mse >= yellow_threshold:
    alert_level = "YELLOW"   # Warning
else:
    alert_level = "GREEN"    # Normal

# Print and (optionally) save the final alert result
print(f"🚨 Final Alert Level: {alert_level} (avg MSE = {avg_mse:.6f})")

# You can also include it in the saved parameters if needed:
params_path = os.path.join(config["out_dir"], "model_params.npz")
np.savez(params_path,
         mean=data_mean,
         std=data_std,
         yellow_threshold=yellow_threshold,
         red_threshold=red_threshold,
         alert_level=alert_level)
print(f"Saved all deployment parameters to: {params_path}")


# Save all necessary parameters for deployment in a single file
params_path = os.path.join(config["out_dir"], "model_params.npz")
# np.savez(params_path,
#          mean=data_mean,
#          std=data_std,
#          yellow_threshold=yellow_threshold,
#          red_threshold=red_threshold)

print(f"Saved all deployment parameters to: {params_path}")

