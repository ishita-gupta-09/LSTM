# ⚙️ LSTM-Based Predictive Maintenance & Anomaly Detection

A deep learning project that uses **Long Short-Term Memory (LSTM)** networks to:
- 🔮 Predict the **Remaining Useful Life (RUL)** of machines  
- 🚨 Detect **anomalies and degradation patterns** using sensor data  
- ⚡ Export optimized **TensorFlow Lite** models for real-world deployment  

---

## 🧠 Project Overview

This repository demonstrates two complementary applications of LSTMs in predictive maintenance:

1. **`main.py`** → Predicts **Remaining Useful Life (RUL)** using multivariate sensor data.  
2. **`model.py`** → Builds an **LSTM autoencoder** for anomaly detection and alert thresholding (Green/Yellow/Red).

Both scripts are designed to work independently, showcasing **data-driven** and **reconstruction-error-based** approaches to machine health monitoring.

---

## 📂 Repository Structure

```
LSTM/
├── main.py             # Predictive Maintenance (RUL prediction with alerts)
├── model.py            # LSTM Autoencoder for Anomaly Detection
├── out_model/          # Folder for saved models and parameters
└── README.md           # Documentation (this file)
```

---

## 🧩 Features

### 🔹 Predictive Maintenance (`main.py`)
- Generates or loads multivariate sensor data (temperature, voltage, vibration, current).  
- Preprocesses and normalizes data for training.  
- Builds a **stacked LSTM** network to predict Remaining Useful Life (RUL).  
- Includes **hybrid alert logic** (rule-based + ML-based):
  - 🟩 **GREEN** → Normal operation  
  - 🟨 **YELLOW** → Maintenance required soon  
  - 🟥 **RED** → Imminent failure  

### 🔹 Anomaly Detection (`model.py`)
- Creates an **LSTM Autoencoder** to learn “normal” operating behavior.  
- Detects deviations using **reconstruction error (MSE)** thresholds:
  - 🟩 Normal  
  - 🟨 Warning (above 95th percentile)  
  - 🟥 Critical (above 99.5th percentile)  
- Converts trained models to **TensorFlow Lite (TFLite)** for edge deployment.

---

## 🧱 Model Architectures

### 🧮 LSTM Autoencoder (model.py)
```
Input (seq_len, 2) → LSTM(16) → RepeatVector → LSTM(16, return_sequences=True) → Dense(2)
```

### 🔧 Predictive LSTM (main.py)
```
Input (seq_len, 4) → LSTM(100, return_sequences=True)
                  → Dropout(0.2)
                  → LSTM(50)
                  → Dense(25, ReLU)
                  → Dense(1)  # Predicted RUL
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ishita-gupta-09/LSTM.git
cd LSTM
```

### 2️⃣ Install Dependencies
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### 3️⃣ (Optional) Create Output Folder
```bash
mkdir out_model
```

---

## 🚀 How to Run

### ▶️ **Train Predictive Maintenance Model**
```bash
python main.py
```
- Simulates sensor data for a degrading machine.
- Trains an LSTM model for RUL prediction.
- Displays alert messages based on predicted RUL and thresholds.

Example Output:
```
Training data shape: (68, 15, 4)
Testing data shape: (18, 15, 4)
--- Training Model ---
...
Alert Status: RED | Message: Critical failure predicted soon. Predicted RUL: 5.12
```

---

### ▶️ **Train Autoencoder & Convert to TFLite**
```bash
python model.py
```
- Trains an LSTM autoencoder on either synthetic or CSV-based data.
- Computes percentile-based thresholds.
- Exports:
  - `out_model/model.h5`
  - `out_model/model_int8.tflite`
  - `out_model/model_params.npz` (mean, std, thresholds)

Example Output:
```
🟩 -> 🟨 Yellow Threshold (p95): 0.002314
🟨 -> 🟥 Red Threshold (p99.5): 0.007215
Saved TFLite model: out_model/model_int8.tflite
```

---

## 📊 Outputs

| File | Description |
|------|--------------|
| `model.h5` | Trained Keras model |
| `model_int8.tflite` | Quantized model for deployment |
| `model_params.npz` | Contains mean, std, and thresholds |
| `out_model/` | Directory for all trained models and parameters |

---

## 🧪 Example Alert Generation

Example alerts generated from the predictive maintenance pipeline:

| Alert | Condition | Description |
|--------|------------|-------------|
| 🟩 GREEN | RUL > 30 | Normal operation |
| 🟨 YELLOW | RUL ≤ 30 | Maintenance recommended soon |
| 🟥 RED | RUL ≤ 10 or temp/vibration critical | Immediate attention required |

---

## 📈 Future Enhancements
- Integrate real-time data streaming (e.g., MQTT / Kafka).  
- Add dashboard visualization (Plotly / Streamlit).  
- Extend to multivariate anomaly detection using CNN-LSTM hybrids.  
- Deploy TFLite model on Raspberry Pi or IoT devices.  

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | TensorFlow Lite (Edge AI) |

---

## 👩‍💻 Author

**Ishita Gupta**  
💡 Passionate about AI, Predictive Analytics, and Edge Computing  
📬 [GitHub Profile](https://github.com/ishita-gupta-09)

---

## 📜 License

This project is released under the **MIT License**.  
You’re free to use, modify, and distribute with attribution.

---

> ✨ _"Predict the future — before it breaks."_  
> — LSTM for Smart Maintenance
