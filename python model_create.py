# model_create.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- Step 1: Dummy dataset create ----------
np.random.seed(42)  # repeatable results
data = pd.DataFrame({
    'ThermalCond': np.random.randint(50, 200, 50),   # W/mK
    'BlockSize': np.random.randint(5, 20, 50),       # cm
    'SourceTemp': np.random.randint(40, 100, 50),    # °C
    'AmbientTemp': np.random.randint(20, 35, 50)     # °C
})

# Target variable (example)
data['Target'] = (
    (data['SourceTemp'] - data['AmbientTemp']) / (data['ThermalCond'] / 100)
) - (data['BlockSize'] / 10)

# ---------- Step 2: Feature & Target split ----------
X = data[['ThermalCond', 'BlockSize', 'SourceTemp', 'AmbientTemp']]
y = data['Target']

# ---------- Step 3: Scale features ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Step 4: Train model ----------
model = LinearRegression()
model.fit(X_scaled, y)

# ---------- Step 5: Save model & scaler ----------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ model.pkl & scaler.pkl created successfully!")
