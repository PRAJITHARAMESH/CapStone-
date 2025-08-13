import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# Flask app
app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def suggest_coolant_and_material(efficiency):
    if efficiency > 80:
        return "Coolant Not Needed", "Aluminium"
    elif efficiency > 60:
        return "Use Water-Based Coolant", "Copper"
    else:
        return "Use Ethylene Glycol Coolant", "High Thermal Conductivity Alloy"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        thermal_cond = float(request.form['thermal_cond'])
        block_size = float(request.form['block_size'])
        source_temp = float(request.form['source_temp'])
        ambient_temp = float(request.form['ambient_temp'])

        # Prepare & scale
        input_data = np.array([[thermal_cond, block_size, source_temp, ambient_temp]])
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        efficiency = max(0, min(100, 100 - abs(prediction)))

        coolant, material = suggest_coolant_and_material(efficiency)

        return render_template("result.html",
                               prediction=round(prediction, 2),
                               efficiency=round(efficiency, 2),
                               coolant=coolant,
                               material=material)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
