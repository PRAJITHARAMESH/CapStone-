from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        thermal_cond = float(request.form['thermal_cond'])
        block_size = float(request.form['block_size'])
        source_temp = float(request.form['source_temp'])
        ambient_temp = float(request.form['ambient_temp'])

        # Prepare input array
        input_data = np.array([[thermal_cond, block_size, source_temp, ambient_temp]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Example logic for efficiency and suggestions
        efficiency = prediction[0]
        if efficiency > 70:
            status = "Good performance"
            suggestion = "No cooling needed."
        elif efficiency > 50:
            status = "Average performance"
            suggestion = "Consider adding a cooling mechanism."
        else:
            status = "Poor performance"
            suggestion = "Add a high-efficiency cooling system."

        return render_template(
            'result.html',
            efficiency=round(efficiency, 2),
            status=status,
            suggestion=suggestion
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
