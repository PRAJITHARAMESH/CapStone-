from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os
import datetime

app = Flask(__name__)

# Downloads folder create
DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Calculation function
def calculate_heat_transfer(thermal_cond, ambient_temp, source_temp, block_size):
    max_temp = source_temp
    avg_temp = (source_temp + ambient_temp) / 2
    center_temp = (source_temp + ambient_temp) / 2.5
    efficiency = ((max_temp - avg_temp) / max_temp) * 100

    # Status & coolant/material suggestion
    if efficiency > 60 or max_temp > 300:
        status = "Danger – Coolant Required!"
        coolant = "Water-Glycol Mix"
        material = "Copper"
    elif efficiency > 40:
        status = "Moderate – Consider Coolant"
        coolant = "Mineral Oil"
        material = "Aluminium"
    else:
        status = "Normal – No Coolant Needed"
        coolant = "None"
        material = "Aluminium"

    return {
        "max_temp": round(max_temp, 2),
        "avg_temp": round(avg_temp, 2),
        "center_temp": round(center_temp, 2),
        "efficiency": round(efficiency, 2),
        "status": status,
        "coolant": coolant,
        "material": material
    }

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    thermal_cond = float(request.form['thermal_cond'])
    ambient_temp = float(request.form['ambient_temp'])
    source_temp = float(request.form['source_temp'])
    block_size = float(request.form['block_size'])

    # Limits check
    if thermal_cond > 500 or ambient_temp > 100 or source_temp > 1000 or block_size > 100:
        return "⚠ Input values exceed limit! Please enter within allowed range."

    result = calculate_heat_transfer(thermal_cond, ambient_temp, source_temp, block_size)

    # Save result to CSV for download
    df = pd.DataFrame([result])
    filename = f"heat_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(DOWNLOAD_FOLDER, filename)
    df.to_csv(filepath, index=False)

    return render_template("result.html", result=result, download_filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
