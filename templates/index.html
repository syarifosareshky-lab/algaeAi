import os, json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from xgboost import XGBRegressor
from datetime import datetime

app = Flask(__name__)

def get_sheet(sheet_name):
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    # Opens the master file and selects the tab based on Container ID (Bio_A, Bio_B, etc)
    return client.open("Algae_Data_Master").worksheet(sheet_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Get Inputs from UI
        container = request.form.get('container') # e.g. "Bio_A"
        ph = float(request.form.get('ph'))
        nitrate = float(request.form.get('nitrate'))
        vol = float(request.form.get('volume')) # In ml
        env = int(request.form.get('environment'))
        
        # 2. Decision Logic (Nitrate Rule: 1ml stock adds 6ppm per 10L)
        kno3_advice = "Stable"
        if nitrate < 70:
            shortfall = 80 - nitrate
            dose = (shortfall / 6) * (vol / 10000) # Math: (Gap / 6) * (Current Liters / 10)
            kno3_advice = f"ADD {round(dose, 2)} ml KNO3 Stock"
        
        # 3. Decision Logic (pH Safety)
        ph_status = "Healthy"
        if ph < 8.2: ph_status = "CRITICAL: ACIDIC (Stop Pineapple)"
        elif ph > 10.0: ph_status = "HIGH: Increase CO2/Air"

        # 4. Save to Google Sheet Tab
        ws = get_sheet(container)
        date_now = datetime.now().strftime("%Y-%m-%d")
        # Matches your sheet: Vol, Date, Blank OD, pH, Nitrate
        ws.append_row([vol/1000, date_now, "", "", "", "", ph, ph, ph, ph, nitrate])

        return jsonify({
            "status": "success",
            "kno3": kno3_advice,
            "ph_warn": ph_status,
            "pineapple": "10.0 ml" # This will come from your model_logic later
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
