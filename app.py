import os
import json
from flask import Flask, render_template, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# Import the AI brain from your other file
from model_logic import train_and_predict_pineapple

app = Flask(__name__)

# --- GOOGLE SHEETS CONNECTION CONFIG ---
def get_google_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set on Render!")
    
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

# --- ROUTES ---

@app.route('/')
def index():
    """Serves the professional dashboard UI"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Capture Data from the Dashboard Form
        container = request.form.get('container') # Must match Tab Name (e.g., Bio_A)
        ph = float(request.form.get('ph'))
        nitrate = float(request.form.get('nitrate'))
        vol_ml = float(request.form.get('volume'))
        env = int(request.form.get('environment'))
        
        # 2. Access Google Sheets
        client = get_google_client()
        spreadsheet = client.open("Algae_Data_Master")
        
        try:
            ws = spreadsheet.worksheet(container)
        except Exception:
            return jsonify({"status": "error", "message": f"Tab '{container}' not found in Google Sheets!"})

        # 3. Decision Support: Nitrate Rule (1ml stock = 6ppm/10L)
        kno3_advice = "Status: Stable"
        if nitrate < 70:
            shortfall = 80 - nitrate
            # Formula: (Gap / 6ppm) * (Current Volume in Liters / 10L)
            kno3_ml = (shortfall / 6) * (vol_ml / 10000)
            kno3_advice = f"ACTION: Add {round(kno3_ml, 2)} ml KNO3 Stock"
        elif nitrate > 100:
            kno3_advice = "ACTION: High Nitrate - Do not dose"

        # 4. Decision Support: pH Safety Warning
        ph_status = "System: Healthy"
        if ph < 8.2:
            ph_status = "CRITICAL: Acidic! Stop Pineapple feeding."
        elif ph > 10.2:
            ph_status = "WARNING: High pH! Increase CO2/Air."

        # 5. AI Brain: Get Pineapple Prediction
        # Fetch all data from the sheet to train the model
        historical_data = ws.get_all_values()
        pineapple_dose, predicted_od = train_and_predict_pineapple(
            historical_data, ph, nitrate, vol_ml, env
        )

        # 6. Data Logging: Append to the correct Google Sheet tab
        date_str = datetime.now().strftime("%Y-%m-%d")
        # Row layout matching your sheet: Vol(L), Date, [Empty ODs], [Empty pHs], Nitrate
        # We fill only one pH column and the nitrate column for consistency
        new_row = [vol_ml/1000, date_str, "", "", "", "", ph, "", "", "", nitrate]
        ws.append_row(new_row)

        # 7. Return Results to the Dashboard
        return jsonify({
            "status": "success",
            "kno3": kno3_advice,
            "ph_warn": ph_status,
            "pineapple": f"{pineapple_dose} ml",
            "prediction": f"Day +1 Predicted OD: {predicted_od}"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health')
def health():
    return "AI Dashboard Active", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
