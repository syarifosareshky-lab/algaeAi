import os, json
from flask import Flask, render_template, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from model_logic import train_and_predict_pineapple

app = Flask(__name__)

def get_google_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        container = request.form.get('container') # e.g., Bio_A
        ph = float(request.form.get('ph'))
        nitrate = float(request.form.get('nitrate'))
        vol_ml = float(request.form.get('volume'))
        env = int(request.form.get('environment'))
        
        client = get_google_client()
        ws = client.open("Algae_Data_Master").worksheet(container)

        # Expert System: Nitrate Rule
        kno3_advice = "Status: Stable"
        if nitrate < 70:
            shortfall = 80 - nitrate
            kno3_ml = (shortfall / 6) * (vol_ml / 10000)
            kno3_advice = f"ACTION: Add {round(kno3_ml, 2)} ml KNO3 Stock"

        # Expert System: pH Safety
        ph_status = "System: Healthy"
        if ph < 8.2: ph_status = "CRITICAL: Acidic! Stop Pineapple."
        elif ph > 10.2: ph_status = "WARNING: High pH! Increase Air."

        # AI Prediction
        historical_data = ws.get_all_values()
        pineapple_dose, pred_od = train_and_predict_pineapple(historical_data, ph, nitrate, vol_ml, env)

        # Log Data to Sheet
        date_str = datetime.now().strftime("%Y-%m-%d")
        ws.append_row([vol_ml/1000, date_str, "", "", "", "", ph, "", "", "", nitrate])

        return jsonify({
            "status": "success",
            "kno3": kno3_advice,
            "ph_warn": ph_status,
            "pineapple": f"{pineapple_dose} ml",
            "prediction": f"Day +1 Predicted OD: {pred_od}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
