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
    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS not found in Render Environment Variables!")
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Inputs from Form
        container = request.form.get('container') # e.g., Bio_A
        vol_ml = float(request.form.get('volume', 10000))
        nitrate = float(request.form.get('nitrate'))
        
        # 2. Average Triplicates
        ph_vals = [float(request.form.get(f'ph{i}')) for i in range(1, 4)]
        avg_ph = round(sum(ph_vals) / 3, 2)
        
        od_vals = [float(request.form.get(f'od{i}')) for i in range(1, 4)]
        avg_od = round(sum(od_vals) / 4, 4)

        # 3. Decision Logic: Nitrate (1ml stock = 6ppm/10L)
        kno3_advice = "Stable"
        if nitrate < 70:
            kno3_ml = ((80 - nitrate) / 6) * (vol_ml / 10000)
            kno3_advice = f"ADD {round(kno3_ml, 2)} ml KNO3"
        elif nitrate > 110:
            kno3_advice = "HIGH: Skip dosing"
        
        # 4. Decision Logic: pH Safety
        ph_warn = "HEALTHY"
        if avg_ph < 8.2: ph_warn = "CRITICAL: ACIDIC (Stop Pineapple)"
        elif avg_ph > 10.2: ph_warn = "WARNING: HIGH pH"

        # 5. AI Prediction (XGBoost)
        client = get_google_client()
        ws = client.open("Algae_Data_Master").worksheet(container)
        historical_data = ws.get_all_values()
        
        pineapple_dose, pred_od = train_and_predict_pineapple(historical_data, avg_ph, nitrate, vol_ml)

        # 6. Log to Google Sheets
        # Columns: Vol(L), Date, OD1, OD2, OD3, Avg_OD, pH1, pH2, pH3, Avg_pH, Nitrate
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_row = [vol_ml/1000, date_str, od_vals[0], od_vals[1], od_vals[2], avg_od,
                   ph_vals[0], ph_vals[1], ph_vals[2], avg_ph, nitrate]
        ws.append_row(new_row)

        return jsonify({
            "status": "success",
            "kno3": kno3_advice,
            "ph_warn": ph_warn,
            "pineapple": f"{pineapple_dose} ml",
            "prediction": f"Day +1 Forecast OD: {pred_od}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
