from flask import Flask, request, jsonify
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from xgboost import XGBRegressor
import os

app = Flask(__name__)

# 1. Connect to Google Sheets
def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    # On Render, you will paste your JSON credentials into an Environment Variable
    creds_dict = eval(os.environ.get("GOOGLE_CREDENTIALS")) 
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open("Algae_Data_Master").get_worksheet(0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sheet = get_sheet()
        raw_data = sheet.get('A1:H1000')
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])

        # Clean Data
        cols = ['pH', 'Nitrate', 'Pineapple_ml', 'Volume_ml', 'Environment', 'Target_OD']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['pH', 'Target_OD'])

        # Train Model
        X = df[['pH', 'Nitrate', 'Pineapple_ml', 'Volume_ml', 'Environment']]
        y = df['Target_OD']
        model = XGBRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(X, y)

        # Optimize for 1.0 OD
        best_dose = 0
        best_pred = 0
        diff = 999
        for dose in [0, 5, 10, 15, 20]:
            pred = model.predict(pd.DataFrame([[data['ph'], data['nitrate'], dose, data['volume'], data['env']]], 
                                            columns=X.columns))[0]
            if abs(1.0 - pred) < diff:
                diff = abs(1.0 - pred)
                best_dose = dose
                best_pred = pred

        return jsonify({"dose": best_dose, "prediction": round(float(best_pred), 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check for UptimeRobot
@app.route('/')
def home():
    return "AI Server is Online"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
