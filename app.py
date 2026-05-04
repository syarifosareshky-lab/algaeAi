import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from xgboost import XGBRegressor
from datetime import datetime

app = Flask(__name__)

# --- GOOGLE SHEETS CONNECTION ---
def get_worksheet():
    # This reads the JSON string you will paste into Render's Environment Variables
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if not creds_json:
        raise ValueError("No GOOGLE_CREDENTIALS found in Environment Variables!")
    
    # Load the JSON string into a dictionary
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    
    # Ensure this matches your Google Sheet filename exactly
    return client.open("Algae_Data_Master").get_worksheet(0)

# --- AI BRAIN (XGBOOST) ---
def train_and_predict(ph, nitrate, volume, env):
    ws = get_worksheet()
    raw = ws.get_all_values()
    
    # Create DataFrame and clean headers
    df = pd.DataFrame(raw[1:], columns=raw[0])
    
    # Convert columns to numbers so the AI can do math
    cols_to_fix = ['pH', 'Nitrate', 'Pineapple_ml', 'Volume_ml', 'Environment', 'Target_OD']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only train on rows that have a result (Target_OD)
    train_df = df.dropna(subset=['pH', 'Target_OD'])
    
    if len(train_df) < 2:
        # Not enough data yet? Return a safe default dose
        return 10.0, 0.500

    # Features (X) and Label (y)
    X = train_df[['pH', 'Nitrate', 'Pineapple_ml', 'Volume_ml', 'Environment']]
    y = train_df['Target_OD']
    
    # XGBoost setup
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    model.fit(X, y)
    
    # OPTIMIZATION LOOP: 
    # The AI tests different doses (0 to 20ml) to see which gets closest to 1.0 OD
    best_dose = 0
    best_pred = 0
    min_diff = 999
    
    for test_dose in [0, 2, 5, 8, 10, 12, 15, 20]:
        # Context: env 1 = Bio, env 0 = Pond
        input_data = pd.DataFrame([[ph, nitrate, test_dose, volume, env]], columns=X.columns)
        prediction = model.predict(input_data)[0]
        
        diff = abs(1.0 - prediction)
        if diff < min_diff:
            min_diff = diff
            best_dose = test_dose
            best_pred = prediction
            
    return float(best_dose), round(float(best_pred), 3)

# --- WEB ROUTES ---

@app.route('/')
def index():
    # This serves your nice UI from templates/index.html
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. Capture data from the Professional Dashboard
        container = request.form.get('container')
        ph = float(request.form.get('ph'))
        nitrate = float(request.form.get('nitrate'))
        vol = float(request.form.get('volume'))
        env = int(request.form.get('environment')) # 1 or 0
        date_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 2. Get AI Recommendation
        suggested_dose, predicted_od = train_and_predict(ph, nitrate, vol, env)
        
        # 3. Store in Google Sheets automatically
        ws = get_worksheet()
        # Row format: Date, Container, pH, Nitrate, Pineapple_ml, Volume_ml, Environment, Target_OD
        ws.append_row([date_now, container, ph, nitrate, suggested_dose, vol, env, ""])
        
        return jsonify({
            "status": "success",
            "dose": suggested_dose,
            "prediction": predicted_od
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Health check endpoint for UptimeRobot
@app.route('/health')
def health():
    return "AI Brain is Active", 200

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
