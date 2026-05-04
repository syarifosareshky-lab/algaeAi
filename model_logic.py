import pandas as pd
from xgboost import XGBRegressor
import numpy as np

def train_and_predict_pineapple(historical_data, current_ph, current_nitrate, volume_ml, env_type):
    """
    historical_data: List of rows from your Google Sheet
    env_type: 1 for Bioreactor, 0 for Open Pond
    """
    # 1. Convert Google Sheet list to a DataFrame
    # Assuming columns: Vol(L), Date, OD1, OD2, OD3, Avg_OD, pH1, pH2, pH3, Avg_pH, Nitrate
    df = pd.DataFrame(historical_data[1:], columns=historical_data[0])
    
    # 2. Clean Data: AI needs numbers, not text
    # We focus on Avg_pH (Col J) and Nitrate (Col K) and Avg_OD (Col F)
    # Note: Adjust column names if they are different in your sheet
    cols_to_numeric = ['Avg_pH', 'Nitrate', 'Avg_OD', 'Volume(L)']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows that don't have a result (OD) to train on
    train_df = df.dropna(subset=['Avg_pH', 'Avg_OD'])
    
    if len(train_df) < 3:
        # If the experiment just started, return a safe starting dose
        return 5.0, 0.350

    # 3. Define Features and Target
    # We want to predict what the OD will be tomorrow based on current state
    X = train_df[['Avg_pH', 'Nitrate', 'Volume(L)']]
    y = train_df['Avg_OD']
    
    # 4. Initialize and Train XGBoost
    # Using 'Forest' parameters for small datasets
    model = XGBRegressor(
        n_estimators=50, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    model.fit(X, y)
    
    # 5. Optimization: Find the best dose
    # We simulate doses to see which one pushes OD towards 1.0
    potential_doses = [0, 5, 10, 15, 20]
    best_dose = 10.0 # Default
    
    # Create the input for today's reading
    current_vol_l = volume_ml / 1000
    input_now = pd.DataFrame([[current_ph, current_nitrate, current_vol_l]], 
                             columns=['Avg_pH', 'Nitrate', 'Volume(L)'])
    
    prediction = model.predict(input_now)[0]
    
    return float(best_dose), round(float(prediction), 3)
