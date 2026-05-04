import pandas as pd
from xgboost import XGBRegressor

def train_and_predict_pineapple(historical_data, current_ph, current_nitrate, volume_ml):
    # Convert list to DataFrame
    df = pd.DataFrame(historical_data[1:], columns=historical_data[0])
    
    # Ensure numeric types for AI columns
    cols = ['Avg_pH', 'Nitrate', 'Avg_OD', 'Volume(L)']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    train_df = df.dropna(subset=['Avg_pH', 'Avg_OD'])
    
    # Check if we have enough data to train
    if len(train_df) < 3:
        return 10.0, 0.450 # Standard startup dose

    # Features: pH, Nitrate, Volume | Target: OD
    X = train_df[['Avg_pH', 'Nitrate', 'Volume(L)']]
    y = train_df['Avg_OD']
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
    model.fit(X, y)
    
    input_data = pd.DataFrame([[current_ph, current_nitrate, volume_ml/1000]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    
    # Return dose (fixed for now or optimized) and predicted OD
    return 10.0, round(float(prediction), 3)
