import pandas as pd
from xgboost import XGBRegressor

def train_and_predict_pineapple(historical_data, current_ph, current_nitrate, volume_ml):
    df = pd.DataFrame(historical_data[1:], columns=historical_data[0])
    
    cols = ['Avg_pH', 'Nitrate', 'Avg_OD', 'Volume(L)']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    train_df = df.dropna(subset=['Avg_pH', 'Avg_OD'])
    
    if len(train_df) < 3:
        return 10.0, 0.450 # Default startup dose

    X = train_df[['Avg_pH', 'Nitrate', 'Volume(L)']]
    y = train_df['Avg_OD']
    
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
    model.fit(X, y)
    
    input_now = pd.DataFrame([[current_ph, current_nitrate, volume_ml/1000]], columns=X.columns)
    prediction = model.predict(input_now)[0]
    
    return 10.0, round(float(prediction), 3)
