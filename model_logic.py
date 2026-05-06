import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_and_predict_next_od(
    historical_data,
    day,
    system_type,
    container_type,
    culture_condition,
    volume_l,
    sampling_volume_l,
    previous_od,
    avg_od,
    avg_ph,
    nitrate,
    pineapple_dose,
    kno3_dose
):
    """
    Predict next-day OD.

    AI target:
    Today data -> Tomorrow AvgOD

    If there is not enough data yet, fallback rule is used.
    """

    fallback_prediction = fallback_next_od_prediction(
        current_od=avg_od,
        previous_od=previous_od,
        avg_ph=avg_ph,
        nitrate=nitrate,
        pineapple_dose=pineapple_dose,
        kno3_dose=kno3_dose
    )

    try:
        if not historical_data or len(historical_data) < 8:
            return fallback_prediction

        df = pd.DataFrame(historical_data[1:], columns=historical_data[0])

        required_cols = [
            "Day",
            "SystemType",
            "ContainerType",
            "CultureCondition",
            "Volume(L)",
            "SamplingVolume(L)",
            "PreviousOD",
            "AvgOD",
            "AvgPH",
            "Nitrate",
            "PineappleDose",
            "KNO3Dose"
        ]

        for col in required_cols:
            if col not in df.columns:
                return fallback_prediction

        numeric_cols = [
            "Day",
            "Volume(L)",
            "SamplingVolume(L)",
            "PreviousOD",
            "AvgOD",
            "AvgPH",
            "Nitrate",
            "PineappleDose",
            "KNO3Dose"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        categorical_cols = [
            "SystemType",
            "ContainerType",
            "CultureCondition"
        ]

        df = df.dropna(subset=numeric_cols)

        if len(df) < 8:
            return fallback_prediction

        df = df.sort_values(by=["Day"]).reset_index(drop=True)

        df["TargetNextOD"] = df["AvgOD"].shift(-1)
        train_df = df.dropna(subset=["TargetNextOD"])

        if len(train_df) < 8:
            return fallback_prediction

        feature_cols = [
            "Day",
            "SystemType",
            "ContainerType",
            "CultureCondition",
            "Volume(L)",
            "SamplingVolume(L)",
            "PreviousOD",
            "AvgOD",
            "AvgPH",
            "Nitrate",
            "PineappleDose",
            "KNO3Dose"
        ]

        X = train_df[feature_cols]
        y = train_df["TargetNextOD"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ],
            remainder="passthrough"
        )

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X, y)

        input_now = pd.DataFrame([{
            "Day": day,
            "SystemType": system_type,
            "ContainerType": container_type,
            "CultureCondition": culture_condition,
            "Volume(L)": volume_l,
            "SamplingVolume(L)": sampling_volume_l,
            "PreviousOD": previous_od,
            "AvgOD": avg_od,
            "AvgPH": avg_ph,
            "Nitrate": nitrate,
            "PineappleDose": pineapple_dose,
            "KNO3Dose": kno3_dose
        }])

        prediction = pipeline.predict(input_now)[0]
        prediction = max(float(prediction), 0)

        return round(prediction, 4)

    except Exception:
        return fallback_prediction


def fallback_next_od_prediction(
    current_od,
    previous_od,
    avg_ph,
    nitrate,
    pineapple_dose,
    kno3_dose
):
    """
    Backup prediction for early data collection.
    This prevents dashboard failure when dataset is still small.
    """

    daily_change = current_od - previous_od

    if daily_change <= 0:
        daily_change = current_od * 0.05

    growth_factor = 1.0

    if avg_ph < 7.0:
        growth_factor *= 0.55
    elif avg_ph < 8.2:
        growth_factor *= 0.75
    elif 8.2 <= avg_ph <= 9.8:
        growth_factor *= 1.1
    elif avg_ph > 10.2:
        growth_factor *= 0.75

    if nitrate < 10:
        growth_factor *= 0.65
    elif 20 <= nitrate <= 100:
        growth_factor *= 1.05
    elif nitrate > 150:
        growth_factor *= 0.9

    if pineapple_dose > 0 and avg_ph >= 8.2:
        growth_factor *= 1.05

    if kno3_dose > 0 and nitrate < 70:
        growth_factor *= 1.05

    predicted = current_od + (daily_change * growth_factor)

    return round(max(predicted, 0), 4)


def estimate_harvest_time(current_od, predicted_next_od, target_od=1.0):
    """
    Estimate harvest time from predicted OD growth.
    """

    current_od = float(current_od)
    predicted_next_od = float(predicted_next_od)
    target_od = float(target_od)

    readiness = min((current_od / target_od) * 100, 100)
    readiness = round(readiness, 1)

    daily_increase = predicted_next_od - current_od

    if current_od >= target_od:
        return {
            "days_to_harvest": 0,
            "harvest_status": "READY TO HARVEST",
            "readiness_percent": readiness
        }

    if daily_increase <= 0:
        return {
            "days_to_harvest": "Unknown",
            "harvest_status": "NOT READY - Growth slowing or plateau",
            "readiness_percent": readiness
        }

    days = math.ceil((target_od - current_od) / daily_increase)

    if readiness < 70:
        status = "NOT READY"
    elif 70 <= readiness < 90:
        status = "GROWING"
    elif 90 <= readiness < 100:
        status = "ALMOST READY"
    else:
        status = "READY TO HARVEST"

    return {
        "days_to_harvest": days,
        "harvest_status": status,
        "readiness_percent": readiness
    }


def recommend_kno3(nitrate, volume_l):
    """
    KNO3 recommendation.

    Rule:
    For 10 L culture, 1 ml KNO3 stock increases about 6 ppm nitrate.
    """

    nitrate = float(nitrate)
    volume_l = float(volume_l)

    if nitrate < 70:
        kno3_ml = ((80 - nitrate) / 6) * (volume_l / 10)
        return f"ADD {round(kno3_ml, 2)} ml KNO3"

    if nitrate > 110:
        return "HIGH nitrate - Skip KNO3 dosing"

    return "Stable - No KNO3 needed"


def recommend_pineapple(avg_ph, nitrate, volume_l, current_pineapple_dose):
    """
    Pineapple recommendation.
    Base dose: 1 ml pineapple per 1 L culture.
    """

    avg_ph = float(avg_ph)
    nitrate = float(nitrate)
    volume_l = float(volume_l)

    base_dose = volume_l * 1.0

    if avg_ph < 8.2:
        return "STOP pineapple - pH too acidic"

    if avg_ph > 10.2:
        suggested = round(base_dose * 0.3, 2)
        return f"Reduce pineapple. Suggested: {suggested} ml"

    if nitrate < 10:
        suggested = round(base_dose * 0.5, 2)
        return f"Low nitrate. Suggested pineapple: {suggested} ml"

    if 8.2 <= avg_ph <= 9.5:
        suggested = round(base_dose, 2)
        return f"Suggested pineapple: {suggested} ml"

    if 9.5 < avg_ph <= 10.2:
        suggested = round(base_dose * 0.7, 2)
        return f"Suggested pineapple: {suggested} ml"

    return f"Suggested pineapple: {round(base_dose, 2)} ml"


def classify_culture_health(avg_ph, nitrate, avg_od, predicted_next_od):
    """
    Overall culture health classification.
    """

    avg_ph = float(avg_ph)
    nitrate = float(nitrate)
    avg_od = float(avg_od)
    predicted_next_od = float(predicted_next_od)

    if avg_ph < 7.0:
        return "CRITICAL - Acidic culture"

    if avg_ph < 8.2:
        return "WARNING - Low pH, avoid pineapple"

    if avg_ph > 10.5:
        return "WARNING - Very high pH"

    if nitrate < 10:
        return "WARNING - Nitrate depleted"

    if predicted_next_od < avg_od:
        return "WARNING - OD may decline"

    return "Healthy / Growing"
