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
    kno3_dose,
    air_flow_rate=None,
    co2_status="Not Recorded",
    co2_flow_rate=None,
    co2_duration_min=None,
    image_mean_r=None,
    image_mean_g=None,
    image_mean_b=None,
    image_brightness=None,
    image_green_index=None,
    image_excess_green=None,
    image_brown_index=None
):
    """
    Predict next-day OD750.

    The system uses:
    1. A biological fallback prediction when data are still limited.
    2. A Random Forest model when enough usable historical records exist.
    3. A prediction guardrail so an unstable early ML result does not create
       an unrealistic sudden OD jump/drop compared with the biological fallback.
    """

    fallback_prediction = fallback_next_od_prediction(
        current_od=avg_od,
        previous_od=previous_od,
        avg_ph=avg_ph,
        nitrate=nitrate,
        pineapple_dose=pineapple_dose,
        kno3_dose=kno3_dose,
        air_flow_rate=air_flow_rate,
        co2_status=co2_status,
        co2_flow_rate=co2_flow_rate
    )

    try:
        if not historical_data or len(historical_data) < 12:
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

        optional_numeric_cols = [
            "AirFlowRate_L_min",
            "CO2FlowRate_L_min",
            "CO2Duration_min",
            "ImageMeanR",
            "ImageMeanG",
            "ImageMeanB",
            "ImageBrightness",
            "ImageGreenIndex",
            "ImageExcessGreen",
            "ImageBrownIndex"
        ]

        optional_categorical_cols = ["CO2Status"]

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

        for col in optional_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        for col in optional_categorical_cols:
            if col not in df.columns:
                df[col] = "Not Recorded"
            df[col] = df[col].replace("", "Not Recorded").fillna("Not Recorded")

        categorical_cols = [
            "SystemType",
            "ContainerType",
            "CultureCondition",
            "CO2Status"
        ]

        df = df.dropna(subset=numeric_cols)

        if len(df) < 12:
            return fallback_prediction

        df = df.sort_values(by=["Day"]).reset_index(drop=True)

        # Today's data -> next recorded day's OD750
        df["TargetNextOD"] = df["AvgOD"].shift(-1)
        train_df = df.dropna(subset=["TargetNextOD"])

        if len(train_df) < 10:
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
            "KNO3Dose",
            "AirFlowRate_L_min",
            "CO2Status",
            "CO2FlowRate_L_min",
            "CO2Duration_min",
            "ImageMeanR",
            "ImageMeanG",
            "ImageMeanB",
            "ImageBrightness",
            "ImageGreenIndex",
            "ImageExcessGreen",
            "ImageBrownIndex"
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
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=2,
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
            "KNO3Dose": kno3_dose,
            "AirFlowRate_L_min": safe_float(air_flow_rate, 0),
            "CO2Status": co2_status or "Not Recorded",
            "CO2FlowRate_L_min": safe_float(co2_flow_rate, 0),
            "CO2Duration_min": safe_float(co2_duration_min, 0),
            "ImageMeanR": safe_float(image_mean_r, 0),
            "ImageMeanG": safe_float(image_mean_g, 0),
            "ImageMeanB": safe_float(image_mean_b, 0),
            "ImageBrightness": safe_float(image_brightness, 0),
            "ImageGreenIndex": safe_float(image_green_index, 0),
            "ImageExcessGreen": safe_float(image_excess_green, 0),
            "ImageBrownIndex": safe_float(image_brown_index, 0)
        }])

        ml_prediction = max(float(pipeline.predict(input_now)[0]), 0)

        prediction = guardrail_ml_prediction(
            ml_prediction=ml_prediction,
            fallback_prediction=fallback_prediction,
            current_od=avg_od,
            avg_ph=avg_ph,
            nitrate=nitrate,
            co2_status=co2_status
        )

        return round(prediction, 4)

    except Exception:
        return fallback_prediction


def guardrail_ml_prediction(
    ml_prediction,
    fallback_prediction,
    current_od,
    avg_ph,
    nitrate,
    co2_status="Not Recorded"
):
    """
    Prevent an early or unstable ML model from producing a confusing prediction.

    The model is still used, but extreme deviations from the biological fallback
    are blended back toward the fallback estimate.
    """

    ml_prediction = safe_float(ml_prediction, fallback_prediction)
    fallback_prediction = safe_float(fallback_prediction, current_od)
    current_od = safe_float(current_od, 0)
    avg_ph = safe_float(avg_ph, 7)
    nitrate = safe_float(nitrate, 0)
    co2_status = str(co2_status or "Not Recorded").upper()

    max_reasonable_gap = max(0.18, current_od * 0.25)
    gap = abs(ml_prediction - fallback_prediction)

    if gap > max_reasonable_gap:
        # Strong fallback weighting when ML is far from a biological estimate.
        blended = (0.75 * fallback_prediction) + (0.25 * ml_prediction)
    else:
        # Balanced blend when both estimates are reasonably close.
        blended = (0.55 * fallback_prediction) + (0.45 * ml_prediction)

    # Critical low pH should not show unrealistic strong growth.
    if avg_ph < 6.5:
        allowed_upper = current_od * 1.03
        blended = min(blended, allowed_upper)

        if co2_status in {"ON", "PULSE"}:
            # Do not predict aggressive positive growth while CO2 is running
            # under clearly acidic pH.
            blended = min(blended, current_od * 1.01)

    # Very low nitrate should not show unrealistic strong growth.
    if nitrate < 5:
        blended = min(blended, current_od * 1.03)

    return max(blended, 0)


def fallback_next_od_prediction(
    current_od,
    previous_od,
    avg_ph,
    nitrate,
    pineapple_dose,
    kno3_dose,
    air_flow_rate=None,
    co2_status="Not Recorded",
    co2_flow_rate=None
):
    """
    Rule-based next-day OD750 prediction used while the dataset is small.

    Improvements:
    - Preserves real negative OD trends instead of forcing them to positive growth.
    - Prevents runaway extrapolation from a single large OD jump.
    - Applies a stronger growth ceiling when pH is critically acidic.
    """

    current_od = safe_float(current_od, 0)
    previous_od = safe_float(previous_od, current_od)
    avg_ph = safe_float(avg_ph, 7)
    nitrate = safe_float(nitrate, 0)
    pineapple_dose = safe_float(pineapple_dose, 0)
    kno3_dose = safe_float(kno3_dose, 0)
    air_flow_rate = safe_float(air_flow_rate, 0)
    co2_flow_rate = safe_float(co2_flow_rate, 0)
    co2_status = str(co2_status or "Not Recorded").upper()

    if current_od <= 0:
        return 0.0

    if previous_od <= 0 or previous_od == current_od:
        observed_change = current_od * 0.03
    else:
        observed_change = current_od - previous_od

    # Avoid unrealistic extrapolation from a single unusual jump/drop.
    observed_change = max(
        min(observed_change, current_od * 0.25),
        -current_od * 0.20
    )

    if observed_change >= 0:
        growth_factor = 1.0

        # pH effect
        if avg_ph < 6.5:
            growth_factor *= 0.20
        elif avg_ph < 7.0:
            growth_factor *= 0.40
        elif avg_ph < 8.2:
            growth_factor *= 0.75
        elif 8.2 <= avg_ph <= 9.8:
            growth_factor *= 1.08
        elif 9.8 < avg_ph <= 10.2:
            growth_factor *= 0.95
        elif avg_ph > 10.2:
            growth_factor *= 0.75

        # nitrate effect
        if nitrate < 5:
            growth_factor *= 0.45
        elif nitrate < 10:
            growth_factor *= 0.65
        elif 20 <= nitrate <= 100:
            growth_factor *= 1.05
        elif nitrate > 150:
            growth_factor *= 0.90

        # pineapple effect
        if pineapple_dose > 0 and avg_ph >= 8.2:
            growth_factor *= 1.04
        elif pineapple_dose > 0 and avg_ph < 8.2:
            growth_factor *= 0.85

        # KNO3 effect
        if kno3_dose > 0 and nitrate < 70:
            growth_factor *= 1.04

        # air flow effect
        if air_flow_rate > 0:
            growth_factor *= 1.02

        # CO2 effect
        if co2_status in {"ON", "PULSE"} and avg_ph < 8.2:
            growth_factor *= 0.75
        elif co2_status in {"ON", "PULSE"} and avg_ph > 10.2:
            growth_factor *= 1.03
        elif co2_flow_rate > 0 and avg_ph < 8.2:
            growth_factor *= 0.75

        adjusted_change = observed_change * growth_factor

    else:
        # If OD already declined, use conditions to assess whether decline may
        # worsen or soften instead of magically forcing growth.
        decline_factor = 1.0

        if avg_ph < 7.0 or avg_ph > 10.5:
            decline_factor *= 1.15
        elif 8.2 <= avg_ph <= 9.8:
            decline_factor *= 0.85

        if nitrate < 10:
            decline_factor *= 1.10
        elif 20 <= nitrate <= 100:
            decline_factor *= 0.90

        if co2_status in {"ON", "PULSE"} and avg_ph < 8.2:
            decline_factor *= 1.10

        adjusted_change = observed_change * decline_factor

    # Severe acid stress: avoid showing a strong optimistic next-day increase.
    if avg_ph < 6.5:
        adjusted_change = min(adjusted_change, current_od * 0.02)

        if co2_status in {"ON", "PULSE"} or co2_flow_rate > 0:
            adjusted_change = min(adjusted_change, current_od * 0.01)

    predicted = current_od + adjusted_change

    return round(max(predicted, 0), 4)


def estimate_harvest_time(current_od, predicted_next_od, target_od=1.0):
    """
    Estimate harvest readiness based on:
    - Current OD750
    - Predicted next-day OD750
    - Target harvest OD750
    """

    current_od = safe_float(current_od, 0)
    predicted_next_od = safe_float(predicted_next_od, current_od)
    target_od = safe_float(target_od, 1.0)

    if target_od <= 0:
        target_od = 1.0

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

    nitrate = safe_float(nitrate, 0)
    volume_l = safe_float(volume_l, 10)

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

    avg_ph = safe_float(avg_ph, 7)
    nitrate = safe_float(nitrate, 0)
    volume_l = safe_float(volume_l, 10)
    current_pineapple_dose = safe_float(current_pineapple_dose, 0)

    base_dose = volume_l * 1.0

    if avg_ph < 8.2:
        return "STOP pineapple - pH too acidic"

    if avg_ph > 10.2:
        suggested = round(base_dose * 0.3, 2)
        if current_pineapple_dose >= suggested and suggested > 0:
            return "No extra pineapple - today’s dose is already sufficient"
        return f"Reduce pineapple. Suggested: {suggested} ml"

    if nitrate < 10:
        suggested = round(base_dose * 0.5, 2)
        if current_pineapple_dose >= suggested and suggested > 0:
            return "No extra pineapple - today’s dose is already sufficient"
        return f"Low nitrate. Suggested pineapple: {suggested} ml"

    if 8.2 <= avg_ph <= 9.5:
        suggested = round(base_dose, 2)
        if current_pineapple_dose >= suggested and suggested > 0:
            return "No extra pineapple - today’s dose is already sufficient"
        return f"Suggested pineapple: {suggested} ml"

    if 9.5 < avg_ph <= 10.2:
        suggested = round(base_dose * 0.7, 2)
        if current_pineapple_dose >= suggested and suggested > 0:
            return "No extra pineapple - today’s dose is already sufficient"
        return f"Suggested pineapple: {suggested} ml"

    return f"Suggested pineapple: {round(base_dose, 2)} ml"


def recommend_co2(avg_ph, current_co2_status="Not Recorded"):
    """
    CO2 control recommendation based on pH.

    Logic:
    - CO2 can lower pH through carbonic acid formation.
    - If pH is acidic, CO2 should be stopped.
    - If pH is too alkaline, controlled CO2 can help reduce pH.
    """

    avg_ph = safe_float(avg_ph, 7)
    current_co2_status = str(current_co2_status or "Not Recorded").upper()

    if avg_ph < 6.5:
        return "CO2 OFF - Critical acidic pH"

    if 6.5 <= avg_ph < 7.0:
        return "CO2 OFF - Acidic pH"

    if 7.0 <= avg_ph < 8.2:
        return "CO2 OFF - pH still low"

    if 8.2 <= avg_ph <= 10.2:
        if current_co2_status in {"ON", "PULSE"}:
            return "CO2 pulse only if needed - pH acceptable"
        return "CO2 normal / pulse only if needed"

    if 10.2 < avg_ph <= 10.8:
        return "CO2 ON - Controlled pulse to reduce high pH"

    return "CO2 ON - Very high pH, monitor closely"


def classify_culture_health(avg_ph, nitrate, avg_od, predicted_next_od):
    """
    Overall culture health classification.
    """

    avg_ph = safe_float(avg_ph, 7)
    nitrate = safe_float(nitrate, 0)
    avg_od = safe_float(avg_od, 0)
    predicted_next_od = safe_float(predicted_next_od, avg_od)

    if avg_ph < 6.5:
        return "CRITICAL - Strongly acidic culture"

    if 6.5 <= avg_ph < 7.0:
        return "CRITICAL - Acidic culture"

    if 7.0 <= avg_ph < 8.2:
        return "WARNING - Low pH, avoid pineapple and CO2"

    if avg_ph > 10.5:
        return "WARNING - Very high pH"

    if nitrate < 10:
        return "WARNING - Nitrate depleted"

    if predicted_next_od < avg_od:
        return "WARNING - OD750 may decline"

    return "Healthy / Growing"


def safe_float(value, default=0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default
