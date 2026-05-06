import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO

import gspread
from flask import Flask, render_template, request, jsonify
from google.oauth2.service_account import Credentials
from PIL import Image

from model_logic import (
    train_and_predict_next_od,
    estimate_harvest_time,
    recommend_kno3,
    recommend_pineapple,
    recommend_co2,
    classify_culture_health
)

app = Flask(__name__)

SPREADSHEET_NAME = "Algae_Data_Master"

STANDARD_HEADERS = [
    "Container",
    "Date",
    "Time",
    "Day",
    "SystemType",
    "ContainerType",
    "CultureCondition",
    "Volume(L)",
    "SamplingVolume(L)",
    "PreviousOD",
    "OD1",
    "OD2",
    "OD3",
    "AvgOD",
    "pH1",
    "pH2",
    "pH3",
    "AvgPH",
    "Nitrate",
    "NitrateStatus",
    "PineappleDose",
    "KNO3Dose",
    "AirFlowRate_L_min",
    "CO2Status",
    "CO2FlowRate_L_min",
    "CO2Duration_min",
    "ImageUploaded",
    "ImageMeanR",
    "ImageMeanG",
    "ImageMeanB",
    "ImageBrightness",
    "ImageGreenIndex",
    "ImageExcessGreen",
    "ImageBrownIndex",
    "NextOD_Predicted",
    "HarvestTargetOD",
    "DaysToHarvest",
    "HarvestStatus",
    "HarvestReadiness(%)",
    "KNO3_Advice",
    "Pineapple_Advice",
    "CO2_Advice",
    "CultureHealth",
    "Biomass_g",
    "Harvested",
    "Notes"
]


def get_google_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds_json = os.environ.get("GOOGLE_CREDENTIALS")

    if not creds_json:
        raise ValueError("GOOGLE_CREDENTIALS not found in Render Environment Variables.")

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        raise ValueError("GOOGLE_CREDENTIALS is not valid JSON. Check your Render environment variable.")

    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


def get_spreadsheet():
    client = get_google_client()
    return client.open(SPREADSHEET_NAME)


def ensure_header(worksheet):
    values = worksheet.get_all_values()

    if not values:
        worksheet.append_row(STANDARD_HEADERS)
        return

    current_header = values[0]

    if current_header == STANDARD_HEADERS:
        return

    worksheet.resize(rows=max(len(values), 1000), cols=len(STANDARD_HEADERS))
    worksheet.update("A1", [STANDARD_HEADERS])


def optional_float(form_value):
    if form_value is None or str(form_value).strip() == "":
        return ""
    return float(form_value)


def safe_parse_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def get_previous_od_from_history(historical_data, current_avg_od):
    """
    Get PreviousOD automatically from the last saved AvgOD in the same container tab.
    If no previous data exists, use current AvgOD to avoid artificial Day 1 jump.
    """

    if not historical_data or len(historical_data) <= 1:
        return current_avg_od

    rows = historical_data[1:]

    for row in reversed(rows):
        if len(row) > 13:
            last_avg_od = safe_parse_float(row[13])
            if last_avg_od is not None:
                return round(last_avg_od, 4)

    return current_avg_od


def extract_image_features(image_file):
    if not image_file or image_file.filename == "":
        return {
            "ImageUploaded": "No",
            "ImageMeanR": "",
            "ImageMeanG": "",
            "ImageMeanB": "",
            "ImageBrightness": "",
            "ImageGreenIndex": "",
            "ImageExcessGreen": "",
            "ImageBrownIndex": ""
        }

    image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((300, 300))

    pixels = list(image.getdata())
    total_pixels = len(pixels)

    sum_r = sum(p[0] for p in pixels)
    sum_g = sum(p[1] for p in pixels)
    sum_b = sum(p[2] for p in pixels)

    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    brightness = (mean_r + mean_g + mean_b) / 3
    rgb_total = mean_r + mean_g + mean_b

    if rgb_total == 0:
        green_index = 0
        brown_index = 0
    else:
        green_index = mean_g / rgb_total
        brown_index = (mean_r + mean_g) / rgb_total

    excess_green = (2 * mean_g) - mean_r - mean_b

    return {
        "ImageUploaded": "Yes",
        "ImageMeanR": round(mean_r, 3),
        "ImageMeanG": round(mean_g, 3),
        "ImageMeanB": round(mean_b, 3),
        "ImageBrightness": round(brightness, 3),
        "ImageGreenIndex": round(green_index, 5),
        "ImageExcessGreen": round(excess_green, 3),
        "ImageBrownIndex": round(brown_index, 5)
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/containers", methods=["GET"])
def containers():
    try:
        spreadsheet = get_spreadsheet()
        worksheets = spreadsheet.worksheets()

        ai_tabs = [ws.title for ws in worksheets if ws.title.endswith("_AI")]
        names = ai_tabs if ai_tabs else [ws.title for ws in worksheets]

        return jsonify({
            "status": "success",
            "containers": names
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/create_container", methods=["POST"])
def create_container():
    try:
        container_name = request.form.get("container_name", "").strip()

        if not container_name:
            raise ValueError("Container name is required.")

        if not container_name.endswith("_AI"):
            container_name = container_name + "_AI"

        spreadsheet = get_spreadsheet()

        try:
            worksheet = spreadsheet.worksheet(container_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=container_name,
                rows=1000,
                cols=len(STANDARD_HEADERS)
            )
            worksheet.append_row(STANDARD_HEADERS)

        return jsonify({
            "status": "success",
            "message": f"{container_name} is ready.",
            "container": container_name
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/history/<container>", methods=["GET"])
def history(container):
    try:
        spreadsheet = get_spreadsheet()
        worksheet = spreadsheet.worksheet(container)

        ensure_header(worksheet)

        rows = worksheet.get_all_values()

        return jsonify({
            "status": "success",
            "history": rows
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/process", methods=["POST"])
def process():
    try:
        container = request.form.get("container", "").strip()

        if not container:
            raise ValueError("Please select a container.")

        day = int(float(request.form.get("day")))

        system_type = request.form.get("system_type", "").strip()
        container_type = request.form.get("container_type", "").strip()

        if container_type == "Other":
            other_liter = request.form.get("other_container_liter", "").strip()
            other_kind = request.form.get("other_container_kind", "").strip()
            other_kind_text = request.form.get("other_container_kind_text", "").strip()

            if not other_liter:
                raise ValueError("Please enter the custom container liter/capacity.")

            if not other_kind:
                raise ValueError("Please choose the custom container type.")

            if other_kind == "Other":
                if not other_kind_text:
                    raise ValueError("Please enter the custom container type name.")
                other_kind = other_kind_text

            container_type = f"{other_liter}L {other_kind}"

        culture_condition = request.form.get("culture_condition", "").strip()

        volume_l = float(request.form.get("volume_l"))
        sampling_volume_l = float(request.form.get("sampling_volume_l", 0))

        nitrate_raw = request.form.get("nitrate", "").strip()
        nitrate_status = request.form.get("nitrate_status", "Measured").strip()

        if nitrate_raw == "":
            nitrate = ""
            nitrate_status = "Not Measured"
            nitrate_for_ai = 80.0
        else:
            nitrate = float(nitrate_raw)
            nitrate_for_ai = nitrate
            nitrate_status = "Measured"

        pineapple_dose = float(request.form.get("pineapple_dose", 0))
        kno3_dose = float(request.form.get("kno3_dose", 0))

        air_flow_rate = optional_float(request.form.get("air_flow_rate"))
        co2_status = request.form.get("co2_status", "Not Recorded")
        co2_flow_rate = optional_float(request.form.get("co2_flow_rate"))
        co2_duration_min = optional_float(request.form.get("co2_duration_min"))

        harvest_target_od = float(request.form.get("harvest_target_od", 1.0))

        biomass_raw = request.form.get("biomass_g", "").strip()
        biomass_g = float(biomass_raw) if biomass_raw else ""

        harvested = request.form.get("harvested", "No")
        notes = request.form.get("notes", "").strip()

        ph_vals = []
        od_vals = []

        for i in range(1, 4):
            ph_value = request.form.get(f"ph{i}")
            od_value = request.form.get(f"od{i}")

            if ph_value is None or ph_value == "":
                raise ValueError(f"pH reading {i} is missing.")

            if od_value is None or od_value == "":
                raise ValueError(f"OD750 reading {i} is missing.")

            ph_vals.append(float(ph_value))
            od_vals.append(float(od_value))

        avg_ph = round(sum(ph_vals) / 3, 2)
        avg_od = round(sum(od_vals) / 3, 4)

        image_file = request.files.get("culture_image")
        image_features = extract_image_features(image_file)

        spreadsheet = get_spreadsheet()
        worksheet = spreadsheet.worksheet(container)

        ensure_header(worksheet)
        historical_data = worksheet.get_all_values()

        previous_od = get_previous_od_from_history(historical_data, avg_od)

        next_od_predicted = train_and_predict_next_od(
            historical_data=historical_data,
            day=day,
            system_type=system_type,
            container_type=container_type,
            culture_condition=culture_condition,
            volume_l=volume_l,
            sampling_volume_l=sampling_volume_l,
            previous_od=previous_od,
            avg_od=avg_od,
            avg_ph=avg_ph,
            nitrate=nitrate_for_ai,
            pineapple_dose=pineapple_dose,
            kno3_dose=kno3_dose,
            air_flow_rate=air_flow_rate,
            co2_status=co2_status,
            co2_flow_rate=co2_flow_rate,
            co2_duration_min=co2_duration_min,
            image_mean_r=image_features["ImageMeanR"],
            image_mean_g=image_features["ImageMeanG"],
            image_mean_b=image_features["ImageMeanB"],
            image_brightness=image_features["ImageBrightness"],
            image_green_index=image_features["ImageGreenIndex"],
            image_excess_green=image_features["ImageExcessGreen"],
            image_brown_index=image_features["ImageBrownIndex"]
        )

        harvest_info = estimate_harvest_time(
            current_od=avg_od,
            predicted_next_od=next_od_predicted,
            target_od=harvest_target_od
        )

        if nitrate_status == "Not Measured":
            kno3_advice = "Nitrate not measured - cannot calculate KNO3 advice"
            pineapple_advice = "Nitrate not measured - use pH and culture observation before dosing"
        else:
            kno3_advice = recommend_kno3(
                nitrate=nitrate_for_ai,
                volume_l=volume_l
            )

            pineapple_advice = recommend_pineapple(
                avg_ph=avg_ph,
                nitrate=nitrate_for_ai,
                volume_l=volume_l,
                current_pineapple_dose=pineapple_dose
            )

        co2_advice = recommend_co2(avg_ph)

        culture_health = classify_culture_health(
            avg_ph=avg_ph,
            nitrate=nitrate_for_ai,
            avg_od=avg_od,
            predicted_next_od=next_od_predicted
        )

        if nitrate_status == "Not Measured":
            culture_health = culture_health + " | Nitrate not measured"

        malaysia_time = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))

        date_str = malaysia_time.strftime("%Y-%m-%d")
        time_str = malaysia_time.strftime("%H:%M:%S")

        new_row = [
            container,
            date_str,
            time_str,
            day,
            system_type,
            container_type,
            culture_condition,
            volume_l,
            sampling_volume_l,
            previous_od,
            od_vals[0],
            od_vals[1],
            od_vals[2],
            avg_od,
            ph_vals[0],
            ph_vals[1],
            ph_vals[2],
            avg_ph,
            nitrate,
            nitrate_status,
            pineapple_dose,
            kno3_dose,
            air_flow_rate,
            co2_status,
            co2_flow_rate,
            co2_duration_min,
            image_features["ImageUploaded"],
            image_features["ImageMeanR"],
            image_features["ImageMeanG"],
            image_features["ImageMeanB"],
            image_features["ImageBrightness"],
            image_features["ImageGreenIndex"],
            image_features["ImageExcessGreen"],
            image_features["ImageBrownIndex"],
            next_od_predicted,
            harvest_target_od,
            harvest_info["days_to_harvest"],
            harvest_info["harvest_status"],
            harvest_info["readiness_percent"],
            kno3_advice,
            pineapple_advice,
            co2_advice,
            culture_health,
            biomass_g,
            harvested,
            notes
        ]

        worksheet.append_row(new_row)

        updated_history = worksheet.get_all_values()

        return jsonify({
            "status": "success",
            "container": container,
            "avg_ph": avg_ph,
            "avg_od": avg_od,
            "previous_od": previous_od,
            "nitrate_status": nitrate_status,
            "next_od_predicted": next_od_predicted,
            "kno3_advice": kno3_advice,
            "pineapple_advice": pineapple_advice,
            "co2_advice": co2_advice,
            "culture_health": culture_health,
            "harvest_status": harvest_info["harvest_status"],
            "days_to_harvest": harvest_info["days_to_harvest"],
            "readiness_percent": harvest_info["readiness_percent"],
            "image_uploaded": image_features["ImageUploaded"],
            "image_green_index": image_features["ImageGreenIndex"],
            "image_brightness": image_features["ImageBrightness"],
            "history": updated_history
        })

    except gspread.exceptions.WorksheetNotFound:
        return jsonify({
            "status": "error",
            "message": "Worksheet not found. Please create the container first."
        })

    except gspread.exceptions.SpreadsheetNotFound:
        return jsonify({
            "status": "error",
            "message": "Google Sheet not found. Make sure the sheet name is Algae_Data_Master and shared with your service account."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
