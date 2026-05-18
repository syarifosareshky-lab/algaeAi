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
METADATA_SHEET_NAME = "Container_Metadata"
METADATA_HEADERS = ["Container", "CultureStartDate"]

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
    If no previous data exists, use current AvgOD to avoid an artificial Day 1 jump.
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


def apply_harvest_override(harvest_status, avg_ph, co2_status):
    """
    When the culture already reaches the harvest target,
    dosing recommendations must not conflict with the harvest decision.
    CO2 advice is still shown because it is a culture safety/pH control note.
    """
    if harvest_status == "READY TO HARVEST":
        return {
            "kno3_advice": "No KNO3 addition - harvest target reached",
            "pineapple_advice": "No pineapple addition - harvest target reached",
            "co2_advice": recommend_co2(avg_ph, co2_status)
        }

    return None


def get_or_create_metadata_worksheet(spreadsheet):
    """Create/read a metadata sheet that stores Day 1 date for each container."""
    try:
        metadata_ws = spreadsheet.worksheet(METADATA_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        metadata_ws = spreadsheet.add_worksheet(
            title=METADATA_SHEET_NAME,
            rows=1000,
            cols=len(METADATA_HEADERS)
        )
        metadata_ws.append_row(METADATA_HEADERS)

    values = metadata_ws.get_all_values()

    if not values:
        metadata_ws.append_row(METADATA_HEADERS)
    elif values[0] != METADATA_HEADERS:
        metadata_ws.resize(rows=max(len(values), 1000), cols=len(METADATA_HEADERS))
        metadata_ws.update("A1", [METADATA_HEADERS])

    return metadata_ws


def get_container_start_date(spreadsheet, container):
    """Return saved Day 1 date for one container, or empty string if not set yet."""
    metadata_ws = get_or_create_metadata_worksheet(spreadsheet)
    values = metadata_ws.get_all_values()

    for row in values[1:]:
        if len(row) >= 2 and row[0] == container and row[1]:
            return row[1]

    return ""


def save_container_start_date(spreadsheet, container, culture_start_date):
    """Save or update the Day 1 date for a container."""
    metadata_ws = get_or_create_metadata_worksheet(spreadsheet)
    values = metadata_ws.get_all_values()

    for idx, row in enumerate(values[1:], start=2):
        if len(row) >= 1 and row[0] == container:
            metadata_ws.update(f"B{idx}", [[culture_start_date]])
            return

    metadata_ws.append_row([container, culture_start_date])


def calculate_culture_day(culture_start_date, data_collection_date):
    """
    Day 1 = first day the culture experiment started.
    The culture day follows real calendar dates, even when some days are skipped.
    """
    start_dt = datetime.strptime(culture_start_date, "%Y-%m-%d").date()
    collection_dt = datetime.strptime(data_collection_date, "%Y-%m-%d").date()

    if collection_dt < start_dt:
        raise ValueError("Data collection date cannot be earlier than the culture start date.")

    return (collection_dt - start_dt).days + 1


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
        culture_start_date = request.form.get("culture_start_date", "").strip()

        if not container_name:
            raise ValueError("Container name is required.")

        if not culture_start_date:
            raise ValueError("Culture start date / Day 1 date is required when creating a new container.")

        datetime.strptime(culture_start_date, "%Y-%m-%d")

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

        save_container_start_date(spreadsheet, container_name, culture_start_date)

        return jsonify({
            "status": "success",
            "message": f"{container_name} is ready. Day 1 date saved as {culture_start_date}.",
            "container": container_name,
            "culture_start_date": culture_start_date
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/container_meta/<container>", methods=["GET"])
def container_meta(container):
    try:
        spreadsheet = get_spreadsheet()
        culture_start_date = get_container_start_date(spreadsheet, container)

        return jsonify({
            "status": "success",
            "container": container,
            "culture_start_date": culture_start_date,
            "has_culture_start_date": bool(culture_start_date)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/set_culture_start_date", methods=["POST"])
def set_culture_start_date():
    try:
        container = request.form.get("container", "").strip()
        culture_start_date = request.form.get("culture_start_date", "").strip()

        if not container:
            raise ValueError("Please select a container.")

        if not culture_start_date:
            raise ValueError("Please enter the culture start date / Day 1 date.")

        datetime.strptime(culture_start_date, "%Y-%m-%d")

        spreadsheet = get_spreadsheet()
        spreadsheet.worksheet(container)
        save_container_start_date(spreadsheet, container, culture_start_date)

        return jsonify({
            "status": "success",
            "message": f"Day 1 date saved for {container}: {culture_start_date}.",
            "container": container,
            "culture_start_date": culture_start_date
        })

    except gspread.exceptions.WorksheetNotFound:
        return jsonify({
            "status": "error",
            "message": "Worksheet not found. Please select an existing container first."
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

        data_collection_date = request.form.get("data_collection_date", "").strip()
        culture_start_date_input = request.form.get("culture_start_date_existing", "").strip()

        if not data_collection_date:
            raise ValueError("Please enter the data collection date.")

        datetime.strptime(data_collection_date, "%Y-%m-%d")

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

        stored_culture_start_date = get_container_start_date(spreadsheet, container)

        # For older containers, the Day 1 date is entered once and then stored permanently.
        if not stored_culture_start_date:
            if not culture_start_date_input:
                raise ValueError(
                    "This container does not have a saved culture start date yet. "
                    "Please enter the Day 1 / culture start date once."
                )
            datetime.strptime(culture_start_date_input, "%Y-%m-%d")
            save_container_start_date(spreadsheet, container, culture_start_date_input)
            stored_culture_start_date = culture_start_date_input

        day = calculate_culture_day(stored_culture_start_date, data_collection_date)

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

        override_advice = apply_harvest_override(
            harvest_status=harvest_info["harvest_status"],
            avg_ph=avg_ph,
            co2_status=co2_status
        )

        if override_advice:
            kno3_advice = override_advice["kno3_advice"]
            pineapple_advice = override_advice["pineapple_advice"]
            co2_advice = override_advice["co2_advice"]

        else:
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

            co2_advice = recommend_co2(avg_ph, co2_status)

        culture_health = classify_culture_health(
            avg_ph=avg_ph,
            nitrate=nitrate_for_ai,
            avg_od=avg_od,
            predicted_next_od=next_od_predicted
        )

        if nitrate_status == "Not Measured":
            culture_health = culture_health + " | Nitrate not measured"

        malaysia_time = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))

        # Date = actual date the data/sample were taken.
        # Time = dashboard submission time.
        date_str = data_collection_date
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
            "culture_start_date": stored_culture_start_date,
            "data_collection_date": data_collection_date,
            "auto_day": day,
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
