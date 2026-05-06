import os
import json
from datetime import datetime

import gspread
from flask import Flask, render_template, request, jsonify
from google.oauth2.service_account import Credentials

from model_logic import (
    train_and_predict_next_od,
    estimate_harvest_time,
    recommend_kno3,
    recommend_pineapple,
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
    "PineappleDose",
    "KNO3Dose",
    "NextOD_Predicted",
    "HarvestTargetOD",
    "DaysToHarvest",
    "HarvestStatus",
    "HarvestReadiness(%)",
    "KNO3_Advice",
    "Pineapple_Advice",
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

    if current_header != STANDARD_HEADERS:
        raise ValueError(
            "Selected worksheet does not match the AI dashboard format. "
            "Please use a clean tab like PBR_1_AI, PBR_2_AI, OPEN_1_AI, etc."
        )


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
        culture_condition = request.form.get("culture_condition", "").strip()

        volume_l = float(request.form.get("volume_l"))
        sampling_volume_l = float(request.form.get("sampling_volume_l", 0))

        previous_od = float(request.form.get("previous_od"))
        nitrate = float(request.form.get("nitrate"))

        pineapple_dose = float(request.form.get("pineapple_dose", 0))
        kno3_dose = float(request.form.get("kno3_dose", 0))

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
                raise ValueError(f"OD reading {i} is missing.")

            ph_vals.append(float(ph_value))
            od_vals.append(float(od_value))

        avg_ph = round(sum(ph_vals) / 3, 2)
        avg_od = round(sum(od_vals) / 3, 4)

        spreadsheet = get_spreadsheet()
        worksheet = spreadsheet.worksheet(container)

        ensure_header(worksheet)

        historical_data = worksheet.get_all_values()

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
            nitrate=nitrate,
            pineapple_dose=pineapple_dose,
            kno3_dose=kno3_dose
        )

        harvest_info = estimate_harvest_time(
            current_od=avg_od,
            predicted_next_od=next_od_predicted,
            target_od=harvest_target_od
        )

        kno3_advice = recommend_kno3(
            nitrate=nitrate,
            volume_l=volume_l
        )

        pineapple_advice = recommend_pineapple(
            avg_ph=avg_ph,
            nitrate=nitrate,
            volume_l=volume_l,
            current_pineapple_dose=pineapple_dose
        )

        culture_health = classify_culture_health(
            avg_ph=avg_ph,
            nitrate=nitrate,
            avg_od=avg_od,
            predicted_next_od=next_od_predicted
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")

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
            pineapple_dose,
            kno3_dose,
            next_od_predicted,
            harvest_target_od,
            harvest_info["days_to_harvest"],
            harvest_info["harvest_status"],
            harvest_info["readiness_percent"],
            kno3_advice,
            pineapple_advice,
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
            "next_od_predicted": next_od_predicted,
            "kno3_advice": kno3_advice,
            "pineapple_advice": pineapple_advice,
            "culture_health": culture_health,
            "harvest_status": harvest_info["harvest_status"],
            "days_to_harvest": harvest_info["days_to_harvest"],
            "readiness_percent": harvest_info["readiness_percent"],
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
