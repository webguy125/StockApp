from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/data/<symbol>")
def get_chart_data(symbol):
    symbol = symbol.upper()
    filename = os.path.join(DATA_DIR, f"{symbol}.csv")

    if not os.path.exists(filename):
        print(f"CSV not found for {symbol}. Fetching now...")
        fetch_and_update_csv(symbol)

    if not os.path.exists(filename):
        print(f"Failed to fetch data for {symbol}.")
        return jsonify([])

    df = pd.read_csv(filename, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return jsonify(df.fillna("").to_dict(orient="records"))

@app.route("/volume", methods=["POST"])
def calculate_volume():
    data = request.get_json()
    symbol = data["symbol"]
    start_date = data["start_date"]
    end_date = data["end_date"]

    print(f"Volume request received: {symbol} from {start_date} to {end_date}")

    filename = os.path.join(DATA_DIR, f"{symbol.upper()}.csv")
    if not os.path.exists(filename):
        print("CSV not found.")
        return jsonify({"error": "CSV not found"}), 404

    df = pd.read_csv(filename, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    try:
        start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
    except Exception as e:
        print("Date parsing error:", e)
        return jsonify({"avg_volume": 0})

    df_range = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]

    if df_range.empty:
        print(f"No candles found in range: {start_dt} to {end_dt}")
        return jsonify({"avg_volume": 0})

    df_range = df_range.copy()
    df_range["Volume"] = pd.to_numeric(df_range["Volume"], errors="coerce")
    df_range = df_range.dropna(subset=["Volume"])
    avg_volume = df_range["Volume"].mean()

    print(f"Avg volume for {symbol} between {start_date} and {end_date}: {avg_volume:.2f}")
    return jsonify({"avg_volume": avg_volume})

@app.route("/save_line", methods=["POST"])
def save_line():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    line = data.get("line")

    if not symbol or not line or "id" not in line:
        print("Missing symbol or line ID")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    lines = []

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = json.load(f)
        except Exception as e:
            print("Error reading line file:", e)

    lines = [l for l in lines if l["id"] != line["id"]]
    lines.append(line)

    try:
        with open(filename, "w") as f:
            json.dump(lines, f)
        print(f"Saved line for {symbol}: {line['id']}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving line:", e)
        return jsonify({"error": "Failed to save line"}), 500

@app.route("/delete_line", methods=["POST"])
def delete_line():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    line_id = data.get("line_id")

    if not symbol or not line_id:
        print("Missing symbol or line ID for deletion")
        return jsonify({"error": "Missing symbol or line ID"}), 400

    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")
    if not os.path.exists(filename):
        print(f"No line file found for {symbol}")
        return jsonify({"success": True})

    try:
        with open(filename, "r") as f:
            lines = json.load(f)
        lines = [l for l in lines if l["id"] != line_id]
        with open(filename, "w") as f:
            json.dump(lines, f)
        print(f"Deleted line {line_id} for {symbol}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error deleting line:", e)
        return jsonify({"error": "Failed to delete line"}), 500

@app.route("/clear_lines", methods=["POST"])
def clear_lines():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")

    try:
        with open(filename, "w") as f:
            json.dump([], f)
        print(f"Cleared all lines for {symbol}")
        return jsonify({"success": True})
    except Exception as e:
        print("Error clearing lines:", e)
        return jsonify({"error": "Failed to clear lines"}), 500

@app.route("/lines/<symbol>")
def get_lines(symbol):
    symbol = symbol.upper()
    filename = os.path.join(DATA_DIR, f"lines_{symbol}.json")

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                lines = json.load(f)
            print(f"Loaded {len(lines)} lines for {symbol}")
            return jsonify(lines)
        except Exception as e:
            print("Error loading lines:", e)
            return jsonify([])
    else:
        print(f"No line file found for {symbol}")
        return jsonify([])

def fetch_and_update_csv(symbol):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"{symbol}.csv")

    start_date = (datetime.today() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start_date)

    if not data.empty:
        data.reset_index(inplace=True)
        data.to_csv(filename, index=False, float_format="%.2f")
        print(f"Created new file with {len(data)} rows: {filename}")
    else:
        print(f"No data found for {symbol}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        symbol = sys.argv[1].strip().upper()
        fetch_and_update_csv(symbol)
    else:
        print("Starting Flask server...")
        app.run(debug=True)