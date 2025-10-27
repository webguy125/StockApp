from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
import dateutil.parser

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/data/<symbol>")
def get_chart_data(symbol):
    symbol = symbol.upper()
    start = request.args.get('start')
    end = request.args.get('end')
    period = request.args.get('period')
    interval = request.args.get('interval', '1d')

    kwargs = {'interval': interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    elif period:
        kwargs['period'] = period
    else:
        kwargs['period'] = 'max'

    data = yf.download(symbol, **kwargs)

    if data.empty:
        print(f"No data found for {symbol}.")
        return jsonify([])

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data = data.rename(columns={'Datetime': 'Date'} if 'Datetime' in data.columns else {'Date': 'Date'})
    data['Date'] = data['Date'].dt.strftime("%Y-%m-%d %H:%M:%S") if 'h' in interval or 'm' in interval else data['Date'].dt.strftime("%Y-%m-%d")
    return jsonify(data[['Date', 'Open', 'High', 'Low', 'Close']].fillna("").to_dict(orient="records"))

@app.route("/volume", methods=["POST"])
def calculate_volume():
    data = request.get_json()
    symbol = data["symbol"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    interval = data.get("interval", "1d")

    print(f"Volume request received: {symbol} from {start_date} to {end_date} interval {interval}")

    try:
        start_dt = dateutil.parser.isoparse(start_date)
        end_dt = dateutil.parser.isoparse(end_date)
    except Exception as e:
        print("Date parsing error:", e)
        return jsonify({"avg_volume": 0})

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_str = start_dt.date().strftime("%Y-%m-%d")
    end_str = (end_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

    if df.empty:
        print(f"No data found in range: {start_str} to {end_str}")
        return jsonify({"avg_volume": 0})

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Volume"])
    avg_volume = df["Volume"].mean() if not df.empty else 0

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

if __name__ == "__main__":
    app.run(debug=True)