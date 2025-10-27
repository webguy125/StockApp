from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
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
        print("No candles found in range.")
        return jsonify({"avg_volume": 0})

    # ✅ Convert Volume to numeric before averaging
    df_range["Volume"] = pd.to_numeric(df_range["Volume"], errors="coerce")
    avg_volume = df_range["Volume"].mean()

    print(f"Avg volume for {symbol} between {start_date} and {end_date}: {avg_volume:.2f}")
    return jsonify({"avg_volume": avg_volume})

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