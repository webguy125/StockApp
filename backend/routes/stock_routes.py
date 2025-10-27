"""
Stock Routes
Handles stock data fetching, volume calculations, and indicators
"""

from flask import Blueprint, request, jsonify, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from services.stock_service import fetch_stock_data, calculate_volume_for_range
from services.indicator_service import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands, calculate_vwap
)

stock_bp = Blueprint('stock', __name__)


def init_stock_routes(frontend_dir):
    """Initialize routes with configuration"""
    @stock_bp.route("/")
    def serve_index():
        return send_from_directory(frontend_dir, "index_complete.html")

    @stock_bp.route("/classic")
    def serve_classic():
        return send_from_directory(frontend_dir, "index.html")

    @stock_bp.route("/enhanced")
    def serve_enhanced():
        return send_from_directory(frontend_dir, "index_enhanced.html")


@stock_bp.route("/data/<symbol>")
def get_chart_data(symbol):
    """Fetch stock chart data"""
    symbol = symbol.upper()
    start = request.args.get('start')
    end = request.args.get('end')
    period = request.args.get('period')
    interval = request.args.get('interval', '1d')

    data = fetch_stock_data(symbol, start, end, period, interval)
    return jsonify(data)


@stock_bp.route("/volume", methods=["POST"])
def calculate_volume():
    """Calculate average volume for a date range"""
    data = request.get_json()
    symbol = data["symbol"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    interval = data.get("interval", "1d")

    print(f"Volume request received: {symbol} from {start_date} to {end_date} interval {interval}")

    avg_volume = calculate_volume_for_range(symbol, start_date, end_date, interval)

    print(f"Avg volume for {symbol} between {start_date} and {end_date}: {avg_volume:.2f}")
    return jsonify({"avg_volume": avg_volume})


@stock_bp.route("/indicators", methods=["POST"])
def calculate_indicators():
    """Calculate technical indicators for a given symbol and date range"""
    data = request.get_json()
    symbol = data["symbol"]
    start = data.get("start")
    end = data.get("end")
    period = data.get("period")
    interval = data.get("interval", "1d")
    indicators = data.get("indicators", [])

    kwargs = {'interval': interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    elif period:
        kwargs['period'] = period
    else:
        kwargs['period'] = 'max'

    df = yf.download(symbol, **kwargs)

    if df.empty:
        return jsonify({"error": "No data found"}), 404

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date'})

    result = {}

    for indicator_config in indicators:
        ind_type = indicator_config.get("type")
        params = indicator_config.get("params", {})

        if ind_type == "SMA":
            period_val = params.get("period", 20)
            sma = calculate_sma(df['Close'], period_val)
            result[f"SMA_{period_val}"] = sma.replace({np.nan: None}).tolist()

        elif ind_type == "EMA":
            period_val = params.get("period", 20)
            ema = calculate_ema(df['Close'], period_val)
            result[f"EMA_{period_val}"] = ema.replace({np.nan: None}).tolist()

        elif ind_type == "RSI":
            period_val = params.get("period", 14)
            rsi = calculate_rsi(df['Close'], period_val)
            result["RSI"] = rsi.replace({np.nan: None}).tolist()

        elif ind_type == "MACD":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            macd_data = calculate_macd(df['Close'], fast, slow, signal)
            result["MACD"] = macd_data['macd'].replace({np.nan: None}).tolist()
            result["MACD_signal"] = macd_data['signal'].replace({np.nan: None}).tolist()
            result["MACD_histogram"] = macd_data['histogram'].replace({np.nan: None}).tolist()

        elif ind_type == "BB":
            period_val = params.get("period", 20)
            std_dev = params.get("std", 2)
            bb_data = calculate_bollinger_bands(df['Close'], period_val, std_dev)
            result["BB_upper"] = bb_data['upper'].replace({np.nan: None}).tolist()
            result["BB_middle"] = bb_data['middle'].replace({np.nan: None}).tolist()
            result["BB_lower"] = bb_data['lower'].replace({np.nan: None}).tolist()

        elif ind_type == "VWAP":
            vwap = calculate_vwap(df)
            result["VWAP"] = vwap.replace({np.nan: None}).tolist()

    # Add dates for alignment
    if 'h' in interval or 'm' in interval:
        result["dates"] = df['Date'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    else:
        result["dates"] = df['Date'].dt.strftime("%Y-%m-%d").tolist()

    return jsonify(result)
