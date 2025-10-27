"""
Analysis Routes
Handles pattern detection, volume profile, comparison, prediction, and trade ideas
"""

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from services.pattern_service import (
    detect_double_top, detect_double_bottom,
    detect_head_shoulders, calculate_volume_profile
)
from services.prediction_service import predict_price
from services.trade_service import generate_trade_ideas

analysis_bp = Blueprint('analysis', __name__)


@analysis_bp.route("/patterns", methods=["POST"])
def detect_patterns():
    """Detect chart patterns in price data"""
    data = request.get_json()
    symbol = data["symbol"]
    start = data.get("start")
    end = data.get("end")
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")

    kwargs = {'interval': interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    else:
        kwargs['period'] = period

    df = yf.download(symbol, **kwargs)

    if df.empty:
        return jsonify({"error": "No data found"}), 404

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date'})

    patterns = []

    # Detect different patterns
    patterns.extend(detect_double_top(df))
    patterns.extend(detect_double_bottom(df))
    patterns.extend(detect_head_shoulders(df))

    return jsonify({'patterns': patterns})


@analysis_bp.route("/volume_profile", methods=["POST"])
def volume_profile_route():
    """Calculate volume profile (volume by price level)"""
    data = request.get_json()
    symbol = data["symbol"]
    start = data.get("start")
    end = data.get("end")
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")
    bins = data.get("bins", 50)

    kwargs = {'interval': interval}
    if start and end:
        kwargs['start'] = start
        kwargs['end'] = end
    else:
        kwargs['period'] = period

    df = yf.download(symbol, **kwargs)

    if df.empty:
        return jsonify({"error": "No data found"}), 404

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    result = calculate_volume_profile(df, bins)

    return jsonify(result)


@analysis_bp.route("/compare", methods=["POST"])
def compare_symbols():
    """Compare multiple symbols on normalized scale"""
    data = request.get_json()
    symbols = data.get("symbols", [])
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")

    if not symbols or len(symbols) < 2:
        return jsonify({"error": "At least 2 symbols required"}), 400

    result = {}

    for symbol in symbols:
        df = yf.download(symbol, period=period, interval=interval)

        if df.empty:
            continue

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={date_col: 'Date'})

        # Normalize to percentage change from first close
        first_close = df['Close'].iloc[0]
        df['Normalized'] = ((df['Close'] - first_close) / first_close) * 100

        result[symbol] = {
            'dates': df['Date'].dt.strftime('%Y-%m-%d' if 'd' in interval or 'wk' in interval or 'mo' in interval else '%Y-%m-%d %H:%M:%S').tolist(),
            'normalized': df['Normalized'].tolist(),
            'close': df['Close'].tolist()
        }

    # Calculate correlation matrix
    if len(result) >= 2:
        correlations = {}
        symbols_list = list(result.keys())
        for i, sym1 in enumerate(symbols_list):
            for sym2 in symbols_list[i+1:]:
                # Get common dates
                data1 = result[sym1]['normalized']
                data2 = result[sym2]['normalized']
                min_len = min(len(data1), len(data2))

                if min_len > 1:
                    corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                    correlations[f"{sym1}_{sym2}"] = float(corr)

        result['correlations'] = correlations

    return jsonify(result)


@analysis_bp.route("/predict", methods=["POST"])
def predict_price_route():
    """ML-based price prediction using linear regression and trend analysis"""
    data = request.get_json()
    symbol = data["symbol"]
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")
    forecast_days = data.get("days", 30)

    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        return jsonify({"error": "No data found"}), 404

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date'})

    result = predict_price(df, forecast_days, interval)

    return jsonify(result)


@analysis_bp.route("/trade_ideas", methods=["POST"])
def generate_trade_ideas_route():
    """Auto-generate trade ideas based on technical analysis"""
    data = request.get_json()
    symbol = data["symbol"]
    period = data.get("period", "3mo")
    interval = data.get("interval", "1d")

    df = yf.download(symbol, period=period, interval=interval)

    if df.empty:
        return jsonify({"error": "No data found"}), 404

    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    current_price = df['Close'].iloc[-1]

    result = generate_trade_ideas(df, symbol, current_price)

    return jsonify(result)
