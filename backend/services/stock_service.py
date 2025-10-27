"""
Stock Service Module
Handles Yahoo Finance data fetching and volume calculations
"""

import yfinance as yf
import pandas as pd
import dateutil.parser
from datetime import timedelta


def fetch_stock_data(symbol, start=None, end=None, period=None, interval='1d'):
    """Fetch stock data from Yahoo Finance"""
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
        return []

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data = data.rename(columns={'Datetime': 'Date'} if 'Datetime' in data.columns else {'Date': 'Date'})

    # Format dates
    if 'h' in interval or 'm' in interval:
        data['Date'] = data['Date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")

    return data[['Date', 'Open', 'High', 'Low', 'Close']].fillna("").to_dict(orient="records")


def calculate_volume_for_range(symbol, start_date, end_date, interval='1d'):
    """Calculate average volume for a date range"""
    try:
        start_dt = dateutil.parser.isoparse(start_date)
        end_dt = dateutil.parser.isoparse(end_date)
    except Exception as e:
        print("Date parsing error:", e)
        return 0

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_str = start_dt.date().strftime("%Y-%m-%d")
    end_str = (end_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

    if df.empty:
        return 0

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Volume"])
    avg_volume = df["Volume"].mean() if not df.empty else 0

    return avg_volume
