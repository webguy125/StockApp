# Example Plugin: Average True Range (ATR)
# Demonstrates volatility measurement plugin

from plugins.base_plugin import BasePlugin
import pandas as pd

class Plugin(BasePlugin):
    """
    Average True Range (ATR) Plugin

    Measures market volatility by calculating the average of true ranges.
    True Range is the greatest of:
    - Current High - Current Low
    - abs(Current High - Previous Close)
    - abs(Current Low - Previous Close)
    """

    name = "Average True Range"
    version = "1.0.0"
    description = "Volatility indicator measuring the average of true ranges"
    author = "StockApp Community"
    parameters = {
        "period": {
            "type": "int",
            "default": 14,
            "min": 1,
            "max": 100,
            "description": "Number of periods for ATR calculation"
        }
    }

    def calculate(self, df, params=None):
        """
        Calculate Average True Range

        Args:
            df: pandas DataFrame with OHLCV data (must have High, Low, Close)
            params: dict with 'period' parameter

        Returns:
            list of ATR values
        """
        if params is None:
            params = {}

        period = params.get('period', 14)

        # Validate required columns
        required_cols = ['High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")

        # Calculate True Range
        df_copy = df.copy()

        # Method 1: Current High - Current Low
        df_copy['HL'] = df_copy['High'] - df_copy['Low']

        # Method 2: abs(Current High - Previous Close)
        df_copy['HC'] = abs(df_copy['High'] - df_copy['Close'].shift(1))

        # Method 3: abs(Current Low - Previous Close)
        df_copy['LC'] = abs(df_copy['Low'] - df_copy['Close'].shift(1))

        # True Range is the maximum of the three
        df_copy['TR'] = df_copy[['HL', 'HC', 'LC']].max(axis=1)

        # Calculate ATR as the moving average of True Range
        atr = df_copy['TR'].rolling(window=period).mean()

        # Convert to list, replacing NaN with None
        atr_values = [None if pd.isna(val) else float(val) for val in atr]

        return atr_values

    def validate_params(self, params):
        """Validate input parameters"""
        if 'period' in params:
            period = params['period']
            if not isinstance(period, int) or period < 1 or period > 100:
                return False
        return True
