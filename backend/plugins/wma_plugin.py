# Example Plugin: Weighted Moving Average (WMA)
# This demonstrates how to create a custom indicator plugin

from plugins.base_plugin import BasePlugin

class Plugin(BasePlugin):
    """
    Weighted Moving Average Plugin

    Calculates WMA where more recent prices have higher weight.
    Formula: WMA = (P1*n + P2*(n-1) + ... + Pn*1) / (n + (n-1) + ... + 1)
    """

    name = "Weighted Moving Average"
    version = "1.0.0"
    description = "Calculates a weighted moving average where recent prices have more influence"
    author = "StockApp Team"
    parameters = {
        "period": {
            "type": "int",
            "default": 20,
            "min": 2,
            "max": 200,
            "description": "Number of periods for the WMA calculation"
        }
    }

    def calculate(self, df, params=None):
        """
        Calculate Weighted Moving Average

        Args:
            df: pandas DataFrame with OHLCV data
            params: dict with 'period' parameter

        Returns:
            list of WMA values
        """
        if params is None:
            params = {}

        period = params.get('period', 20)

        # Validate period
        if period < 2 or period > len(df):
            raise ValueError(f"Period must be between 2 and {len(df)}")

        # Calculate WMA
        wma_values = []

        for i in range(len(df)):
            if i < period - 1:
                wma_values.append(None)
            else:
                # Get the last 'period' closing prices
                prices = df['Close'].iloc[i - period + 1:i + 1].values

                # Create weights (most recent has highest weight)
                weights = list(range(1, period + 1))

                # Calculate weighted average
                wma = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
                wma_values.append(float(wma))

        return wma_values

    def validate_params(self, params):
        """Validate input parameters"""
        if 'period' in params:
            period = params['period']
            if not isinstance(period, int) or period < 2 or period > 200:
                return False
        return True
