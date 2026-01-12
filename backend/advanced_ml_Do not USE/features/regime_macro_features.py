"""
Market Regime & Macro Economic Features
Adds 25 new features for market regime detection and macro indicators

NEW Features:
- Market Regime Detection: 10 features
- Macro Economic Indicators: 15 features

Total: 25 additional features
Combined with existing 179 → 204 total features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any
from datetime import datetime, timedelta


class RegimeMacroFeatures:
    """
    Detects market regime and adds macro economic indicators
    Designed to improve model performance in different market conditions
    """

    def __init__(self):
        self.version = "1.0.0"
        self._cache = {}  # Cache for market data to avoid repeated API calls

    def get_features(self, date: datetime, symbol: str = None) -> Dict[str, Any]:
        """
        Get all regime and macro features for a given date

        Args:
            date: The date to get features for
            symbol: Optional symbol (for future symbol-specific regime features)

        Returns:
            Dictionary with 25 regime + macro features
        """
        features = {}

        # Convert pandas Timestamp to datetime if needed
        if hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()

        # Ensure timezone-naive datetime
        if hasattr(date, 'replace') and hasattr(date, 'tzinfo') and date.tzinfo is not None:
            date = date.replace(tzinfo=None)

        # Get market regime features
        regime_features = self._get_market_regime(date)
        features.update(regime_features)

        # Get macro indicators
        macro_features = self._get_macro_indicators(date)
        features.update(macro_features)

        return features

    def _get_spy_data(self, date: datetime, days_back: int = 250) -> pd.DataFrame:
        """Get SPY data with caching"""
        cache_key = f"SPY_{date.date()}_{days_back}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            start_date = date - timedelta(days=days_back)
            end_date = date + timedelta(days=1)

            spy = yf.Ticker("SPY")
            spy_data = spy.history(start=start_date, end=end_date)

            if len(spy_data) > 0:
                self._cache[cache_key] = spy_data
                return spy_data
        except Exception as e:
            print(f"[WARNING] Could not fetch SPY data: {e}")

        return pd.DataFrame()

    def _get_market_regime(self, date: datetime) -> Dict[str, Any]:
        """
        Detect current market regime (10 features)

        Returns features indicating:
        - Trend regime (bull/bear/choppy)
        - Volatility regime (low/normal/high)
        - Momentum strength
        - Regime stability
        """
        features = {}

        spy_data = self._get_spy_data(date, days_back=250)

        if len(spy_data) < 200:
            # Return default values if insufficient data
            return self._default_regime_features()

        try:
            # Calculate key indicators
            current_price = spy_data['Close'].iloc[-1]
            ma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
            ma200 = spy_data['Close'].rolling(200).mean().iloc[-1]

            # Get VIX for volatility
            vix = self._get_vix(date)

            # Calculate returns for momentum
            returns_1m = (current_price / spy_data['Close'].iloc[-20] - 1) if len(spy_data) >= 20 else 0
            returns_3m = (current_price / spy_data['Close'].iloc[-60] - 1) if len(spy_data) >= 60 else 0
            returns_6m = (current_price / spy_data['Close'].iloc[-120] - 1) if len(spy_data) >= 120 else 0

            # Feature 1-3: Trend Regime (categorical)
            is_bull = 1 if (current_price > ma200 and ma50 > ma200) else 0
            is_bear = 1 if (current_price < ma200 and ma50 < ma200) else 0
            is_choppy = 1 if abs(current_price - ma200) / ma200 < 0.02 else 0

            features['is_bull_market'] = is_bull
            features['is_bear_market'] = is_bear
            features['is_choppy_market'] = is_choppy

            # Feature 4-6: Volatility Regime (categorical)
            features['is_low_vol'] = 1 if vix < 15 else 0
            features['is_normal_vol'] = 1 if 15 <= vix <= 25 else 0
            features['is_high_vol'] = 1 if vix > 25 else 0

            # Feature 7-9: Momentum Regime (continuous)
            features['market_momentum_1m'] = returns_1m
            features['market_momentum_3m'] = returns_3m
            features['market_momentum_6m'] = returns_6m

            # Feature 10: Trend Strength (continuous)
            features['trend_strength'] = (ma50 - ma200) / ma200 if not pd.isna(ma200) and ma200 != 0 else 0

        except Exception as e:
            print(f"[WARNING] Error calculating regime features: {e}")
            return self._default_regime_features()

        return features

    def _get_macro_indicators(self, date: datetime) -> Dict[str, Any]:
        """
        Get macro economic indicators (15 features)

        Returns features for:
        - VIX (volatility/fear)
        - Treasury yields (interest rates)
        - Dollar index (currency strength)
        - Sector rotation (risk on/off)
        """
        features = {}

        try:
            # Feature 1-3: VIX (Fear Index)
            vix_value = self._get_vix(date)
            vix_ma20 = self._get_vix_ma(date, 20)

            features['vix'] = vix_value
            features['vix_ma20'] = vix_ma20
            features['vix_change'] = vix_value - vix_ma20

            # Feature 4-6: Interest Rates
            yield_10y = self._get_treasury_yield(date, "^TNX")  # 10-year
            yield_2y = self._get_treasury_yield(date, "^IRX")   # 2-year (using 13-week as proxy)

            features['yield_10y'] = yield_10y
            features['yield_2y'] = yield_2y
            features['yield_spread'] = yield_10y - yield_2y  # Inverted curve = recession signal

            # Feature 7-9: Dollar & Commodities
            dollar_index = self._get_ticker_value(date, "DX-Y.NYB")  # Dollar Index
            gold_price = self._get_ticker_value(date, "GC=F")        # Gold
            oil_price = self._get_ticker_value(date, "CL=F")         # Oil

            features['dollar_index'] = dollar_index
            features['gold_price'] = gold_price
            features['oil_price'] = oil_price

            # Feature 10-12: Sector Rotation (Risk On/Off)
            xlk_perf = self._get_sector_performance(date, "XLK")  # Technology (risk-on)
            xlu_perf = self._get_sector_performance(date, "XLU")  # Utilities (risk-off)

            features['tech_performance'] = xlk_perf
            features['utilities_performance'] = xlu_perf
            features['risk_on_indicator'] = 1 if xlk_perf > xlu_perf else 0

            # Feature 13-15: Credit Spreads & Market Breadth
            hyg_tlt_ratio = self._get_credit_spread_proxy(date)

            features['credit_spread_proxy'] = hyg_tlt_ratio
            features['is_inverted_yield_curve'] = 1 if features['yield_spread'] < 0 else 0

        except Exception as e:
            print(f"[WARNING] Error calculating macro features: {e}")
            return self._default_macro_features()

        return features

    def _get_vix(self, date: datetime) -> float:
        """Get VIX value for given date"""
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            if len(vix_data) > 0:
                return vix_data['Close'].iloc[-1]
        except:
            pass
        return 20.0  # Default value

    def _get_vix_ma(self, date: datetime, period: int) -> float:
        """Get VIX moving average"""
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=date - timedelta(days=period + 10), end=date + timedelta(days=1))
            if len(vix_data) >= period:
                return vix_data['Close'].rolling(period).mean().iloc[-1]
        except:
            pass
        return 20.0  # Default value

    def _get_treasury_yield(self, date: datetime, ticker: str) -> float:
        """Get treasury yield for given ticker"""
        try:
            treasury = yf.Ticker(ticker)
            data = treasury.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            if len(data) > 0:
                return data['Close'].iloc[-1] / 100  # Convert to decimal (e.g., 4.5% → 0.045)
        except:
            pass
        return 0.04  # Default 4%

    def _get_ticker_value(self, date: datetime, ticker: str) -> float:
        """Get ticker price for given date"""
        try:
            instrument = yf.Ticker(ticker)
            data = instrument.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            if len(data) > 0:
                return data['Close'].iloc[-1]
        except:
            pass
        return 100.0  # Default value

    def _get_sector_performance(self, date: datetime, ticker: str, period: int = 20) -> float:
        """Get sector performance over period"""
        try:
            sector_etf = yf.Ticker(ticker)
            data = sector_etf.history(start=date - timedelta(days=period + 10), end=date + timedelta(days=1))
            if len(data) >= period:
                return (data['Close'].iloc[-1] / data['Close'].iloc[-period] - 1) * 100  # Return as percentage
        except:
            pass
        return 0.0  # Default

    def _get_credit_spread_proxy(self, date: datetime) -> float:
        """Get credit spread proxy (HYG/TLT ratio)"""
        try:
            hyg = yf.Ticker("HYG")  # High yield bonds
            tlt = yf.Ticker("TLT")  # Treasury bonds

            hyg_data = hyg.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            tlt_data = tlt.history(start=date - timedelta(days=5), end=date + timedelta(days=1))

            if len(hyg_data) > 0 and len(tlt_data) > 0:
                return hyg_data['Close'].iloc[-1] / tlt_data['Close'].iloc[-1]
        except:
            pass
        return 1.0  # Default

    def _default_regime_features(self) -> Dict[str, Any]:
        """Return default regime features when data unavailable"""
        return {
            'is_bull_market': 0,
            'is_bear_market': 0,
            'is_choppy_market': 1,  # Default to choppy (safest)
            'is_low_vol': 0,
            'is_normal_vol': 1,  # Default to normal vol
            'is_high_vol': 0,
            'market_momentum_1m': 0.0,
            'market_momentum_3m': 0.0,
            'market_momentum_6m': 0.0,
            'trend_strength': 0.0
        }

    def _default_macro_features(self) -> Dict[str, Any]:
        """Return default macro features when data unavailable"""
        return {
            'vix': 20.0,
            'vix_ma20': 20.0,
            'vix_change': 0.0,
            'yield_10y': 0.04,
            'yield_2y': 0.035,
            'yield_spread': 0.005,
            'dollar_index': 100.0,
            'gold_price': 2000.0,
            'oil_price': 75.0,
            'tech_performance': 0.0,
            'utilities_performance': 0.0,
            'risk_on_indicator': 1,
            'credit_spread_proxy': 1.0,
            'is_inverted_yield_curve': 0
        }


# Convenience function
def get_regime_macro_features(date: datetime, symbol: str = None) -> Dict[str, Any]:
    """
    Convenience function to get all regime + macro features

    Args:
        date: Date to get features for
        symbol: Optional symbol

    Returns:
        Dictionary with 25 features
    """
    extractor = RegimeMacroFeatures()
    return extractor.get_features(date, symbol)
