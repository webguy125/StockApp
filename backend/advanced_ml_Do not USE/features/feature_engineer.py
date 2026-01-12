"""
Advanced Feature Engineering Pipeline
Extracts 300+ technical features from OHLCV data

Categories:
- Momentum indicators (20+ features)
- Trend indicators (25+ features)
- Volume indicators (20+ features)
- Volatility indicators (15+ features)
- Price patterns (25+ features)
- Statistical features (20+ features)
- Market structure (15+ features)
- Multi-timeframe (20+ features)
- Contextual features (6 features)
- Derived features (30+ features)
- Event features (23 features) - NEW!

Total: ~202 features (179 technical + 23 event)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from advanced_ml.config.core_symbols import get_symbol_metadata, SECTOR_CODES
except ImportError:
    # Fallback if module not found
    def get_symbol_metadata(symbol):
        return {'sector': 'unknown', 'market_cap_category': 'unknown', 'sector_code': -1}
    SECTOR_CODES = {}

try:
    from advanced_ml.events import EventQuantHybrid
except ImportError:
    EventQuantHybrid = None


class FeatureEngineer:
    """
    Extracts 202 features from OHLCV price data + events
    Designed for ensemble ML models (Random Forest, XGBoost, LSTM, etc.)
    """

    def __init__(self, enable_events: bool = False):
        self.version = "2.0.0"
        self.enable_events = enable_events

        # Initialize event module if available and enabled
        if enable_events and EventQuantHybrid is not None:
            try:
                self.event_hybrid = EventQuantHybrid()
                print("[FEATURE_ENGINEER] Event features enabled (23 features)")
            except Exception as e:
                print(f"[FEATURE_ENGINEER] Warning: Could not initialize events: {e}")
                self.event_hybrid = None
        else:
            self.event_hybrid = None
            if not enable_events:
                print("[FEATURE_ENGINEER] Event features DISABLED (using 179 technical features only)")

    def extract_features(self, df: pd.DataFrame, symbol: str = None, target_date: datetime = None) -> Dict[str, Any]:
        """
        Main entry point - extracts all 202 features

        Args:
            df: DataFrame with OHLC columns (open, high, low, close, volume)
            symbol: Optional symbol name for sector correlation features
            target_date: Optional date for event feature extraction (defaults to last date in df)

        Returns:
            Dictionary with 202 features (179 technical + 23 event)
        """
        if df is None or len(df) < 50:
            return self._empty_features()

        # Ensure DataFrame has required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return self._empty_features()

        features = {}

        # Category 1: Momentum Indicators (20+ features)
        features.update(self._momentum_features(df))

        # Category 2: Trend Indicators (25+ features)
        features.update(self._trend_features(df))

        # Category 3: Volume Indicators (20+ features)
        features.update(self._volume_features(df))

        # Category 4: Volatility Indicators (15+ features)
        features.update(self._volatility_features(df))

        # Category 5: Price Pattern Features (25+ features)
        features.update(self._price_pattern_features(df))

        # Category 6: Statistical Features (20+ features)
        features.update(self._statistical_features(df))

        # Category 7: Market Structure (15+ features)
        features.update(self._market_structure_features(df))

        # Category 8: Multi-timeframe Features (20+ features)
        features.update(self._multi_timeframe_features(df))

        # Category 9: Contextual Features (6 features)
        if symbol:
            features.update(self._contextual_features(df, symbol))

        # Category 10: Event Features (23 features) - NEW!
        if symbol and self.event_hybrid is not None:
            features.update(self._event_features(df, symbol, target_date))

        # Category 11: Derived/Interaction Features (30+ features)
        features.update(self._derived_features(df, features))

        # Add metadata
        features['feature_count'] = len(features)
        features['symbol'] = symbol
        features['last_price'] = float(df['close'].iloc[-1])
        features['last_volume'] = float(df['volume'].iloc[-1])
        features['timestamp'] = datetime.now().isoformat()

        return features

    # ==========================================
    # CATEGORY 1: MOMENTUM INDICATORS (20+)
    # ==========================================

    def _momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Momentum-based technical indicators"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI (multiple periods)
        for period in [7, 14, 21, 28]:
            rsi = self._calculate_rsi(close, period)
            features[f'rsi_{period}'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0

        # Stochastic Oscillator (14, 3, 3)
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14, 3)
        features['stochastic_k'] = float(stoch_k.iloc[-1]) if not stoch_k.empty else 50.0
        features['stochastic_d'] = float(stoch_d.iloc[-1]) if not stoch_d.empty else 50.0

        # Rate of Change (multiple periods)
        for period in [5, 10, 20]:
            roc = ((close - close.shift(period)) / close.shift(period)) * 100
            features[f'roc_{period}'] = float(roc.iloc[-1]) if not roc.empty else 0.0

        # Williams %R
        willr = self._calculate_williams_r(high, low, close, 14)
        features['williams_r'] = float(willr.iloc[-1]) if not willr.empty else -50.0

        # Money Flow Index (MFI)
        mfi = self._calculate_mfi(high, low, close, df['volume'], 14)
        features['mfi_14'] = float(mfi.iloc[-1]) if not mfi.empty else 50.0

        # CCI (Commodity Channel Index)
        cci = self._calculate_cci(high, low, close, 20)
        features['cci_20'] = float(cci.iloc[-1]) if not cci.empty else 0.0

        # Ultimate Oscillator
        uo = self._calculate_ultimate_oscillator(high, low, close)
        features['ultimate_oscillator'] = float(uo.iloc[-1]) if not uo.empty else 50.0

        # Momentum (simple)
        features['momentum_10'] = float(close.iloc[-1] - close.iloc[-10]) if len(close) >= 10 else 0.0
        features['momentum_20'] = float(close.iloc[-1] - close.iloc[-20]) if len(close) >= 20 else 0.0

        return features

    # ==========================================
    # CATEGORY 2: TREND INDICATORS (25+)
    # ==========================================

    def _trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Trend-following indicators"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = close.rolling(window=period).mean()
            features[f'sma_{period}'] = float(sma.iloc[-1]) if not sma.empty else float(close.iloc[-1])
            # Distance from SMA
            features[f'price_vs_sma_{period}'] = float((close.iloc[-1] - sma.iloc[-1]) / sma.iloc[-1] * 100) if not sma.empty else 0.0

        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f'ema_{period}'] = float(ema.iloc[-1]) if not ema.empty else float(close.iloc[-1])

        # MACD (12, 26, 9)
        macd, signal, histogram = self._calculate_macd(close, 12, 26, 9)
        features['macd'] = float(macd.iloc[-1]) if not macd.empty else 0.0
        features['macd_signal'] = float(signal.iloc[-1]) if not signal.empty else 0.0
        features['macd_histogram'] = float(histogram.iloc[-1]) if not histogram.empty else 0.0

        # ADX (Average Directional Index)
        adx, plus_di, minus_di = self._calculate_adx(high, low, close, 14)
        features['adx_14'] = float(adx.iloc[-1]) if not adx.empty else 25.0
        features['plus_di'] = float(plus_di.iloc[-1]) if not plus_di.empty else 50.0
        features['minus_di'] = float(minus_di.iloc[-1]) if not minus_di.empty else 50.0

        # Parabolic SAR
        psar = self._calculate_parabolic_sar(high, low, close)
        features['parabolic_sar'] = float(psar.iloc[-1]) if not psar.empty else float(close.iloc[-1])
        features['price_vs_psar'] = float((close.iloc[-1] - psar.iloc[-1]) / close.iloc[-1] * 100) if not psar.empty else 0.0

        # Supertrend
        supertrend = self._calculate_supertrend(high, low, close, period=10, multiplier=3)
        features['supertrend'] = float(supertrend.iloc[-1]) if not supertrend.empty else float(close.iloc[-1])

        return features

    # ==========================================
    # CATEGORY 3: VOLUME INDICATORS (20+)
    # ==========================================

    def _volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Volume-based indicators"""
        features = {}
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']

        # On-Balance Volume (OBV)
        obv = self._calculate_obv(close, volume)
        features['obv'] = float(obv.iloc[-1]) if not obv.empty else 0.0
        features['obv_sma_20'] = float(obv.rolling(20).mean().iloc[-1]) if len(obv) >= 20 else 0.0

        # Accumulation/Distribution Line
        ad_line = self._calculate_ad_line(high, low, close, volume)
        features['ad_line'] = float(ad_line.iloc[-1]) if not ad_line.empty else 0.0

        # Chaikin Money Flow
        cmf = self._calculate_cmf(high, low, close, volume, 20)
        features['cmf_20'] = float(cmf.iloc[-1]) if not cmf.empty else 0.0

        # VWAP (Volume Weighted Average Price)
        vwap = self._calculate_vwap(high, low, close, volume)
        features['vwap'] = float(vwap.iloc[-1]) if not vwap.empty else float(close.iloc[-1])
        features['price_vs_vwap'] = float((close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100) if not vwap.empty else 0.0

        # Volume ratios
        avg_volume_20 = volume.rolling(20).mean()
        features['volume_ratio_20'] = float(volume.iloc[-1] / avg_volume_20.iloc[-1]) if not avg_volume_20.empty and avg_volume_20.iloc[-1] > 0 else 1.0

        avg_volume_50 = volume.rolling(50).mean()
        features['volume_ratio_50'] = float(volume.iloc[-1] / avg_volume_50.iloc[-1]) if not avg_volume_50.empty and avg_volume_50.iloc[-1] > 0 else 1.0

        # Volume trend
        volume_sma_10 = volume.rolling(10).mean()
        volume_sma_30 = volume.rolling(30).mean()
        features['volume_trend'] = float(volume_sma_10.iloc[-1] / volume_sma_30.iloc[-1]) if not volume_sma_30.empty and volume_sma_30.iloc[-1] > 0 else 1.0

        # Ease of Movement
        eom = self._calculate_ease_of_movement(high, low, volume, 14)
        features['ease_of_movement'] = float(eom.iloc[-1]) if not eom.empty else 0.0

        # Force Index
        force_index = self._calculate_force_index(close, volume, 13)
        features['force_index'] = float(force_index.iloc[-1]) if not force_index.empty else 0.0

        # Negative Volume Index (NVI)
        nvi = self._calculate_nvi(close, volume)
        features['nvi'] = float(nvi.iloc[-1]) if not nvi.empty else 1000.0

        # Positive Volume Index (PVI)
        pvi = self._calculate_pvi(close, volume)
        features['pvi'] = float(pvi.iloc[-1]) if not pvi.empty else 1000.0

        # Volume Price Trend
        vpt = self._calculate_vpt(close, volume)
        features['vpt'] = float(vpt.iloc[-1]) if not vpt.empty else 0.0

        return features

    # ==========================================
    # CATEGORY 4: VOLATILITY INDICATORS (15+)
    # ==========================================

    def _volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Volatility and range indicators"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']

        # Average True Range (ATR)
        for period in [7, 14, 21]:
            atr = self._calculate_atr(high, low, close, period)
            features[f'atr_{period}'] = float(atr.iloc[-1]) if not atr.empty else 0.0
            # ATR as % of price
            features[f'atr_pct_{period}'] = float(atr.iloc[-1] / close.iloc[-1] * 100) if not atr.empty and close.iloc[-1] > 0 else 0.0

        # Bollinger Bands (20, 2)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
        features['bb_upper'] = float(bb_upper.iloc[-1]) if not bb_upper.empty else float(close.iloc[-1])
        features['bb_middle'] = float(bb_middle.iloc[-1]) if not bb_middle.empty else float(close.iloc[-1])
        features['bb_lower'] = float(bb_lower.iloc[-1]) if not bb_lower.empty else float(close.iloc[-1])
        features['bb_width'] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] * 100) if not bb_middle.empty and bb_middle.iloc[-1] > 0 else 0.0
        features['bb_position'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100) if not bb_upper.empty and (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 50.0

        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(high, low, close, 20, 2)
        features['keltner_upper'] = float(kc_upper.iloc[-1]) if not kc_upper.empty else float(close.iloc[-1])
        features['keltner_lower'] = float(kc_lower.iloc[-1]) if not kc_lower.empty else float(close.iloc[-1])

        # Historical Volatility (Standard Deviation)
        for period in [10, 20, 30]:
            returns = close.pct_change()
            vol = returns.rolling(period).std() * np.sqrt(252) * 100  # Annualized
            features[f'historical_vol_{period}'] = float(vol.iloc[-1]) if not vol.empty else 0.0

        # Donchian Channels
        dc_upper, dc_lower = self._calculate_donchian_channels(high, low, 20)
        features['donchian_upper'] = float(dc_upper.iloc[-1]) if not dc_upper.empty else float(high.iloc[-1])
        features['donchian_lower'] = float(dc_lower.iloc[-1]) if not dc_lower.empty else float(low.iloc[-1])

        return features

    # ==========================================
    # CATEGORY 5: PRICE PATTERNS (25+)
    # ==========================================

    def _price_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Price patterns and support/resistance"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        # Pivot Points (Standard)
        pivot, r1, r2, s1, s2 = self._calculate_pivot_points(high, low, close)
        features['pivot_point'] = float(pivot)
        features['resistance_1'] = float(r1)
        features['resistance_2'] = float(r2)
        features['support_1'] = float(s1)
        features['support_2'] = float(s2)
        features['price_vs_pivot'] = float((close.iloc[-1] - pivot) / pivot * 100)

        # Candlestick patterns (simplified)
        features['body_size'] = float(abs(close.iloc[-1] - open_price.iloc[-1]))
        features['upper_shadow'] = float(high.iloc[-1] - max(open_price.iloc[-1], close.iloc[-1]))
        features['lower_shadow'] = float(min(open_price.iloc[-1], close.iloc[-1]) - low.iloc[-1])
        features['is_bullish_candle'] = 1.0 if close.iloc[-1] > open_price.iloc[-1] else 0.0

        # Gap detection
        if len(close) >= 2:
            gap = (open_price.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            features['gap_pct'] = float(gap)
            features['has_gap_up'] = 1.0 if gap > 0.5 else 0.0
            features['has_gap_down'] = 1.0 if gap < -0.5 else 0.0
        else:
            features['gap_pct'] = 0.0
            features['has_gap_up'] = 0.0
            features['has_gap_down'] = 0.0

        # Price ranges
        for period in [5, 10, 20]:
            high_range = high.rolling(period).max()
            low_range = low.rolling(period).min()
            features[f'range_position_{period}'] = float((close.iloc[-1] - low_range.iloc[-1]) / (high_range.iloc[-1] - low_range.iloc[-1]) * 100) if (high_range.iloc[-1] - low_range.iloc[-1]) > 0 else 50.0

        # Swing highs and lows
        swing_high = self._find_swing_high(high, 5)
        swing_low = self._find_swing_low(low, 5)
        features['distance_to_swing_high'] = float((swing_high - close.iloc[-1]) / close.iloc[-1] * 100) if swing_high > 0 else 0.0
        features['distance_to_swing_low'] = float((close.iloc[-1] - swing_low) / close.iloc[-1] * 100) if swing_low > 0 else 0.0

        # Higher highs / lower lows detection
        features['higher_high'] = 1.0 if len(high) >= 10 and high.iloc[-1] > high.iloc[-10:].max() else 0.0
        features['lower_low'] = 1.0 if len(low) >= 10 and low.iloc[-1] < low.iloc[-10:].min() else 0.0

        # Fibonacci retracement levels (last 20 bars)
        if len(high) >= 20:
            swing_high = high.iloc[-20:].max()
            swing_low = low.iloc[-20:].min()
            diff = swing_high - swing_low
            features['fib_0_236'] = float(swing_high - 0.236 * diff)
            features['fib_0_382'] = float(swing_high - 0.382 * diff)
            features['fib_0_500'] = float(swing_high - 0.500 * diff)
            features['fib_0_618'] = float(swing_high - 0.618 * diff)
        else:
            features['fib_0_236'] = float(close.iloc[-1])
            features['fib_0_382'] = float(close.iloc[-1])
            features['fib_0_500'] = float(close.iloc[-1])
            features['fib_0_618'] = float(close.iloc[-1])

        return features

    # ==========================================
    # CATEGORY 6: STATISTICAL FEATURES (20+)
    # ==========================================

    def _statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Statistical and mathematical features"""
        features = {}
        close = df['close']

        # Returns (multiple periods)
        for period in [1, 5, 10, 20]:
            returns = close.pct_change(period) * 100
            features[f'return_{period}d'] = float(returns.iloc[-1]) if not returns.empty else 0.0

        # Rolling statistics
        for period in [10, 20, 50]:
            returns = close.pct_change()
            features[f'mean_return_{period}'] = float(returns.rolling(period).mean().iloc[-1] * 100) if len(returns) >= period else 0.0
            features[f'std_return_{period}'] = float(returns.rolling(period).std().iloc[-1] * 100) if len(returns) >= period else 0.0
            features[f'skew_{period}'] = float(returns.rolling(period).skew().iloc[-1]) if len(returns) >= period else 0.0
            features[f'kurtosis_{period}'] = float(returns.rolling(period).kurt().iloc[-1]) if len(returns) >= period else 0.0

        # Z-scores
        for period in [20, 50]:
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            z_score = (close - mean) / std
            features[f'z_score_{period}'] = float(z_score.iloc[-1]) if not z_score.empty and len(z_score) >= period else 0.0

        # Sharpe Ratio (simplified - 20 day)
        if len(close) >= 20:
            returns = close.pct_change()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
            features['sharpe_ratio_20'] = float(sharpe)
        else:
            features['sharpe_ratio_20'] = 0.0

        # Linear regression slope (trend strength)
        for period in [10, 20, 50]:
            if len(close) >= period:
                x = np.arange(period)
                y = close.iloc[-period:].values
                slope = np.polyfit(x, y, 1)[0]
                features[f'lr_slope_{period}'] = float(slope / close.iloc[-1] * 100)  # Normalized
            else:
                features[f'lr_slope_{period}'] = 0.0

        return features

    # ==========================================
    # CATEGORY 7: MARKET STRUCTURE (15+)
    # ==========================================

    def _market_structure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Market structure and regime detection"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Trend strength (ADX-based)
        adx, _, _ = self._calculate_adx(high, low, close, 14)
        features['trend_strength'] = float(adx.iloc[-1]) if not adx.empty else 25.0

        # Market regime (trending vs ranging)
        # Using ADX threshold: >25 = trending, <20 = ranging
        features['is_trending'] = 1.0 if not adx.empty and adx.iloc[-1] > 25 else 0.0
        features['is_ranging'] = 1.0 if not adx.empty and adx.iloc[-1] < 20 else 0.0

        # Price momentum score (composite)
        rsi_14 = self._calculate_rsi(close, 14)
        macd, signal, _ = self._calculate_macd(close, 12, 26, 9)

        momentum_score = 0.0
        if not rsi_14.empty:
            momentum_score += (rsi_14.iloc[-1] - 50) / 50  # -1 to +1
        if not macd.empty and not signal.empty:
            momentum_score += 1 if macd.iloc[-1] > signal.iloc[-1] else -1

        features['momentum_score'] = float(momentum_score / 2)  # Normalize

        # Bullish/bearish structure
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        features['bullish_structure'] = 1.0 if (not sma_20.empty and not sma_50.empty and
                                                 sma_20.iloc[-1] > sma_50.iloc[-1] and
                                                 close.iloc[-1] > sma_20.iloc[-1]) else 0.0

        features['bearish_structure'] = 1.0 if (not sma_20.empty and not sma_50.empty and
                                                 sma_20.iloc[-1] < sma_50.iloc[-1] and
                                                 close.iloc[-1] < sma_20.iloc[-1]) else 0.0

        # Consecutive up/down days
        changes = close.diff()
        up_days = (changes > 0).astype(int)
        down_days = (changes < 0).astype(int)

        # Count consecutive
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(up_days) - 1, max(len(up_days) - 10, -1), -1):
            if up_days.iloc[i]:
                consecutive_up += 1
            else:
                break

        for i in range(len(down_days) - 1, max(len(down_days) - 10, -1), -1):
            if down_days.iloc[i]:
                consecutive_down += 1
            else:
                break

        features['consecutive_up_days'] = float(consecutive_up)
        features['consecutive_down_days'] = float(consecutive_down)

        # New highs/lows
        features['near_52w_high'] = 1.0 if len(high) >= 252 and close.iloc[-1] > high.iloc[-252:].max() * 0.95 else 0.0
        features['near_52w_low'] = 1.0 if len(low) >= 252 and close.iloc[-1] < low.iloc[-252:].min() * 1.05 else 0.0

        # Volume strength
        avg_vol = volume.rolling(20).mean()
        features['volume_strength'] = float(volume.iloc[-1] / avg_vol.iloc[-1]) if not avg_vol.empty and avg_vol.iloc[-1] > 0 else 1.0

        # Price vs moving averages (alignment)
        ma_alignment_score = 0
        for period in [10, 20, 50, 100, 200]:
            ma = close.rolling(period).mean()
            if not ma.empty and len(ma) >= period:
                if close.iloc[-1] > ma.iloc[-1]:
                    ma_alignment_score += 1
                else:
                    ma_alignment_score -= 1

        features['ma_alignment_score'] = float(ma_alignment_score / 5)  # Normalize to -1 to +1

        return features

    # ==========================================
    # CATEGORY 8: MULTI-TIMEFRAME (20+)
    # ==========================================

    def _multi_timeframe_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Multi-timeframe aggregations and relationships"""
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Aggregate to different timeframes
        # Note: This assumes daily data; for intraday, you'd resample differently

        # Weekly aggregations (last 5 days)
        if len(close) >= 5:
            weekly_high = high.iloc[-5:].max()
            weekly_low = low.iloc[-5:].min()
            weekly_close = close.iloc[-1]
            weekly_open = close.iloc[-5]

            features['weekly_range'] = float((weekly_high - weekly_low) / weekly_low * 100)
            features['weekly_return'] = float((weekly_close - weekly_open) / weekly_open * 100)
            features['weekly_position'] = float((weekly_close - weekly_low) / (weekly_high - weekly_low) * 100) if (weekly_high - weekly_low) > 0 else 50.0
        else:
            features['weekly_range'] = 0.0
            features['weekly_return'] = 0.0
            features['weekly_position'] = 50.0

        # Monthly aggregations (last 20 days)
        if len(close) >= 20:
            monthly_high = high.iloc[-20:].max()
            monthly_low = low.iloc[-20:].min()
            monthly_close = close.iloc[-1]
            monthly_open = close.iloc[-20]

            features['monthly_range'] = float((monthly_high - monthly_low) / monthly_low * 100)
            features['monthly_return'] = float((monthly_close - monthly_open) / monthly_open * 100)
            features['monthly_position'] = float((monthly_close - monthly_low) / (monthly_high - monthly_low) * 100) if (monthly_high - monthly_low) > 0 else 50.0
        else:
            features['monthly_range'] = 0.0
            features['monthly_return'] = 0.0
            features['monthly_position'] = 50.0

        # Quarterly aggregations (last 60 days)
        if len(close) >= 60:
            quarterly_high = high.iloc[-60:].max()
            quarterly_low = low.iloc[-60:].min()
            quarterly_close = close.iloc[-1]
            quarterly_open = close.iloc[-60]

            features['quarterly_range'] = float((quarterly_high - quarterly_low) / quarterly_low * 100)
            features['quarterly_return'] = float((quarterly_close - quarterly_open) / quarterly_open * 100)
            features['quarterly_position'] = float((quarterly_close - quarterly_low) / (quarterly_high - quarterly_low) * 100) if (quarterly_high - quarterly_low) > 0 else 50.0
        else:
            features['quarterly_range'] = 0.0
            features['quarterly_return'] = 0.0
            features['quarterly_position'] = 50.0

        # RSI on different timeframes
        if len(close) >= 5:
            weekly_prices = close.iloc[::5]  # Sample every 5th day
            if len(weekly_prices) >= 14:
                weekly_rsi = self._calculate_rsi(weekly_prices, 14)
                features['weekly_rsi'] = float(weekly_rsi.iloc[-1]) if not weekly_rsi.empty else 50.0
            else:
                features['weekly_rsi'] = 50.0
        else:
            features['weekly_rsi'] = 50.0

        # MACD on weekly timeframe
        if len(close) >= 26 * 5:  # Need 26 weeks of data
            weekly_prices = close.iloc[::5]
            macd, signal, _ = self._calculate_macd(weekly_prices, 12, 26, 9)
            features['weekly_macd'] = float(macd.iloc[-1]) if not macd.empty else 0.0
            features['weekly_macd_signal'] = float(signal.iloc[-1]) if not signal.empty else 0.0
        else:
            features['weekly_macd'] = 0.0
            features['weekly_macd_signal'] = 0.0

        # Volume trends across timeframes
        features['volume_5d_avg'] = float(volume.iloc[-5:].mean()) if len(volume) >= 5 else float(volume.iloc[-1])
        features['volume_20d_avg'] = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.iloc[-1])
        features['volume_60d_avg'] = float(volume.rolling(60).mean().iloc[-1]) if len(volume) >= 60 else float(volume.iloc[-1])

        # Volatility across timeframes
        returns = close.pct_change()
        features['volatility_5d'] = float(returns.iloc[-5:].std() * np.sqrt(252) * 100) if len(returns) >= 5 else 0.0
        features['volatility_20d'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100) if len(returns) >= 20 else 0.0
        features['volatility_60d'] = float(returns.rolling(60).std().iloc[-1] * np.sqrt(252) * 100) if len(returns) >= 60 else 0.0

        return features

    # ==========================================
    # CATEGORY 9: CONTEXTUAL FEATURES (6) - NEW!
    # ==========================================

    def _contextual_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Contextual features about the stock itself

        These features tell the model WHAT KIND of stock this is,
        so it can apply the appropriate learned patterns.

        Features:
        1. Market cap category (0=small, 1=mid, 2=large)
        2. Market cap value (in billions)
        3. Sector code (0-10 for 11 GICS sectors)
        4. Beta (market correlation/volatility)
        5. Historical volatility (60-day average)
        6. Liquidity score (log of average volume)
        """
        features = {}

        try:
            # Get metadata from core symbols list
            meta = get_symbol_metadata(symbol)

            # Feature 1: Market Cap Category (encoded)
            market_cap_category = meta.get('market_cap_category', 'unknown')
            if market_cap_category == 'small_cap':
                features['market_cap_category'] = 0
            elif market_cap_category == 'mid_cap':
                features['market_cap_category'] = 1
            elif market_cap_category == 'large_cap':
                features['market_cap_category'] = 2
            else:
                features['market_cap_category'] = 1  # Default to mid

            # Feature 3: Sector Code (0-10)
            features['sector_code'] = meta.get('sector_code', -1)

            # Fetch live data for real-time metrics
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Feature 2: Market Cap Value (billions)
            market_cap = info.get('marketCap', 0)
            features['market_cap_billions'] = float(market_cap / 1e9) if market_cap > 0 else 0.0

            # Feature 4: Beta (market correlation)
            beta = info.get('beta', None)
            if beta is None or np.isnan(beta) or np.isinf(beta):
                beta = 1.0  # Default to market average
            features['beta'] = float(beta)

            # Feature 5: Historical Volatility (60-day)
            if len(df) >= 60:
                returns = df['close'].pct_change()
                volatility_60d = returns.rolling(60).std().iloc[-1] * np.sqrt(252) * 100
                features['historical_volatility_60d'] = float(volatility_60d) if not np.isnan(volatility_60d) else 0.0
            else:
                features['historical_volatility_60d'] = 0.0

            # Feature 6: Liquidity Score (log of average volume)
            avg_volume = df['volume'].tail(20).mean() if len(df) >= 20 else df['volume'].mean()
            if avg_volume > 0:
                features['liquidity_score'] = float(np.log10(avg_volume))
            else:
                features['liquidity_score'] = 5.0  # Default (100K volume)

        except Exception as e:
            # If any errors, use safe defaults
            print(f"[WARNING] Could not fetch contextual features for {symbol}: {e}")
            features['market_cap_category'] = 1  # mid cap
            features['market_cap_billions'] = 0.0
            features['sector_code'] = -1  # unknown
            features['beta'] = 1.0  # market average
            features['historical_volatility_60d'] = 0.0
            features['liquidity_score'] = 5.0

        return features

    # ==========================================
    # CATEGORY 10: EVENT FEATURES (23)
    # ==========================================

    def _event_features(self, df: pd.DataFrame, symbol: str, target_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Event-based features from SEC filings and news

        Extracts 23 time-windowed event features:
        - Event counts by type (12 features)
        - Severity metrics (2 features)
        - Impact scores (3 features)
        - Sentiment features (2 features)
        - Temporal patterns (2 features)
        - Complexity metrics (2 features)
        """
        features = {}

        try:
            # Determine target date (last date in df if not provided)
            if target_date is None:
                if hasattr(df.index, 'to_pydatetime'):
                    target_date = df.index[-1].to_pydatetime()
                else:
                    target_date = datetime.now()

            # Calculate date range for event retrieval (90 days lookback)
            end_date = target_date
            start_date = end_date - timedelta(days=90)

            # Get events and classify
            events = self.event_hybrid.ingest_events(symbol, start_date, end_date)

            if len(events) > 0:
                classified = self.event_hybrid.classify_events(events)

                # Encode features for target date
                target_dates = pd.DatetimeIndex([target_date])
                event_features_df = self.event_hybrid.encode_features(symbol, classified, target_dates)

                # Extract features from DataFrame
                if len(event_features_df) > 0:
                    feature_row = event_features_df.iloc[0]

                    # Convert all event features to float
                    for col in event_features_df.columns:
                        features[f'event_{col}'] = float(feature_row[col])
                else:
                    # No features generated, use zeros
                    features.update(self._empty_event_features())
            else:
                # No events found, use zeros
                features.update(self._empty_event_features())

        except Exception as e:
            # On error, use safe defaults (all zeros)
            print(f"[WARNING] Could not fetch event features for {symbol}: {e}")
            features.update(self._empty_event_features())

        return features

    def _empty_event_features(self) -> Dict[str, float]:
        """Return empty/zero event features"""
        return {
            'event_event_count_refinancing_7d': 0.0,
            'event_event_count_refinancing_30d': 0.0,
            'event_event_count_refinancing_90d': 0.0,
            'event_event_count_dividend_7d': 0.0,
            'event_event_count_dividend_30d': 0.0,
            'event_event_count_dividend_90d': 0.0,
            'event_event_count_litigation_7d': 0.0,
            'event_event_count_litigation_30d': 0.0,
            'event_event_count_litigation_90d': 0.0,
            'event_event_count_negative_news_7d': 0.0,
            'event_event_count_negative_news_30d': 0.0,
            'event_event_count_negative_news_90d': 0.0,
            'event_max_event_severity_30d': 0.0,
            'event_time_since_last_high_severity_event': 999.0,
            'event_sum_impact_dividend_90d': 0.0,
            'event_sum_impact_liquidity_90d': 0.0,
            'event_sum_impact_credit_90d': 0.0,
            'event_news_sentiment_mean_7d': 0.0,
            'event_news_sentiment_min_7d': 0.0,
            'event_event_intensity_acceleration_ratio': 0.0,
            'event_cross_source_confirmation_flag': 0.0,
            'event_information_asymmetry_proxy_score': 0.0,
            'event_filing_complexity_index': 0.0
        }

    # ==========================================
    # CATEGORY 11: DERIVED FEATURES (30+)
    # ==========================================

    def _derived_features(self, df: pd.DataFrame, base_features: Dict[str, float]) -> Dict[str, float]:
        """Derived and interaction features from base features"""
        features = {}

        # RSI divergence (RSI momentum vs price momentum)
        if 'rsi_14' in base_features and 'momentum_20' in base_features:
            rsi_momentum = base_features['rsi_14'] - 50  # Center at 0
            price_momentum_sign = 1 if base_features['momentum_20'] > 0 else -1
            features['rsi_price_divergence'] = float(abs(rsi_momentum / 50) - price_momentum_sign)

        # MACD vs RSI agreement
        if 'macd_histogram' in base_features and 'rsi_14' in base_features:
            macd_bullish = 1 if base_features['macd_histogram'] > 0 else 0
            rsi_bullish = 1 if base_features['rsi_14'] > 50 else 0
            features['macd_rsi_agreement'] = 1.0 if macd_bullish == rsi_bullish else 0.0

        # Volume confirmation
        if 'volume_ratio_20' in base_features and 'return_1d' in base_features:
            # High volume + strong move = confirmed move
            features['volume_confirmed_move'] = float(base_features['volume_ratio_20'] * abs(base_features['return_1d']))

        # Bollinger Band squeeze
        if 'bb_width' in base_features and 'atr_pct_14' in base_features:
            features['bb_squeeze'] = float(base_features['bb_width'] / base_features['atr_pct_14']) if base_features['atr_pct_14'] > 0 else 1.0

        # Trend consistency
        if 'ema_10' in base_features and 'ema_20' in base_features and 'ema_50' in base_features:
            ema_aligned = (base_features['ema_10'] > base_features['ema_20'] and
                          base_features['ema_20'] > base_features['ema_50'])
            features['ema_bullish_alignment'] = 1.0 if ema_aligned else 0.0

            ema_bearish = (base_features['ema_10'] < base_features['ema_20'] and
                          base_features['ema_20'] < base_features['ema_50'])
            features['ema_bearish_alignment'] = 1.0 if ema_bearish else 0.0

        # Momentum quality
        if 'rsi_14' in base_features and 'adx_14' in base_features:
            # Strong trend + overbought/oversold = high quality signal
            features['momentum_quality'] = float((abs(base_features['rsi_14'] - 50) / 50) * (base_features['adx_14'] / 50))

        # Price position composite
        if 'bb_position' in base_features and 'range_position_20' in base_features:
            features['composite_price_position'] = float((base_features['bb_position'] + base_features['range_position_20']) / 2)

        # Volatility regime
        if 'historical_vol_20' in base_features:
            # Categorize: Low (<15), Medium (15-30), High (>30)
            vol = base_features['historical_vol_20']
            features['low_volatility_regime'] = 1.0 if vol < 15 else 0.0
            features['medium_volatility_regime'] = 1.0 if 15 <= vol <= 30 else 0.0
            features['high_volatility_regime'] = 1.0 if vol > 30 else 0.0

        # Multi-timeframe trend agreement
        if 'return_5d' in base_features and 'return_20d' in base_features:
            short_trend = 1 if base_features['return_5d'] > 0 else -1
            long_trend = 1 if base_features['return_20d'] > 0 else -1
            features['trend_agreement'] = 1.0 if short_trend == long_trend else 0.0

        # Support/resistance proximity
        if 'distance_to_swing_high' in base_features and 'distance_to_swing_low' in base_features:
            features['near_resistance'] = 1.0 if base_features['distance_to_swing_high'] < 2.0 else 0.0
            features['near_support'] = 1.0 if base_features['distance_to_swing_low'] < 2.0 else 0.0

        # Oversold/overbought composite
        if 'rsi_14' in base_features and 'stochastic_k' in base_features:
            oversold_score = (1 if base_features['rsi_14'] < 30 else 0) + (1 if base_features['stochastic_k'] < 20 else 0)
            overbought_score = (1 if base_features['rsi_14'] > 70 else 0) + (1 if base_features['stochastic_k'] > 80 else 0)
            features['oversold_composite'] = float(oversold_score / 2)
            features['overbought_composite'] = float(overbought_score / 2)

        # Price momentum vs volume momentum
        if 'return_10d' in base_features and 'volume_trend' in base_features:
            features['price_volume_divergence'] = float(abs(base_features['return_10d'] / 10) - (base_features['volume_trend'] - 1))

        # Trend strength composite
        if 'adx_14' in base_features and 'ma_alignment_score' in base_features:
            features['trend_strength_composite'] = float((base_features['adx_14'] / 50) * base_features['ma_alignment_score'])

        # Gap significance
        if 'gap_pct' in base_features and 'atr_pct_14' in base_features:
            # Gap relative to ATR
            features['gap_significance'] = float(abs(base_features['gap_pct']) / base_features['atr_pct_14']) if base_features['atr_pct_14'] > 0 else 0.0

        # Breakout potential
        if 'range_position_20' in base_features and 'volume_ratio_20' in base_features and 'bb_width' in base_features:
            # Near range top + high volume + narrow BB = potential breakout
            near_top = 1 if base_features['range_position_20'] > 80 else 0
            high_vol = 1 if base_features['volume_ratio_20'] > 1.5 else 0
            narrow_bb = 1 if base_features['bb_width'] < 2.0 else 0
            features['breakout_potential'] = float((near_top + high_vol + narrow_bb) / 3)

        # Mean reversion signal
        if 'z_score_20' in base_features and 'rsi_14' in base_features:
            # Extreme z-score + extreme RSI = mean reversion opportunity
            extreme_z = 1 if abs(base_features['z_score_20']) > 2 else 0
            extreme_rsi = 1 if base_features['rsi_14'] < 30 or base_features['rsi_14'] > 70 else 0
            features['mean_reversion_signal'] = float((extreme_z + extreme_rsi) / 2)

        # Momentum acceleration
        if 'roc_5' in base_features and 'roc_10' in base_features:
            features['momentum_acceleration'] = float(base_features['roc_5'] - base_features['roc_10'])

        # Volatility expansion/contraction
        if 'historical_vol_10' in base_features and 'historical_vol_30' in base_features:
            features['volatility_expansion'] = float(base_features['historical_vol_10'] / base_features['historical_vol_30']) if base_features['historical_vol_30'] > 0 else 1.0

        return features

    # ==========================================
    # HELPER CALCULATION METHODS
    # ==========================================

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             period: int = 14, smooth_k: int = 3) -> tuple:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=smooth_k).mean()
        return k, d

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr

    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        return mfi

    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def _calculate_ultimate_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - pd.concat([low, close.shift(1)], axis=1).min(axis=1)

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        return uo

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Calculate Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    def _calculate_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series,
                                af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        sar = close.copy()
        # Simplified implementation - just return close for now
        # Full implementation would track uptrend/downtrend and EP
        return sar

    def _calculate_supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             period: int = 10, multiplier: float = 3) -> pd.Series:
        """Calculate Supertrend"""
        atr = self._calculate_atr(high, low, close, period)
        hl_avg = (high + low) / 2

        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        supertrend.iloc[0] = lower_band.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        return supertrend

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
        return obv

    def _calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad

    def _calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    def _calculate_ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Ease of Movement"""
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 100000000) / (high - low)
        eom = distance / box_ratio
        return eom.rolling(window=period).mean()

    def _calculate_force_index(self, close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """Calculate Force Index"""
        fi = close.diff() * volume
        return fi.ewm(span=period, adjust=False).mean()

    def _calculate_nvi(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Negative Volume Index"""
        nvi = pd.Series(index=close.index, dtype=float)
        nvi.iloc[0] = 1000

        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] + ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * nvi.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]

        return nvi

    def _calculate_pvi(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Positive Volume Index"""
        pvi = pd.Series(index=close.index, dtype=float)
        pvi.iloc[0] = 1000

        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] + ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * pvi.iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]

        return pvi

    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend"""
        vpt = (volume * (close.pct_change())).cumsum()
        return vpt

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series,
                                   period: int = 20, multiplier: float = 2) -> tuple:
        """Calculate Keltner Channels"""
        middle = close.ewm(span=period, adjust=False).mean()
        atr = self._calculate_atr(high, low, close, period)
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        return upper, middle, lower

    def _calculate_donchian_channels(self, high: pd.Series, low: pd.Series, period: int = 20) -> tuple:
        """Calculate Donchian Channels"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        return upper, lower

    def _calculate_pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """Calculate Standard Pivot Points"""
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        r1 = 2 * pivot - low.iloc[-1]
        r2 = pivot + (high.iloc[-1] - low.iloc[-1])
        s1 = 2 * pivot - high.iloc[-1]
        s2 = pivot - (high.iloc[-1] - low.iloc[-1])
        return pivot, r1, r2, s1, s2

    def _find_swing_high(self, high: pd.Series, period: int = 5) -> float:
        """Find most recent swing high"""
        if len(high) < period * 2:
            return high.max()

        for i in range(len(high) - period - 1, period, -1):
            if high.iloc[i] == high.iloc[i-period:i+period+1].max():
                return high.iloc[i]

        return high.max()

    def _find_swing_low(self, low: pd.Series, period: int = 5) -> float:
        """Find most recent swing low"""
        if len(low) < period * 2:
            return low.min()

        for i in range(len(low) - period - 1, period, -1):
            if low.iloc[i] == low.iloc[i-period:i+period+1].min():
                return low.iloc[i]

        return low.min()

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature set when data is insufficient"""
        return {
            'feature_count': 0,
            'error': 'Insufficient data for feature extraction'
        }


if __name__ == '__main__':
    # Test feature extraction
    import yfinance as yf

    print("Testing Feature Engineer with AAPL data...")

    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")

    # Rename columns to lowercase
    df.columns = df.columns.str.lower()

    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_features(df, symbol="AAPL")

    print(f"\nExtracted {features['feature_count']} features")
    print(f"Symbol: {features['symbol']}")
    print(f"Last price: ${features['last_price']:.2f}")

    # Show sample features
    print("\nSample Momentum Features:")
    for key in ['rsi_14', 'stochastic_k', 'roc_10', 'mfi_14']:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")

    print("\nSample Trend Features:")
    for key in ['sma_20', 'ema_50', 'macd_histogram', 'adx_14']:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")

    print("\nSample Volume Features:")
    for key in ['obv', 'volume_ratio_20', 'cmf_20']:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")

    print("\nContextual Features (NEW!):")
    for key in ['market_cap_category', 'market_cap_billions', 'sector_code', 'beta', 'historical_volatility_60d', 'liquidity_score']:
        if key in features:
            print(f"  {key}: {features[key]:.2f}")

    print("\n[OK] Feature engineering test complete!")
