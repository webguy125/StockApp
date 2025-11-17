"""
Feature Engineering Module
Computes ML features for pivot reliability prediction
Pure Python implementations - no TA-Lib dependency
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(closes, period=14):
    """
    Calculate Relative Strength Index (RSI)

    Args:
        closes: Array of closing prices
        period: RSI period (default: 14)

    Returns:
        numpy array: RSI values (0-100)
    """
    try:
        closes = np.array(closes, dtype=np.float64)

        # Calculate price changes
        deltas = np.diff(closes)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Initialize arrays
        rsi = np.zeros_like(closes)
        rsi[:] = np.nan

        # Calculate initial averages using SMA
        if len(gains) >= period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            # Wilder's smoothing method
            for i in range(period, len(closes)):
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    except Exception as e:
        logger.error(f"❌ RSI calculation failed: {e}")
        return np.full(len(closes), np.nan)


def calculate_atr(highs, lows, closes, period=14):
    """
    Calculate Average True Range (ATR)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ATR period (default: 14)

    Returns:
        numpy array: ATR values
    """
    try:
        highs = np.array(highs, dtype=np.float64)
        lows = np.array(lows, dtype=np.float64)
        closes = np.array(closes, dtype=np.float64)

        # True Range components
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))

        # True Range is max of the three
        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # First TR is undefined (no previous close)
        tr[0] = high_low[0]

        # Calculate ATR using Wilder's smoothing
        atr = np.zeros_like(closes)
        atr[:] = np.nan

        if len(tr) >= period:
            # Initial ATR is SMA of TR
            atr[period - 1] = np.mean(tr[:period])

            # Wilder's smoothing
            for i in range(period, len(closes)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    except Exception as e:
        logger.error(f"❌ ATR calculation failed: {e}")
        return np.full(len(closes), np.nan)


def calculate_adx(highs, lows, closes, period=14):
    """
    Calculate Average Directional Index (ADX)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ADX period (default: 14)

    Returns:
        numpy array: ADX values (0-100)
    """
    try:
        highs = np.array(highs, dtype=np.float64)
        lows = np.array(lows, dtype=np.float64)
        closes = np.array(closes, dtype=np.float64)

        # Calculate directional movements
        high_diff = np.diff(highs)
        low_diff = -np.diff(lows)

        # Positive and negative directional movements
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Add zero at start to align with price arrays
        plus_dm = np.insert(plus_dm, 0, 0)
        minus_dm = np.insert(minus_dm, 0, 0)

        # Calculate ATR for normalization
        atr = calculate_atr(highs, lows, closes, period)

        # Smooth directional movements using Wilder's method
        plus_di = np.zeros_like(closes)
        minus_di = np.zeros_like(closes)
        plus_di[:] = np.nan
        minus_di[:] = np.nan

        if len(plus_dm) >= period:
            # Initial smoothed values
            smoothed_plus_dm = np.mean(plus_dm[:period])
            smoothed_minus_dm = np.mean(minus_dm[:period])

            for i in range(period, len(closes)):
                smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm[i]) / period
                smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm[i]) / period

                if atr[i] > 0:
                    plus_di[i] = 100 * smoothed_plus_dm / atr[i]
                    minus_di[i] = 100 * smoothed_minus_dm / atr[i]

        # Calculate DX (Directional Movement Index)
        dx = np.zeros_like(closes)
        dx[:] = np.nan

        for i in range(len(closes)):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                di_sum = plus_di[i] + minus_di[i]
                if di_sum > 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum

        # Calculate ADX (smoothed DX)
        adx = np.zeros_like(closes)
        adx[:] = np.nan

        # Find first valid DX index
        valid_idx = np.where(~np.isnan(dx))[0]
        if len(valid_idx) >= period:
            start_idx = valid_idx[0] + period - 1
            adx[start_idx] = np.nanmean(dx[valid_idx[0]:start_idx + 1])

            # Wilder's smoothing for ADX
            for i in range(start_idx + 1, len(closes)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx

    except Exception as e:
        logger.error(f"❌ ADX calculation failed: {e}")
        return np.full(len(closes), np.nan)


def calculate_sma(values, period):
    """
    Calculate Simple Moving Average

    Args:
        values: Array of values
        period: SMA period

    Returns:
        numpy array: SMA values
    """
    values = np.array(values, dtype=np.float64)
    sma = np.full(len(values), np.nan)

    for i in range(period - 1, len(values)):
        sma[i] = np.mean(values[i - period + 1:i + 1])

    return sma


def calculate_linear_regression(values, period):
    """
    Calculate linear regression slope over a rolling window

    Args:
        values: Array of values
        period: Regression window

    Returns:
        numpy array: Regression slopes
    """
    values = np.array(values, dtype=np.float64)
    slopes = np.full(len(values), np.nan)

    x = np.arange(period)

    for i in range(period - 1, len(values)):
        y = values[i - period + 1:i + 1]

        if not np.any(np.isnan(y)):
            # Linear regression: y = mx + b
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator != 0:
                slopes[i] = numerator / denominator

    return slopes


def normalize_values(values, lookback):
    """
    Normalize values to range [-1, 1] over rolling window

    Args:
        values: Array of values to normalize
        lookback: Window size for min/max

    Returns:
        numpy array: Normalized values
    """
    values = np.array(values, dtype=np.float64)
    normalized = np.full(len(values), np.nan)

    for i in range(lookback - 1, len(values)):
        window = values[i - lookback + 1:i + 1]

        if not np.any(np.isnan(window)):
            min_val = np.min(window)
            max_val = np.max(window)

            if max_val - min_val > 0:
                # Scale to [-1, 1]
                normalized[i] = 2 * (values[i] - min_val) / (max_val - min_val) - 1
            else:
                normalized[i] = 0

    return normalized


def calculate_triad_components(candles, settings):
    """
    Calculate the 3 core Triad Trend Pulse components

    Args:
        candles: DataFrame with OHLCV data
        settings: dict with indicator settings

    Returns:
        dict: {weighted_trend, oscillator, short_trend}
    """
    try:
        closes = candles['close'].values
        highs = candles['high'].values
        lows = candles['low'].values
        volumes = candles['volume'].values

        reg_length = settings.get('reg_length', 20)
        trend_bar_length = settings.get('trend_bar_length', 5)

        # 1. Weighted Price Regression
        hlc3 = (highs + lows + closes) / 3
        weighted_price = (hlc3 * 0.5) + (closes * 0.5)

        price_regression = calculate_linear_regression(closes, reg_length)
        weighted_regression = calculate_linear_regression(weighted_price, reg_length)

        chart_blend = 0.6
        blended_regression = (price_regression * chart_blend +
                            weighted_regression * (1 - chart_blend))

        weighted_trend = normalize_values(blended_regression, 50)

        # 2. Adaptive Oscillator
        price_trend = calculate_linear_regression(closes, reg_length)

        # Momentum
        momentum = np.full(len(closes), np.nan)
        for i in range(reg_length, len(closes)):
            momentum[i] = (closes[i] - closes[i - reg_length]) / reg_length

        # Volume trend
        volume_sma = calculate_sma(volumes, reg_length)
        volume_trend = np.full(len(volumes), np.nan)
        volume_trend[reg_length:] = volumes[reg_length:] / volume_sma[reg_length:] - 1

        # Combine components
        oscillator = (price_trend * 0.3 + momentum * 0.3 +
                     volume_trend * 0.2 + price_trend * 0.2)

        # Smooth
        oscillator_smooth = calculate_sma(oscillator, 3)
        oscillator_norm = normalize_values(oscillator_smooth, 50)

        # 3. Short-Term Trend
        short_trend = np.full(len(closes), np.nan)
        for i in range(trend_bar_length, len(closes)):
            short_trend[i] = (closes[i] - closes[i - trend_bar_length]) / trend_bar_length

        short_trend_norm = normalize_values(short_trend, 50)

        return {
            'weighted_trend': weighted_trend,
            'oscillator': oscillator_norm,
            'short_trend': short_trend_norm
        }

    except Exception as e:
        logger.error(f"❌ Triad component calculation failed: {e}")
        return {
            'weighted_trend': np.full(len(candles), np.nan),
            'oscillator': np.full(len(candles), np.nan),
            'short_trend': np.full(len(candles), np.nan)
        }


def compute_ml_features(candles, indicator_outputs=None, timeframe='1d'):
    """
    Compute all 9 ML features for pivot reliability prediction

    Args:
        candles: DataFrame with OHLCV data
        indicator_outputs: dict with Triad components (optional, will calculate if not provided)
        timeframe: Timeframe string for encoding

    Returns:
        pandas DataFrame with 9 feature columns
    """
    try:
        logger.info(f"📊 Computing ML features for {len(candles)} bars...")

        # Extract OHLCV
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        volumes = candles['volume'].values

        # Default settings for Triad components
        settings = {
            'reg_length': 20,
            'rsi_len': 14,
            'trend_bar_length': 5,
            'pivot_lookback': 5,
            'adx_threshold': 25
        }

        # Calculate Triad components if not provided
        if indicator_outputs is None:
            indicator_outputs = calculate_triad_components(candles, settings)

        # Feature 1-3: Triad components (already normalized to [-1, 1])
        weighted_trend_norm = indicator_outputs['weighted_trend']
        oscillator_norm = indicator_outputs['oscillator']
        short_trend_norm = indicator_outputs['short_trend']

        # Feature 4: ADX (normalized to [0, 1])
        adx = calculate_adx(highs, lows, closes, 14)
        adx_norm = adx / 100.0

        # Feature 5: Volume change (current volume / 20-bar SMA)
        volume_sma = calculate_sma(volumes, 20)
        volume_change = np.full(len(volumes), np.nan)
        volume_change[20:] = volumes[20:] / volume_sma[20:]
        volume_change_norm = np.clip(volume_change, 0, 3) / 3  # Cap at 3x, normalize to [0, 1]

        # Feature 6: ATR (normalized by price)
        atr = calculate_atr(highs, lows, closes, 14)
        atr_norm = atr / closes  # ATR as percentage of price

        # Feature 7: RSI (normalized to [0, 1])
        rsi = calculate_rsi(closes, 14)
        rsi_norm = rsi / 100.0

        # Feature 8: Momentum (price change over lookback)
        lookback = settings['reg_length']
        momentum = np.full(len(closes), np.nan)
        for i in range(lookback, len(closes)):
            momentum[i] = (closes[i] - closes[i - lookback]) / closes[i - lookback]
        momentum_norm = np.tanh(momentum * 10) / 2 + 0.5  # Tanh squashing to [0, 1]

        # Feature 9: Timeframe encoding
        try:
            from timeframe_aggregator import get_timeframe_encoding
        except ImportError:
            from .timeframe_aggregator import get_timeframe_encoding
        timeframe_code = get_timeframe_encoding(timeframe)
        timeframe_norm = timeframe_code / 12.0  # Normalize to [0, 1]

        # Create feature DataFrame
        features = pd.DataFrame({
            'oscillator_range': oscillator_norm,
            'weighted_price_trend_norm': weighted_trend_norm,
            'short_trend_norm': short_trend_norm,
            'adx': adx_norm,
            'volume_change': volume_change_norm,
            'atr': atr_norm,
            'rsi': rsi_norm,
            'momentum': momentum_norm,
            'timeframe': np.full(len(candles), timeframe_norm)
        })

        logger.info(f"✅ Computed {len(features.columns)} features")
        return features

    except Exception as e:
        logger.error(f"❌ Feature computation failed: {e}")
        raise


def extract_pivot_features(candles, pivot_indices, timeframe='1d'):
    """
    Extract features for specific pivot points

    Args:
        candles: DataFrame with OHLCV data
        pivot_indices: List of indices where pivots occur
        timeframe: Timeframe string

    Returns:
        numpy array: Shape (n_pivots, 9) with features for each pivot
    """
    try:
        # Compute all features
        features = compute_ml_features(candles, timeframe=timeframe)

        # Extract only pivot rows
        pivot_features = features.iloc[pivot_indices].values

        # Remove any rows with NaN
        valid_mask = ~np.any(np.isnan(pivot_features), axis=1)
        pivot_features = pivot_features[valid_mask]

        logger.info(f"✅ Extracted features for {len(pivot_features)} valid pivots")
        return pivot_features

    except Exception as e:
        logger.error(f"❌ Pivot feature extraction failed: {e}")
        raise


def batch_compute_features(candles_dict, timeframes):
    """
    Compute features for multiple assets and timeframes

    Args:
        candles_dict: dict of {asset: DataFrame}
        timeframes: list of timeframe strings

    Returns:
        dict: {(asset, timeframe): features_df}
    """
    try:
        results = {}

        for asset, candles in candles_dict.items():
            for timeframe in timeframes:
                logger.info(f"📊 Processing {asset} @ {timeframe}...")

                # Aggregate to timeframe if needed
                from .timeframe_aggregator import aggregate_ohlcv
                candles_agg = aggregate_ohlcv(candles.copy(), timeframe)

                # Compute features
                features = compute_ml_features(candles_agg, timeframe=timeframe)

                results[(asset, timeframe)] = features

        logger.info(f"✅ Batch processing complete: {len(results)} datasets")
        return results

    except Exception as e:
        logger.error(f"❌ Batch feature computation failed: {e}")
        raise


def validate_features(features_df):
    """
    Validate feature DataFrame for ML training

    Args:
        features_df: DataFrame with feature columns

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check for required columns
        required_cols = [
            'oscillator_range',
            'weighted_price_trend_norm',
            'short_trend_norm',
            'adx',
            'volume_change',
            'atr',
            'rsi',
            'momentum',
            'timeframe'
        ]

        missing = set(required_cols) - set(features_df.columns)
        if missing:
            return False, f"Missing columns: {missing}"

        # Check for NaN
        nan_count = features_df[required_cols].isna().sum().sum()
        if nan_count > 0:
            return False, f"Contains {nan_count} NaN values"

        # Check value ranges
        for col in required_cols:
            if col == 'timeframe':
                continue  # Timeframe can be constant

            vals = features_df[col].values
            if np.all(vals == vals[0]):
                return False, f"Column '{col}' has constant values"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {e}"
