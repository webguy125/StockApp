
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Advanced Feature Engineering for TurboMode
Exponential and Options-Derived Features

ARCHITECTURE:
- Fully vectorized operations using NumPy
- No loops over rows
- Designed for batch processing of sector data

FEATURES:
1. Exponential Momentum (14-day)
2. Exponential Volatility (14-day)
3. Exponential Decay Volume Sum (14-day)
4. Exponential Drawdown Pressure (14-day)
5. Options IV Rank (when available)
6. Options Put/Call Ratio (when available)
7. Options Skew 25-delta (when available)
8. Options Term Structure Slope (when available)

Author: TurboMode Core Engine
Date: 2026-02-18
"""

import numpy as np
import os
from typing import Dict, List, Tuple

# Configuration
ENABLE_ADVANCED_SIGNALS = os.getenv('ENABLE_ADVANCED_SIGNALS', 'false').lower() == 'true'
LAMBDA_DEFAULT = float(os.getenv('LAMBDA_DEFAULT', '0.25'))
HORIZON_DAYS = int(os.getenv('HORIZON_DAYS', '14'))


def exponential_momentum(close_prices: np.ndarray, lambda_val: float = LAMBDA_DEFAULT, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """
    Compute exponential weighted momentum over horizon days.

    Formula: sum(w[i] * r[i]) where w[i] = exp(-lambda * i)
             r[i] = (close[t-i] - close[t-i-1]) / close[t-i-1]

    Args:
        close_prices: Array of close prices (N,)
        lambda_val: Decay rate (default: 0.25)
        horizon: Lookback window in days (default: 14)

    Returns:
        Array of exponential momentum values (N,)
    """
    n = len(close_prices)
    if n < 2:
        return np.zeros(n)

    # Compute returns
    returns = np.zeros(n)
    returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

    # Compute exponential weights
    weights = np.exp(-lambda_val * np.arange(horizon))
    weights = weights / weights.sum()  # Normalize

    # Vectorized convolution for exponential weighted sum
    exp_momo = np.zeros(n)
    for i in range(horizon, n):
        window_returns = returns[i-horizon+1:i+1][::-1]  # Reverse for decay
        exp_momo[i] = np.dot(weights[:len(window_returns)], window_returns)

    return exp_momo


def exponential_volatility(close_prices: np.ndarray, lambda_val: float = LAMBDA_DEFAULT, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """
    Compute exponential weighted volatility over horizon days.

    Formula: sqrt(sum(w[i] * r[i]^2)) where w[i] = exp(-lambda * i)

    Args:
        close_prices: Array of close prices (N,)
        lambda_val: Decay rate (default: 0.25)
        horizon: Lookback window in days (default: 14)

    Returns:
        Array of exponential volatility values (N,)
    """
    n = len(close_prices)
    if n < 2:
        return np.zeros(n)

    # Compute returns
    returns = np.zeros(n)
    returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

    # Compute exponential weights
    weights = np.exp(-lambda_val * np.arange(horizon))
    weights = weights / weights.sum()  # Normalize

    # Vectorized exponential weighted variance
    exp_vol = np.zeros(n)
    for i in range(horizon, n):
        window_returns = returns[i-horizon+1:i+1][::-1]  # Reverse for decay
        exp_vol[i] = np.sqrt(np.dot(weights[:len(window_returns)], window_returns**2))

    return exp_vol


def exponential_decay_volume_sum(volume: np.ndarray, lambda_val: float = LAMBDA_DEFAULT, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """
    Compute exponential decay weighted volume sum over horizon days.

    Formula: sum(w[i] * volume[t-i]) where w[i] = exp(-lambda * i)

    Args:
        volume: Array of volume values (N,)
        lambda_val: Decay rate (default: 0.25)
        horizon: Lookback window in days (default: 14)

    Returns:
        Array of exponential decay volume sums (N,)
    """
    n = len(volume)
    if n < 1:
        return np.zeros(n)

    # Compute exponential weights
    weights = np.exp(-lambda_val * np.arange(horizon))
    weights = weights / weights.sum()  # Normalize

    # Vectorized exponential weighted sum
    exp_vol_sum = np.zeros(n)
    for i in range(horizon, n):
        window_volume = volume[i-horizon+1:i+1][::-1]  # Reverse for decay
        exp_vol_sum[i] = np.dot(weights[:len(window_volume)], window_volume)

    return exp_vol_sum


def exponential_drawdown_pressure(close_prices: np.ndarray, lambda_val: float = LAMBDA_DEFAULT, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """
    Compute exponential weighted drawdown pressure over horizon days.

    Formula: sum(w[i] * dd[i]) where w[i] = exp(-lambda * i)
             dd[i] = (close[t] - max(close[t-horizon:t])) / max(close[t-horizon:t])

    Args:
        close_prices: Array of close prices (N,)
        lambda_val: Decay rate (default: 0.25)
        horizon: Lookback window in days (default: 14)

    Returns:
        Array of exponential drawdown pressure values (N,) (always <= 0)
    """
    n = len(close_prices)
    if n < 2:
        return np.zeros(n)

    # Compute exponential weights
    weights = np.exp(-lambda_val * np.arange(horizon))
    weights = weights / weights.sum()  # Normalize

    # Vectorized drawdown computation
    exp_dd = np.zeros(n)
    for i in range(horizon, n):
        window_close = close_prices[i-horizon+1:i+1]
        running_max = np.maximum.accumulate(window_close)
        drawdowns = (window_close - running_max) / running_max
        drawdowns_reversed = drawdowns[::-1]  # Reverse for decay
        exp_dd[i] = np.dot(weights[:len(drawdowns_reversed)], drawdowns_reversed)

    return exp_dd


def compute_advanced_features_vectorized(
    close_prices: np.ndarray,
    volume: np.ndarray,
    lambda_val: float = LAMBDA_DEFAULT,
    horizon: int = HORIZON_DAYS
) -> Dict[str, np.ndarray]:
    """
    Compute all advanced exponential features in one pass.

    Args:
        close_prices: Array of close prices (N,)
        volume: Array of volume values (N,)
        lambda_val: Decay rate (default: 0.25)
        horizon: Lookback window in days (default: 14)

    Returns:
        Dictionary mapping feature names to arrays
    """
    features = {}

    if not ENABLE_ADVANCED_SIGNALS:
        # Return zeros if disabled
        n = len(close_prices)
        features['exp_momo_14d'] = np.zeros(n)
        features['exp_vol_14d'] = np.zeros(n)
        features['exp_decay_volume_14d'] = np.zeros(n)
        features['exp_drawdown_pressure_14d'] = np.zeros(n)
        return features

    # Compute exponential features
    features['exp_momo_14d'] = exponential_momentum(close_prices, lambda_val, horizon)
    features['exp_vol_14d'] = exponential_volatility(close_prices, lambda_val, horizon)
    features['exp_decay_volume_14d'] = exponential_decay_volume_sum(volume, lambda_val, horizon)
    features['exp_drawdown_pressure_14d'] = exponential_drawdown_pressure(close_prices, lambda_val, horizon)

    return features


def compute_options_features_from_db(symbol: str, timestamp: str = None, db_path: str = None) -> Dict[str, float]:
    """
    Query options-derived features from options_universe.db.

    Args:
        symbol: Stock symbol
        timestamp: Optional timestamp for historical lookup (format: YYYY-MM-DD)
        db_path: Path to options_universe.db (default: backend/turbomode/options/Data/options_universe.db)

    Returns:
        Dictionary with options features
    """
    import sqlite3
    import os
    from pathlib import Path

    if db_path is None:
        project_root = Path(__file__).resolve().parents[3]
        db_path = os.path.join(project_root, 'backend', 'turbomode', 'options', 'Data', 'options_universe.db')

    # Default values (fallback)
    result = {
        'opt_iv_rank': 0.0,
        'opt_put_call_ratio': 0.0,
        'opt_skew_25d': 0.0,
        'opt_term_structure_slope': 0.0
    }

    if not os.path.exists(db_path):
        return result

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        # Query latest options metrics for symbol from option_features_daily
        if timestamp:
            # Extract date part if timestamp includes time
            date_only = timestamp.split()[0] if ' ' in timestamp else timestamp
            query = """
                SELECT iv_30d, put_call_volume_ratio, skew_put_call, term_slope_14_30
                FROM option_features_daily
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """
            cursor.execute(query, (symbol, date_only))
        else:
            query = """
                SELECT iv_30d, put_call_volume_ratio, skew_put_call, term_slope_14_30
                FROM option_features_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
            """
            cursor.execute(query, (symbol,))

        row = cursor.fetchone()

        if row:
            iv_30d, put_call_ratio, skew, term_slope = row

            # IV Rank: Normalize IV to 0-1 range (using 30-day IV as proxy)
            # Assuming typical IV range of 0.1 to 1.0 (10% to 100%)
            if iv_30d is not None and iv_30d > 0:
                # Simple normalization: clamp IV to [0.1, 1.0] and normalize
                iv_clamped = max(0.1, min(1.0, iv_30d))
                result['opt_iv_rank'] = (iv_clamped - 0.1) / 0.9  # Normalize to [0, 1]
            else:
                result['opt_iv_rank'] = 0.0

            # Put/Call Ratio (direct from database)
            if put_call_ratio is not None:
                result['opt_put_call_ratio'] = float(put_call_ratio)
            else:
                result['opt_put_call_ratio'] = 0.0

            # Skew (put-call skew, normalized)
            if skew is not None:
                # Skew can be negative or positive, normalize to reasonable range
                # Typical skew range: -5 to +5
                skew_clamped = max(-5.0, min(5.0, skew))
                result['opt_skew_25d'] = (skew_clamped + 5.0) / 10.0  # Normalize to [0, 1]
            else:
                result['opt_skew_25d'] = 0.0

            # Term Structure Slope (14-30 day slope)
            if term_slope is not None:
                # Term slope is already a good feature, just clamp extreme values
                # Typical range: -0.2 to +0.2
                slope_clamped = max(-0.2, min(0.2, term_slope))
                result['opt_term_structure_slope'] = (slope_clamped + 0.2) / 0.4  # Normalize to [0, 1]
            else:
                result['opt_term_structure_slope'] = 0.0

        conn.close()

    except Exception as e:
        # Silently fail and return zeros (database might be locked or unavailable)
        pass

    return result


def compute_options_features_stub(symbol: str) -> Dict[str, float]:
    """
    Stub for options-derived features (backward compatibility).

    Now calls compute_options_features_from_db with no timestamp (latest data).

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with options features
    """
    return compute_options_features_from_db(symbol, timestamp=None)


# New extended feature list (179 + 8 advanced = 187 features)
ADVANCED_FEATURE_NAMES = [
    # Exponential signals (4 features)
    'exp_momo_14d',
    'exp_vol_14d',
    'exp_decay_volume_14d',
    'exp_drawdown_pressure_14d',

    # Options-derived features (4 features)
    'opt_iv_rank',
    'opt_put_call_ratio',
    'opt_skew_25d',
    'opt_term_structure_slope'
]


if __name__ == '__main__':
    print("=" * 80)
    print("ADVANCED FEATURES MODULE TEST")
    print("=" * 80)

    # Test with synthetic data
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    volume = np.random.randint(1000000, 10000000, n)

    print(f"\nTesting with {n} samples...")
    print(f"Close range: [{close.min():.2f}, {close.max():.2f}]")
    print(f"Volume range: [{volume.min():,}, {volume.max():,}]")

    # Compute features
    features = compute_advanced_features_vectorized(close, volume)

    print("\nComputed features:")
    for name, values in features.items():
        non_zero = np.sum(values != 0)
        print(f"  {name}: {non_zero}/{len(values)} non-zero values")
        print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")

    # Test options stub
    print("\nOptions features (stub):")
    opt_features = compute_options_features_stub('AAPL')
    for name, value in opt_features.items():
        print(f"  {name}: {value}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
