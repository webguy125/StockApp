"""
Timeframe Aggregation Module
Aggregates 1-minute OHLCV data to any timeframe using pandas resample
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Timeframe mapping from string to pandas offset
TIMEFRAME_OFFSETS = {
    '1min': '1T',
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1h': '1H',
    '2h': '2H',
    '4h': '4H',
    '6h': '6H',
    '1d': '1D',
    'daily': '1D',
    '1w': '1W',
    'weekly': '1W',
    '1mo': '1M',
    'monthly': '1M',
    '3mo': '3M'
}

# Timeframe integer encoding for ML features
TIMEFRAME_ENCODING = {
    '1min': 1,
    '5min': 2,
    '15min': 3,
    '30min': 4,
    '1h': 5,
    '2h': 6,
    '4h': 7,
    '6h': 8,
    '1d': 9,
    'daily': 9,
    '1w': 10,
    'weekly': 10,
    '1mo': 11,
    'monthly': 11,
    '3mo': 12
}


def aggregate_ohlcv(df, timeframe):
    """
    Aggregate OHLCV data to specified timeframe

    Args:
        df: pandas DataFrame with timestamp index and OHLCV columns
        timeframe: Target timeframe ('1min', '5min', '1h', '1d', etc.)

    Returns:
        pandas DataFrame with aggregated data
    """
    try:
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided for aggregation")
            return df

        # Get pandas offset string
        offset = TIMEFRAME_OFFSETS.get(timeframe)
        if not offset:
            logger.error(f"❌ Invalid timeframe: {timeframe}")
            return df

        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Define aggregation rules
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Add indicator columns if present (use mean for aggregation)
        indicator_cols = [
            'triad_weighted_trend',
            'triad_oscillator',
            'triad_short_trend',
            'triad_pivot_score'
        ]

        for col in indicator_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'

        # Handle boolean pivot columns specially
        if 'triad_pivot_high' in df.columns:
            agg_dict['triad_pivot_high'] = 'max'  # Any pivot in period
        if 'triad_pivot_low' in df.columns:
            agg_dict['triad_pivot_low'] = 'max'

        # Perform resampling
        df_agg = df.resample(offset).agg(agg_dict)

        # Remove rows with NaN in OHLC (incomplete candles)
        df_agg.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        # Fill volume NaNs with 0
        if 'volume' in df_agg.columns:
            df_agg['volume'].fillna(0, inplace=True)

        # Convert to float32 for memory efficiency
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_agg.columns:
                df_agg[col] = df_agg[col].astype(np.float32)

        logger.info(f"✅ Aggregated {len(df)} bars to {len(df_agg)} bars ({timeframe})")
        return df_agg

    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}")
        return df


def aggregate_from_database(db, asset, timeframe, start_date=None, end_date=None):
    """
    Fetch data from database and aggregate to specified timeframe

    Args:
        db: OHLCVDatabase instance
        asset: Asset identifier
        timeframe: Target timeframe
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        pandas DataFrame with aggregated data
    """
    try:
        # Fetch 1-minute data from database
        df = db.get_ohlcv(asset, start_date=start_date, end_date=end_date)

        if df.empty:
            logger.warning(f"⚠️ No data found for {asset}")
            return df

        # Aggregate to target timeframe
        df_agg = aggregate_ohlcv(df, timeframe)

        return df_agg

    except Exception as e:
        logger.error(f"❌ Database aggregation failed: {e}")
        return pd.DataFrame()


def get_timeframe_encoding(timeframe):
    """
    Get integer encoding for timeframe (for ML features)

    Args:
        timeframe: Timeframe string

    Returns:
        int: Encoded timeframe (1-12)
    """
    return TIMEFRAME_ENCODING.get(timeframe, 1)


def resample_indicators(df, timeframe):
    """
    Resample indicator outputs to a different timeframe

    This is useful when you have indicator values at one timeframe
    and need them at another for analysis or comparison.

    Args:
        df: DataFrame with indicator columns
        timeframe: Target timeframe

    Returns:
        Resampled DataFrame
    """
    try:
        if df.empty:
            return df

        offset = TIMEFRAME_OFFSETS.get(timeframe)
        if not offset:
            return df

        # Ensure datetime index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Define aggregation for indicator columns
        agg_dict = {}

        # Numeric indicators - use mean
        numeric_cols = [
            'triad_weighted_trend',
            'triad_oscillator',
            'triad_short_trend',
            'triad_pivot_score'
        ]

        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'

        # Boolean indicators - use max (any True becomes True)
        bool_cols = ['triad_pivot_high', 'triad_pivot_low']
        for col in bool_cols:
            if col in df.columns:
                agg_dict[col] = 'max'

        # Resample
        df_resampled = df.resample(offset).agg(agg_dict)

        # Drop incomplete periods
        df_resampled.dropna(how='all', inplace=True)

        return df_resampled

    except Exception as e:
        logger.error(f"❌ Indicator resampling failed: {e}")
        return df


def align_timeframes(df_base, df_indicators, timeframe):
    """
    Align OHLCV data with indicator data at a specific timeframe

    Useful when merging price data with pre-computed indicators
    from different sources or calculations.

    Args:
        df_base: Base OHLCV DataFrame
        df_indicators: DataFrame with indicator columns
        timeframe: Timeframe to align to

    Returns:
        Merged DataFrame with aligned timestamps
    """
    try:
        # Ensure both are at same timeframe
        df_base_agg = aggregate_ohlcv(df_base, timeframe)
        df_ind_agg = resample_indicators(df_indicators, timeframe)

        # Merge on timestamp index
        df_merged = df_base_agg.join(df_ind_agg, how='left')

        return df_merged

    except Exception as e:
        logger.error(f"❌ Timeframe alignment failed: {e}")
        return df_base
