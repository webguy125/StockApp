"""
Pattern Service Module
Chart pattern detection and volume profile calculations
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def detect_double_top(df):
    """Detect double top patterns"""
    patterns = []

    # Find local maxima
    high_order = max(5, len(df) // 50)
    local_max = argrelextrema(df['High'].values, np.greater, order=high_order)[0]

    if len(local_max) >= 2:
        for i in range(len(local_max) - 1):
            idx1 = local_max[i]
            idx2 = local_max[i + 1]
            price1 = df.iloc[idx1]['High']
            price2 = df.iloc[idx2]['High']

            # Check if peaks are similar (within 2%)
            if abs(price1 - price2) / price1 < 0.02:
                # Find valley between peaks
                valley_idx = df.iloc[idx1:idx2]['Low'].idxmin()
                valley_price = df.iloc[valley_idx]['Low']

                # Confirm significant valley (at least 3% drop)
                if (price1 - valley_price) / price1 > 0.03:
                    patterns.append({
                        'type': 'Double Top',
                        'confidence': 0.75,
                        'date1': df.iloc[idx1]['Date'].strftime('%Y-%m-%d'),
                        'date2': df.iloc[idx2]['Date'].strftime('%Y-%m-%d'),
                        'price': float(price1),
                        'support': float(valley_price),
                        'resistance': float(price1),
                        'description': f'Double top at ${price1:.2f}'
                    })

    return patterns


def detect_double_bottom(df):
    """Detect double bottom patterns"""
    patterns = []

    # Find local minima
    low_order = max(5, len(df) // 50)
    local_min = argrelextrema(df['Low'].values, np.less, order=low_order)[0]

    if len(local_min) >= 2:
        for i in range(len(local_min) - 1):
            idx1 = local_min[i]
            idx2 = local_min[i + 1]
            price1 = df.iloc[idx1]['Low']
            price2 = df.iloc[idx2]['Low']

            # Check if bottoms are similar (within 2%)
            if abs(price1 - price2) / price1 < 0.02:
                # Find peak between bottoms
                peak_idx = df.iloc[idx1:idx2]['High'].idxmax()
                peak_price = df.iloc[peak_idx]['High']

                # Confirm significant peak (at least 3% rise)
                if (peak_price - price1) / price1 > 0.03:
                    patterns.append({
                        'type': 'Double Bottom',
                        'confidence': 0.75,
                        'date1': df.iloc[idx1]['Date'].strftime('%Y-%m-%d'),
                        'date2': df.iloc[idx2]['Date'].strftime('%Y-%m-%d'),
                        'price': float(price1),
                        'support': float(price1),
                        'resistance': float(peak_price),
                        'description': f'Double bottom at ${price1:.2f}'
                    })

    return patterns


def detect_head_shoulders(df):
    """Detect head and shoulders patterns"""
    patterns = []

    # Find local maxima
    high_order = max(5, len(df) // 50)
    local_max = argrelextrema(df['High'].values, np.greater, order=high_order)[0]

    if len(local_max) >= 3:
        for i in range(len(local_max) - 2):
            left_idx = local_max[i]
            head_idx = local_max[i + 1]
            right_idx = local_max[i + 2]

            left_price = df.iloc[left_idx]['High']
            head_price = df.iloc[head_idx]['High']
            right_price = df.iloc[right_idx]['High']

            # Check if middle peak is highest and shoulders are similar
            if (head_price > left_price and head_price > right_price and
                abs(left_price - right_price) / left_price < 0.03):

                # Find neckline (lowest low between left and right)
                neckline_idx = df.iloc[left_idx:right_idx]['Low'].idxmin()
                neckline = df.iloc[neckline_idx]['Low']

                patterns.append({
                    'type': 'Head and Shoulders',
                    'confidence': 0.80,
                    'date1': df.iloc[left_idx]['Date'].strftime('%Y-%m-%d'),
                    'date2': df.iloc[head_idx]['Date'].strftime('%Y-%m-%d'),
                    'date3': df.iloc[right_idx]['Date'].strftime('%Y-%m-%d'),
                    'head_price': float(head_price),
                    'shoulder_price': float((left_price + right_price) / 2),
                    'neckline': float(neckline),
                    'description': f'Head & Shoulders with neckline at ${neckline:.2f}'
                })

    return patterns


def calculate_volume_profile(df, bins=50):
    """Calculate volume profile (volume by price level)"""
    # Calculate typical price for each period
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Create price bins
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)

    # Assign volume to price levels
    volume_by_price = np.zeros(bins)

    for idx, row in df.iterrows():
        # Find which bin this candle's typical price falls into
        bin_idx = np.digitize(row['TP'], price_bins) - 1
        bin_idx = max(0, min(bin_idx, bins - 1))
        volume_by_price[bin_idx] += row['Volume']

    # Find Point of Control (POC) - price level with highest volume
    poc_idx = np.argmax(volume_by_price)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

    # Calculate Value Area (70% of volume)
    total_volume = volume_by_price.sum()
    target_volume = total_volume * 0.70

    # Start from POC and expand outward
    va_volume = volume_by_price[poc_idx]
    va_low_idx = poc_idx
    va_high_idx = poc_idx

    while va_volume < target_volume and (va_low_idx > 0 or va_high_idx < bins - 1):
        low_vol = volume_by_price[va_low_idx - 1] if va_low_idx > 0 else 0
        high_vol = volume_by_price[va_high_idx + 1] if va_high_idx < bins - 1 else 0

        if low_vol > high_vol and va_low_idx > 0:
            va_low_idx -= 1
            va_volume += low_vol
        elif va_high_idx < bins - 1:
            va_high_idx += 1
            va_volume += high_vol
        else:
            break

    va_low = price_bins[va_low_idx]
    va_high = price_bins[va_high_idx + 1]

    # Prepare profile data
    profile_data = []
    for i in range(bins):
        profile_data.append({
            'price': float((price_bins[i] + price_bins[i + 1]) / 2),
            'volume': float(volume_by_price[i]),
            'price_low': float(price_bins[i]),
            'price_high': float(price_bins[i + 1])
        })

    return {
        'profile': profile_data,
        'poc': float(poc_price),
        'value_area_high': float(va_high),
        'value_area_low': float(va_low),
        'total_volume': float(total_volume)
    }
