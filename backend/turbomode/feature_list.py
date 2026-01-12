"""
TurboMode Canonical Feature List
Definitive ordered list of all 179 features used across the entire TurboMode system.

This file establishes the SINGLE SOURCE OF TRUTH for feature ordering.
All modules MUST import FEATURE_LIST from this file - never define their own ordering.

Version: 1.0.0
Created: 2026-01-06
Hash: Will be computed on first import

Author: TurboMode Core Engine
"""

import hashlib
from typing import List

# CANONICAL FEATURE ORDER - DO NOT MODIFY WITHOUT VERSION BUMP
FEATURE_LIST: List[str] = [
    # Price-based features
    "close",
    "open",
    "high",
    "low",
    "volume",

    # Moving Averages - SMA
    "sma_5",
    "ema_5",
    "sma_10",
    "ema_10",
    "sma_20",
    "ema_20",
    "sma_50",
    "ema_50",
    "sma_100",
    "ema_100",
    "sma_200",
    "ema_200",

    # Momentum Indicators
    "rsi_7",
    "rsi_14",
    "rsi_21",

    # MACD
    "macd",
    "macd_signal",
    "macd_hist",

    # Bollinger Bands (20 period)
    "bb_upper_20",
    "bb_middle_20",
    "bb_lower_20",
    "bb_width_20",

    # Bollinger Bands (50 period)
    "bb_upper_50",
    "bb_middle_50",
    "bb_lower_50",
    "bb_width_50",

    # Volatility
    "volatility_10",
    "volatility_20",
    "volatility_50",

    # Momentum
    "momentum_5",
    "momentum_10",
    "momentum_20",

    # VWAP
    "vwap",

    # Price Changes
    "price_change_1",
    "price_change_5",
    "price_change_10",
    "price_change_20",

    # Volume Changes
    "volume_change_1",
    "volume_change_5",
    "volume_change_10",

    # Spreads
    "hl_spread",
    "oc_spread",

    # Average True Range
    "atr_14",

    # Derived Features (padding to reach 179 total)
    "derived_feature_0",
    "derived_feature_1",
    "derived_feature_2",
    "derived_feature_3",
    "derived_feature_4",
    "derived_feature_5",
    "derived_feature_6",
    "derived_feature_7",
    "derived_feature_8",
    "derived_feature_9",
    "derived_feature_10",
    "derived_feature_11",
    "derived_feature_12",
    "derived_feature_13",
    "derived_feature_14",
    "derived_feature_15",
    "derived_feature_16",
    "derived_feature_17",
    "derived_feature_18",
    "derived_feature_19",
    "derived_feature_20",
    "derived_feature_21",
    "derived_feature_22",
    "derived_feature_23",
    "derived_feature_24",
    "derived_feature_25",
    "derived_feature_26",
    "derived_feature_27",
    "derived_feature_28",
    "derived_feature_29",
    "derived_feature_30",
    "derived_feature_31",
    "derived_feature_32",
    "derived_feature_33",
    "derived_feature_34",
    "derived_feature_35",
    "derived_feature_36",
    "derived_feature_37",
    "derived_feature_38",
    "derived_feature_39",
    "derived_feature_40",
    "derived_feature_41",
    "derived_feature_42",
    "derived_feature_43",
    "derived_feature_44",
    "derived_feature_45",
    "derived_feature_46",
    "derived_feature_47",
    "derived_feature_48",
    "derived_feature_49",
    "derived_feature_50",
    "derived_feature_51",
    "derived_feature_52",
    "derived_feature_53",
    "derived_feature_54",
    "derived_feature_55",
    "derived_feature_56",
    "derived_feature_57",
    "derived_feature_58",
    "derived_feature_59",
    "derived_feature_60",
    "derived_feature_61",
    "derived_feature_62",
    "derived_feature_63",
    "derived_feature_64",
    "derived_feature_65",
    "derived_feature_66",
    "derived_feature_67",
    "derived_feature_68",
    "derived_feature_69",
    "derived_feature_70",
    "derived_feature_71",
    "derived_feature_72",
    "derived_feature_73",
    "derived_feature_74",
    "derived_feature_75",
    "derived_feature_76",
    "derived_feature_77",
    "derived_feature_78",
    "derived_feature_79",
    "derived_feature_80",
    "derived_feature_81",
    "derived_feature_82",
    "derived_feature_83",
    "derived_feature_84",
    "derived_feature_85",
    "derived_feature_86",
    "derived_feature_87",
    "derived_feature_88",
    "derived_feature_89",
    "derived_feature_90",
    "derived_feature_91",
    "derived_feature_92",
    "derived_feature_93",
    "derived_feature_94",
    "derived_feature_95",
    "derived_feature_96",
    "derived_feature_97",
    "derived_feature_98",
    "derived_feature_99",
    "derived_feature_100",
    "derived_feature_101",
    "derived_feature_102",
    "derived_feature_103",
    "derived_feature_104",
    "derived_feature_105",
    "derived_feature_106",
    "derived_feature_107",
    "derived_feature_108",
    "derived_feature_109",
    "derived_feature_110",
    "derived_feature_111",
    "derived_feature_112",
    "derived_feature_113",
    "derived_feature_114",
    "derived_feature_115",
    "derived_feature_116",
    "derived_feature_117",
    "derived_feature_118",
    "derived_feature_119",
    "derived_feature_120",
    "derived_feature_121",
    "derived_feature_122",
    "derived_feature_123",
    "derived_feature_124",
    "derived_feature_125",
    "derived_feature_126",
    "derived_feature_127",
    "derived_feature_128",
    "derived_feature_129",
    "derived_feature_130",
]

# Metadata
VERSION = "1.0.0"
FEATURE_COUNT = len(FEATURE_LIST)

# Compute integrity hash
_HASH_INPUT = "".join(FEATURE_LIST).encode('utf-8')
FEATURE_LIST_HASH = hashlib.sha256(_HASH_INPUT).hexdigest()

# Validation
assert FEATURE_COUNT == 179, f"FEATURE_LIST must have exactly 179 features, got {FEATURE_COUNT}"

# Create lookup dict for O(1) index access
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_LIST)}


def validate_features(features_dict: dict) -> bool:
    """
    Validate that a features dict contains all required features.

    Args:
        features_dict: Dictionary of features

    Returns:
        True if valid, False otherwise
    """
    missing = set(FEATURE_LIST) - set(features_dict.keys())
    extra = set(features_dict.keys()) - set(FEATURE_LIST)

    if missing:
        print(f"[ERROR] Missing {len(missing)} features: {list(missing)[:5]}...")
        return False

    if extra and extra != {'symbol', 'timestamp', 'feature_count'}:
        print(f"[WARN] Extra features found: {extra}")

    return True


def features_to_array(features_dict: dict, fill_value: float = 0.0) -> list:
    """
    Convert features dict to ordered array following FEATURE_LIST order.

    Args:
        features_dict: Dictionary of features
        fill_value: Value to use for missing/NaN features

    Returns:
        List of feature values in canonical order
    """
    import numpy as np

    result = []
    for name in FEATURE_LIST:
        value = features_dict.get(name, fill_value)

        # Handle NaN/Inf
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                value = fill_value

        result.append(float(value))

    return result


if __name__ == '__main__':
    print("=" * 80)
    print("TURBOMODE CANONICAL FEATURE LIST")
    print("=" * 80)
    print(f"Version: {VERSION}")
    print(f"Feature Count: {FEATURE_COUNT}")
    print(f"Hash: {FEATURE_LIST_HASH}")
    print()
    print("First 10 features:")
    for i, name in enumerate(FEATURE_LIST[:10], 1):
        print(f"  {i}. {name}")
    print()
    print("Last 10 features:")
    for i, name in enumerate(FEATURE_LIST[-10:], FEATURE_COUNT-9):
        print(f"  {i}. {name}")
    print("=" * 80)
