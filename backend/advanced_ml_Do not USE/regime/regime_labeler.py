"""
Market Regime Labeler

Assigns a single regime label to each sample based on volatility and price rules.
Regimes: normal, crash, recovery, high_volatility, low_volatility

Labeling Rules:
1. Volatility-based (VIX thresholds):
   - VIX > 35 → crash
   - 25 <= VIX <= 35 → high_volatility
   - 15 <= VIX < 25 → normal
   - VIX < 15 → low_volatility

2. Price-based (overrides volatility):
   - price_drop_10d <= -15% → crash
   - price_rise_after_crash >= 10% → recovery

Priority: Price-based rules override volatility-based rules for crash/recovery.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class RegimeLabeler:
    """
    Assigns market regime labels to trading samples
    """

    def __init__(self):
        self.version = "1.0.0"
        self.regimes = ["normal", "crash", "recovery", "high_volatility", "low_volatility"]

        # VIX thresholds
        self.vix_crash_threshold = 35
        self.vix_high_vol_min = 25
        self.vix_high_vol_max = 35
        self.vix_normal_min = 15
        self.vix_normal_max = 25
        self.vix_low_vol_threshold = 15

        # Price movement thresholds
        self.crash_price_drop_pct = -15.0  # -15% in 10 days
        self.recovery_price_rise_pct = 10.0  # +10% after crash
        self.lookback_days = 10  # For price movement calculations

    def assign_regime(self, sample: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        """
        Assign regime label to a sample

        Args:
            sample: Sample dictionary with features, date, symbol
            df: Optional DataFrame with price history (for price-based rules)

        Returns:
            Regime label string
        """
        # Extract VIX from sample features
        features = sample.get('features', {})
        vix = features.get('vix', None)

        # If no VIX in features, default to normal
        if vix is None or np.isnan(vix):
            return "normal"

        # Check price-based rules first (they override volatility-based)
        price_regime = self._check_price_based_regime(sample, df)
        if price_regime:
            return price_regime

        # Apply volatility-based rules
        return self._check_volatility_based_regime(vix)

    def _check_price_based_regime(self, sample: Dict[str, Any], df: Optional[pd.DataFrame]) -> Optional[str]:
        """
        Check price-based regime rules (crash and recovery)

        Args:
            sample: Sample dictionary
            df: Price history DataFrame

        Returns:
            Regime label if price rules match, None otherwise
        """
        if df is None or len(df) < self.lookback_days:
            return None

        try:
            # Get sample date
            sample_date = sample.get('date')
            if not sample_date:
                return None

            # Convert to datetime if string
            if isinstance(sample_date, str):
                sample_date = datetime.strptime(sample_date, '%Y-%m-%d')

            # Find index of sample date in dataframe
            df_copy = df.copy()
            df_copy.index = pd.to_datetime(df_copy.index)

            # Find closest date
            idx = df_copy.index.get_indexer([sample_date], method='nearest')[0]

            if idx < self.lookback_days:
                # Not enough history
                return None

            # Get current price and price 10 days ago
            current_price = df_copy.iloc[idx]['Close']
            lookback_price = df_copy.iloc[idx - self.lookback_days]['Close']

            # Calculate price change percentage
            price_change_pct = ((current_price - lookback_price) / lookback_price) * 100

            # Check crash condition: -15% or worse in 10 days
            if price_change_pct <= self.crash_price_drop_pct:
                return "crash"

            # Check recovery condition: +10% or better after recent crash
            # Look for crash in previous 20 days
            if idx >= 20:
                recent_low_idx = df_copy.iloc[idx-20:idx]['Close'].idxmin()
                recent_low_price = df_copy.loc[recent_low_idx, 'Close']
                recovery_pct = ((current_price - recent_low_price) / recent_low_price) * 100

                if recovery_pct >= self.recovery_price_rise_pct:
                    # Verify there was actually a crash before this
                    # Check if price dropped significantly before the low
                    low_idx_pos = df_copy.index.get_loc(recent_low_idx)
                    if low_idx_pos >= self.lookback_days:
                        pre_crash_price = df_copy.iloc[low_idx_pos - self.lookback_days]['Close']
                        crash_drop = ((recent_low_price - pre_crash_price) / pre_crash_price) * 100

                        if crash_drop <= self.crash_price_drop_pct:
                            return "recovery"

        except Exception as e:
            # If price analysis fails, fall back to VIX-based
            pass

        return None

    def _check_volatility_based_regime(self, vix: float) -> str:
        """
        Check volatility-based regime rules

        Args:
            vix: VIX value

        Returns:
            Regime label based on VIX thresholds
        """
        if vix > self.vix_crash_threshold:
            return "crash"
        elif self.vix_high_vol_min <= vix <= self.vix_high_vol_max:
            return "high_volatility"
        elif self.vix_normal_min <= vix < self.vix_normal_max:
            return "normal"
        else:  # vix < 15
            return "low_volatility"

    def label_sample(self, sample: Dict[str, Any], symbol: str = None, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Add regime label to a sample

        Args:
            sample: Sample dictionary
            symbol: Stock symbol (for fetching price data if needed)
            df: Optional price DataFrame

        Returns:
            Sample with 'regime' field added
        """
        regime = self.assign_regime(sample, df)
        sample['regime'] = regime
        return sample

    def get_regime_stats(self, samples: list) -> Dict[str, Any]:
        """
        Calculate regime distribution statistics

        Args:
            samples: List of labeled samples

        Returns:
            Dictionary with regime counts and percentages
        """
        total = len(samples)
        if total == 0:
            return {}

        regime_counts = {regime: 0 for regime in self.regimes}

        for sample in samples:
            regime = sample.get('regime', 'normal')
            if regime in regime_counts:
                regime_counts[regime] += 1

        regime_stats = {}
        for regime, count in regime_counts.items():
            pct = (count / total) * 100
            regime_stats[regime] = {
                'count': count,
                'percentage': round(pct, 2)
            }

        return regime_stats


def assign_regime_label(sample: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
    """
    Convenience function to assign regime label to a single sample

    Args:
        sample: Sample dictionary with features
        df: Optional price DataFrame

    Returns:
        Regime label string
    """
    labeler = RegimeLabeler()
    return labeler.assign_regime(sample, df)


if __name__ == '__main__':
    # Test regime labeler
    print("Testing Regime Labeler...")

    labeler = RegimeLabeler()

    # Test VIX-based classification
    test_cases = [
        {'features': {'vix': 45.0}, 'date': '2020-03-20'},
        {'features': {'vix': 28.0}, 'date': '2022-06-15'},
        {'features': {'vix': 18.0}, 'date': '2023-08-10'},
        {'features': {'vix': 12.0}, 'date': '2024-01-05'},
    ]

    print("\nVIX-based regime assignment:")
    for sample in test_cases:
        regime = labeler.assign_regime(sample)
        vix = sample['features']['vix']
        print(f"  VIX={vix:5.1f} -> {regime}")

    # Test regime stats
    labeled_samples = [labeler.label_sample(s) for s in test_cases]
    stats = labeler.get_regime_stats(labeled_samples)

    print("\nRegime Distribution:")
    for regime, data in stats.items():
        if data['count'] > 0:
            print(f"  {regime:18s} {data['count']:3d} ({data['percentage']:5.1f}%)")
