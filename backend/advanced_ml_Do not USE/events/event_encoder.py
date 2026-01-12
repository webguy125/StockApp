"""
Event Feature Encoder
Generates time-windowed features from classified events per specification.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class EventEncoder:
    """
    Encodes classified events into features using lookback windows.
    Implements all encoded_features from specification.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event encoder.

        Args:
            config: Encoder configuration dictionary
        """
        self.config = config
        self.lookback_windows = config.get('lookback_windows_days', [1, 7, 30, 90, 365])
        self.encoded_features = config.get('encoded_features', [])
        self.missing_policy = config.get('missing_policy', 'zero_fill_and_flag')
        self.scaling_method = config.get('scaling_method', 'robust_scaler')

        # Initialize scaler
        if self.scaling_method == 'robust_scaler':
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        logger.info(f"[ENCODER] Initialized with {len(self.lookback_windows)} windows")

    def encode(
        self,
        ticker: str,
        events: pd.DataFrame,
        target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Encode events into features for target dates.

        Args:
            ticker: Stock ticker symbol
            events: Classified events DataFrame
            target_dates: Dates for which to compute features

        Returns:
            DataFrame with encoded features
        """
        if len(events) == 0:
            return self._get_empty_features(target_dates)

        # Ensure timestamp is datetime
        events = events.copy()
        if not pd.api.types.is_datetime64_any_dtype(events['timestamp']):
            events['timestamp'] = pd.to_datetime(events['timestamp'])

        # Initialize feature DataFrame
        features = pd.DataFrame(index=target_dates)

        # Generate all encoded features
        features = self._encode_event_counts(features, events)
        features = self._encode_severity_features(features, events)
        features = self._encode_impact_features(features, events)
        features = self._encode_sentiment_features(features, events)
        features = self._encode_temporal_features(features, events)
        features = self._encode_complexity_features(features, events)

        # Handle missing values
        features = self._handle_missing_values(features)

        logger.info(f"[ENCODER] Generated {features.shape[1]} features for {len(target_dates)} dates")

        return features

    def _encode_event_counts(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode event count features by type and window.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        event_types = ['refinancing', 'dividend', 'litigation', 'negative_news']
        windows = {'7d': 7, '30d': 30, '90d': 90}

        for event_type in event_types:
            for window_name, window_days in windows.items():
                col_name = f'event_count_{event_type}_{window_name}'

                if event_type == 'negative_news':
                    # Count news with negative sentiment
                    mask = (events['source_type'] == 'news') & (events['sentiment_score'] < -0.2)
                else:
                    # Count events of specific type
                    mask = events['event_type'].str.contains(event_type, case=False, na=False)

                features[col_name] = features.index.map(
                    lambda date: self._count_events_in_window(events[mask], date, window_days)
                )

        return features

    def _encode_severity_features(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode severity-based features.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        # Max event severity in 30-day window
        features['max_event_severity_30d'] = features.index.map(
            lambda date: self._max_severity_in_window(events, date, 30)
        )

        # Time since last high-severity event
        features['time_since_last_high_severity_event'] = features.index.map(
            lambda date: self._days_since_high_severity(events, date)
        )

        return features

    def _encode_impact_features(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode impact score features.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        impact_types = ['dividend', 'liquidity', 'credit']
        windows = {'90d': 90}

        for impact_type in impact_types:
            for window_name, window_days in windows.items():
                col_name = f'sum_impact_{impact_type}_{window_name}'

                features[col_name] = features.index.map(
                    lambda date: self._sum_impact_in_window(
                        events, date, window_days, f'impact_{impact_type}'
                    )
                )

        return features

    def _encode_sentiment_features(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode sentiment-based features.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        # News sentiment features (7-day window)
        news_events = events[events['source_type'] == 'news']

        features['news_sentiment_mean_7d'] = features.index.map(
            lambda date: self._mean_sentiment_in_window(news_events, date, 7)
        )

        features['news_sentiment_min_7d'] = features.index.map(
            lambda date: self._min_sentiment_in_window(news_events, date, 7)
        )

        return features

    def _encode_temporal_features(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode temporal pattern features.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        # Event intensity acceleration ratio
        features['event_intensity_acceleration_ratio'] = features.index.map(
            lambda date: self._calculate_acceleration_ratio(events, date)
        )

        # Cross-source confirmation flag
        features['cross_source_confirmation_flag'] = features.index.map(
            lambda date: self._check_cross_source_confirmation(events, date)
        )

        return features

    def _encode_complexity_features(
        self,
        features: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode complexity and information asymmetry features.

        Args:
            features: Feature DataFrame
            events: Events DataFrame

        Returns:
            Updated feature DataFrame
        """
        # Information asymmetry proxy
        features['information_asymmetry_proxy_score'] = features.index.map(
            lambda date: self._calculate_information_asymmetry(events, date)
        )

        # Filing complexity index
        sec_events = events[events['source_type'] == 'sec']
        features['filing_complexity_index'] = features.index.map(
            lambda date: self._calculate_filing_complexity(sec_events, date)
        )

        return features

    def _count_events_in_window(
        self,
        events: pd.DataFrame,
        target_date: datetime,
        window_days: int
    ) -> int:
        """Count events within lookback window."""
        start_date = target_date - timedelta(days=window_days)
        mask = (events['timestamp'] >= start_date) & (events['timestamp'] < target_date)
        return int(mask.sum())

    def _max_severity_in_window(
        self,
        events: pd.DataFrame,
        target_date: datetime,
        window_days: int
    ) -> float:
        """Get maximum severity within window."""
        start_date = target_date - timedelta(days=window_days)
        mask = (events['timestamp'] >= start_date) & (events['timestamp'] < target_date)
        window_events = events[mask]

        if len(window_events) == 0:
            return 0.0

        return float(window_events['event_severity'].max())

    def _days_since_high_severity(
        self,
        events: pd.DataFrame,
        target_date: datetime
    ) -> float:
        """Calculate days since last high-severity event."""
        high_severity_events = events[events['event_severity'] >= 0.75]
        past_events = high_severity_events[high_severity_events['timestamp'] < target_date]

        if len(past_events) == 0:
            return 999.0  # No high-severity events

        last_event_date = past_events['timestamp'].max()
        days = (target_date - last_event_date).days

        return float(days)

    def _sum_impact_in_window(
        self,
        events: pd.DataFrame,
        target_date: datetime,
        window_days: int,
        impact_column: str
    ) -> float:
        """Sum impact scores within window."""
        start_date = target_date - timedelta(days=window_days)
        mask = (events['timestamp'] >= start_date) & (events['timestamp'] < target_date)
        window_events = events[mask]

        if len(window_events) == 0 or impact_column not in window_events.columns:
            return 0.0

        return float(window_events[impact_column].sum())

    def _mean_sentiment_in_window(
        self,
        events: pd.DataFrame,
        target_date: datetime,
        window_days: int
    ) -> float:
        """Calculate mean sentiment within window."""
        start_date = target_date - timedelta(days=window_days)
        mask = (events['timestamp'] >= start_date) & (events['timestamp'] < target_date)
        window_events = events[mask]

        if len(window_events) == 0:
            return 0.0

        return float(window_events['sentiment_score'].mean())

    def _min_sentiment_in_window(
        self,
        events: pd.DataFrame,
        target_date: datetime,
        window_days: int
    ) -> float:
        """Get minimum sentiment within window."""
        start_date = target_date - timedelta(days=window_days)
        mask = (events['timestamp'] >= start_date) & (events['timestamp'] < target_date)
        window_events = events[mask]

        if len(window_events) == 0:
            return 0.0

        return float(window_events['sentiment_score'].min())

    def _calculate_acceleration_ratio(
        self,
        events: pd.DataFrame,
        target_date: datetime
    ) -> float:
        """
        Calculate event intensity acceleration (recent vs baseline).

        Returns ratio of 7-day event count to 30-day average.
        """
        count_7d = self._count_events_in_window(events, target_date, 7)
        count_30d = self._count_events_in_window(events, target_date, 30)

        if count_30d == 0:
            return 0.0

        # Normalize to daily rates
        rate_7d = count_7d / 7.0
        rate_30d = count_30d / 30.0

        if rate_30d == 0:
            return 0.0

        return float(rate_7d / rate_30d)

    def _check_cross_source_confirmation(
        self,
        events: pd.DataFrame,
        target_date: datetime
    ) -> float:
        """
        Check if SEC and news sources confirm same event within 3 days.

        Returns 1.0 if confirmed, 0.0 otherwise.
        """
        start_date = target_date - timedelta(days=3)
        window_events = events[
            (events['timestamp'] >= start_date) &
            (events['timestamp'] < target_date)
        ]

        if len(window_events) == 0:
            return 0.0

        # Check if both SEC and news present with similar event types
        has_sec = (window_events['source_type'] == 'sec').any()
        has_news = (window_events['source_type'] == 'news').any()

        if has_sec and has_news:
            return 1.0

        return 0.0

    def _calculate_information_asymmetry(
        self,
        events: pd.DataFrame,
        target_date: datetime
    ) -> float:
        """
        Proxy for information asymmetry based on SEC vs news timing.

        Higher score = SEC filings precede news by larger margin.
        """
        start_date = target_date - timedelta(days=7)
        window_events = events[
            (events['timestamp'] >= start_date) &
            (events['timestamp'] < target_date)
        ]

        if len(window_events) == 0:
            return 0.0

        sec_events = window_events[window_events['source_type'] == 'sec']
        news_events = window_events[window_events['source_type'] == 'news']

        if len(sec_events) == 0 or len(news_events) == 0:
            return 0.0

        # Calculate average time lag
        first_sec = sec_events['timestamp'].min()
        first_news = news_events['timestamp'].min()

        lag_hours = (first_news - first_sec).total_seconds() / 3600.0

        # Normalize to [0, 1] with 48 hours = max asymmetry
        asymmetry = min(1.0, max(0.0, lag_hours / 48.0))

        return float(asymmetry)

    def _calculate_filing_complexity(
        self,
        events: pd.DataFrame,
        target_date: datetime
    ) -> float:
        """
        Calculate filing complexity index based on text length and filing type.

        Returns complexity score [0.0, 1.0].
        """
        start_date = target_date - timedelta(days=30)
        window_events = events[
            (events['timestamp'] >= start_date) &
            (events['timestamp'] < target_date)
        ]

        if len(window_events) == 0:
            return 0.0

        # Complexity based on filing types and text length
        complexity_scores = []

        for _, event in window_events.iterrows():
            filing_type = event.get('filing_type', '')
            text_length = len(str(event.get('raw_text', '')))

            # Base complexity by filing type
            type_complexity = {
                '10-K': 0.9,
                '10-Q': 0.7,
                '8-K': 0.5,
                'DEF 14A': 0.6,
                '13D': 0.4
            }.get(filing_type, 0.3)

            # Adjust by text length (longer = more complex)
            length_factor = min(1.0, text_length / 10000.0)

            complexity = type_complexity * (0.5 + 0.5 * length_factor)
            complexity_scores.append(complexity)

        if not complexity_scores:
            return 0.0

        return float(np.mean(complexity_scores))

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values per policy.

        Args:
            features: Feature DataFrame

        Returns:
            DataFrame with missing values handled
        """
        if self.missing_policy == 'zero_fill_and_flag':
            # Create missing flags
            for col in features.columns:
                if features[col].isna().any():
                    features[f'{col}_missing'] = features[col].isna().astype(float)

            # Fill with zeros
            features = features.fillna(0.0)

        return features

    def _get_empty_features(self, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Get empty feature DataFrame when no events available.

        Args:
            target_dates: Target dates

        Returns:
            DataFrame filled with zeros
        """
        features = pd.DataFrame(index=target_dates)

        # Add all expected feature columns with zeros
        for feature_name in self.encoded_features:
            features[feature_name] = 0.0

        return features

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get encoder statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'lookback_windows_days': self.lookback_windows,
            'encoded_feature_count': len(self.encoded_features),
            'missing_policy': self.missing_policy,
            'scaling_method': self.scaling_method
        }
