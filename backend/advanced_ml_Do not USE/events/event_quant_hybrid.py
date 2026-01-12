"""
Event Quant Hybrid Module v1.1.0
Unified SEC + news event intelligence for supervised 8-model ensemble trading system.
Enhanced with semantic cross-referencing, multi-modal signal fusion, and drift detection.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Module configuration
MODULE_VERSION = "1.1.0"
MODULE_NAME = "event_quant_hybrid"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventQuantHybrid:
    """
    Unified event intelligence system integrating SEC filings and news sources.
    Deterministic implementation per specification.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize event quant hybrid system.

        Args:
            config_path: Path to configuration JSON (uses default if None)
        """
        self.version = MODULE_VERSION
        self.name = MODULE_NAME

        # Load configuration
        if config_path is None:
            config_path = self._get_default_config_path()

        self.config = self._load_config(config_path)

        # Initialize components
        self.ingestion = None
        self.classifier = None
        self.encoder = None
        self.archive = None

        # State tracking
        self.last_update = None
        self.event_cache = {}

        logger.info(f"[EVENT_QUANT_HYBRID] Initialized v{self.version}")

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        base_dir = Path(__file__).parent
        return str(base_dir / "config" / "event_quant_hybrid_config.json")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration JSON

        Returns:
            Configuration dictionary
        """
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"[CONFIG] Loaded from {config_path}")
            return config
        else:
            logger.warning(f"[CONFIG] File not found: {config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration matching JSON specification."""
        return {
            "module_name": "event_quant_hybrid",
            "version": "1.1.0",
            "ingestion": {
                "sec": {
                    "enabled": True,
                    "filing_types": ["8-K", "10-Q", "10-K", "13D", "13G", "13F", "S-1", "S-3", "424B", "DEF 14A"],
                    "max_lag_minutes": 15
                },
                "news": {
                    "enabled": True,
                    "sources": ["dow_jones", "reuters", "company_wires", "bloomberg_terminal_feed", "nasdaq_disclosures"],
                    "languages": ["en"],
                    "dedupe_window_minutes": 10
                }
            },
            "event_classifier": {
                "model_id": "event_cls_v1_distilbert_ensemble",
                "thresholds": {
                    "min_confidence": 0.60,
                    "high_severity_cutoff": 0.75,
                    "ambiguity_flag_threshold": 0.20
                }
            },
            "feature_encoder": {
                "lookback_windows_days": [1, 7, 30, 90, 365],
                "missing_policy": "zero_fill_and_flag",
                "scaling_method": "robust_scaler"
            },
            "rare_event_archive": {
                "enabled": True,
                "severity_threshold": 0.80,
                "post_event_window_days": 60
            },
            "shap_logging": {
                "enabled": True,
                "log_path": "C:/NeuralNet/ShapLogs",
                "store_top_features": 20,
                "store_event_vs_quant_attribution": True
            }
        }

    def initialize_components(self):
        """Initialize all module components."""
        from .event_ingestion import EventIngestion
        from .event_classifier import EventClassifier
        from .event_encoder import EventEncoder
        from .event_archive import EventArchive

        # Initialize ingestion
        self.ingestion = EventIngestion(self.config['ingestion'])
        logger.info("[COMPONENTS] Event ingestion initialized")

        # Initialize classifier
        self.classifier = EventClassifier(self.config['event_classifier'])
        logger.info("[COMPONENTS] Event classifier initialized")

        # Initialize encoder
        self.encoder = EventEncoder(self.config['feature_encoder'])
        logger.info("[COMPONENTS] Feature encoder initialized")

        # Initialize archive
        if self.config['rare_event_archive']['enabled']:
            self.archive = EventArchive(self.config['rare_event_archive'])
            logger.info("[COMPONENTS] Rare event archive initialized")

    def ingest_events(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ingest events from SEC and news sources.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for event retrieval
            end_date: End date for event retrieval
            sources: Optional list of sources ['sec', 'news']

        Returns:
            DataFrame with normalized event schema
        """
        if self.ingestion is None:
            self.initialize_components()

        if sources is None:
            sources = []
            if self.config['ingestion']['sec']['enabled']:
                sources.append('sec')
            if self.config['ingestion']['news']['enabled']:
                sources.append('news')

        events = self.ingestion.fetch_events(ticker, start_date, end_date, sources)

        logger.info(f"[INGESTION] Retrieved {len(events)} events for {ticker}")
        return events

    def classify_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Classify events into types with severity and impact scores.

        Args:
            events: DataFrame of raw events

        Returns:
            DataFrame with classification outputs
        """
        if self.classifier is None:
            self.initialize_components()

        classified = self.classifier.classify(events)

        logger.info(f"[CLASSIFICATION] Classified {len(classified)} events")
        return classified

    def encode_features(
        self,
        ticker: str,
        events: pd.DataFrame,
        target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Encode event features for target dates using lookback windows.

        Args:
            ticker: Stock ticker symbol
            events: Classified events DataFrame
            target_dates: Dates for which to compute features

        Returns:
            DataFrame with encoded event features
        """
        if self.encoder is None:
            self.initialize_components()

        features = self.encoder.encode(ticker, events, target_dates)

        logger.info(f"[ENCODING] Generated {features.shape[1]} features for {len(target_dates)} dates")
        return features

    def archive_rare_events(self, events: pd.DataFrame) -> int:
        """
        Archive high-severity events for future training.

        Args:
            events: Classified events with severity scores

        Returns:
            Number of events archived
        """
        if self.archive is None:
            if self.config['rare_event_archive']['enabled']:
                from .event_archive import EventArchive
                self.archive = EventArchive(self.config['rare_event_archive'])
            else:
                logger.warning("[ARCHIVE] Rare event archive is disabled")
                return 0

        archived_count = self.archive.store_events(events)

        logger.info(f"[ARCHIVE] Stored {archived_count} rare events")
        return archived_count

    def get_event_features_for_ensemble(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        target_dates: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        End-to-end pipeline: ingest → classify → encode event features.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for event retrieval
            end_date: End date for event retrieval
            target_dates: Specific dates for feature computation

        Returns:
            DataFrame with event features ready for ensemble models
        """
        # Step 1: Ingest events
        events = self.ingest_events(ticker, start_date, end_date)

        if len(events) == 0:
            logger.warning(f"[PIPELINE] No events found for {ticker}")
            if target_dates is not None:
                return pd.DataFrame(index=target_dates)
            else:
                return pd.DataFrame()

        # Step 2: Classify events
        classified_events = self.classify_events(events)

        # Step 3: Archive rare events
        if self.config['rare_event_archive']['enabled']:
            self.archive_rare_events(classified_events)

        # Step 4: Encode features
        if target_dates is None:
            target_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        features = self.encode_features(ticker, classified_events, target_dates)

        logger.info(f"[PIPELINE] Complete: {features.shape[0]} samples, {features.shape[1]} features")
        return features

    def integrate_with_ensemble_features(
        self,
        event_features: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
        sector_features: Optional[pd.DataFrame] = None,
        drift_features: Optional[pd.DataFrame] = None,
        price_volume_features: Optional[pd.DataFrame] = None,
        volatility_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge event features with other feature sets for ensemble models.

        Args:
            event_features: Event-derived features
            regime_features: Regime classification features
            sector_features: Sector-based features
            drift_features: Distribution drift features
            price_volume_features: Price/volume technical features
            volatility_features: Volatility-based features

        Returns:
            Combined feature DataFrame
        """
        combined = event_features.copy()

        # Merge each feature set
        feature_sets = {
            'regime': regime_features,
            'sector': sector_features,
            'drift': drift_features,
            'price_volume': price_volume_features,
            'volatility': volatility_features
        }

        for name, features in feature_sets.items():
            if features is not None:
                combined = combined.join(features, how='left')
                logger.info(f"[INTEGRATION] Merged {name} features: {features.shape[1]} columns")

        logger.info(f"[INTEGRATION] Final feature count: {combined.shape[1]}")
        return combined

    def get_shap_log_path(self) -> str:
        """Get SHAP logging directory path."""
        if self.config['shap_logging']['enabled']:
            return self.config['shap_logging']['log_path']
        else:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get module statistics and status.

        Returns:
            Dictionary with module statistics
        """
        stats = {
            'module_name': self.name,
            'version': self.version,
            'last_update': self.last_update,
            'components_initialized': {
                'ingestion': self.ingestion is not None,
                'classifier': self.classifier is not None,
                'encoder': self.encoder is not None,
                'archive': self.archive is not None
            },
            'config': {
                'sec_enabled': self.config['ingestion']['sec']['enabled'],
                'news_enabled': self.config['ingestion']['news']['enabled'],
                'archive_enabled': self.config['rare_event_archive']['enabled'],
                'shap_logging_enabled': self.config['shap_logging']['enabled']
            }
        }

        if self.archive is not None:
            stats['archive_stats'] = self.archive.get_statistics()

        return stats


def create_default_config():
    """Create default configuration file."""
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "event_quant_hybrid_config.json"

    default_config = {
        "module_name": "event_quant_hybrid",
        "version": "1.1.0",
        "description": "Unified SEC + news event intelligence module for the supervised 8-model ensemble trading system",
        "ingestion": {
            "sec": {
                "enabled": True,
                "filing_types": [
                    "8-K", "10-Q", "10-K", "13D", "13G", "13F",
                    "S-1", "S-3", "424B", "DEF 14A"
                ],
                "max_lag_minutes": 15,
                "normalize_schema": [
                    "source_type", "ticker", "timestamp", "filing_type",
                    "raw_text", "metadata", "accession_number",
                    "item_codes", "is_amendment"
                ],
                "processing_enhancements": {
                    "table_extraction": "enabled",
                    "exhibit_parsing": ["99.1", "10.1"],
                    "redline_comparison": "vs_previous_filing"
                }
            },
            "news": {
                "enabled": True,
                "sources": [
                    "dow_jones", "reuters", "company_wires",
                    "bloomberg_terminal_feed", "nasdaq_disclosures"
                ],
                "languages": ["en"],
                "dedupe_window_minutes": 10,
                "normalize_schema": [
                    "source_type", "ticker", "timestamp", "headline",
                    "raw_text", "metadata", "urgency_level",
                    "entity_relevance_score"
                ],
                "noise_filtering": {
                    "exclude_clickbait": True,
                    "min_article_length_chars": 150,
                    "blacklist_keywords": ["sponsored", "stock_picks"]
                }
            }
        },
        "event_classifier": {
            "model_id": "event_cls_v1_distilbert_ensemble",
            "label_space": {
                "event_type": [
                    "refinancing", "dividend_change", "earnings", "guidance",
                    "litigation", "credit_facility", "covenant_change",
                    "management_change", "regulatory", "macro", "m_and_a",
                    "stock_buyback", "product_recall",
                    "cybersecurity_incident", "other"
                ]
            },
            "output_schema": [
                "event_type", "event_subtype", "event_severity",
                "impact_dividend", "impact_liquidity", "impact_credit",
                "impact_growth", "is_rare_event", "confidence",
                "sentiment_score", "temporal_relevance"
            ],
            "thresholds": {
                "min_confidence": 0.60,
                "high_severity_cutoff": 0.75,
                "ambiguity_flag_threshold": 0.20
            },
            "alternative_approaches": [
                "zero_shot_llm_verification",
                "semantic_similarity_clustering"
            ]
        },
        "feature_encoder": {
            "lookback_windows_days": [1, 7, 30, 90, 365],
            "encoded_features": [
                "event_count_refinancing_30d",
                "event_count_dividend_30d",
                "event_count_litigation_30d",
                "event_count_negative_news_7d",
                "max_event_severity_30d",
                "sum_impact_dividend_90d",
                "sum_impact_liquidity_90d",
                "sum_impact_credit_90d",
                "time_since_last_high_severity_event",
                "news_sentiment_mean_7d",
                "news_sentiment_min_7d",
                "event_intensity_acceleration_ratio",
                "cross_source_confirmation_flag",
                "information_asymmetry_proxy_score",
                "filing_complexity_index"
            ],
            "missing_policy": "zero_fill_and_flag",
            "scaling_method": "robust_scaler"
        },
        "integration": {
            "join_key": ["ticker", "date", "timestamp_utc"],
            "merge_with": [
                "regime_features", "sector_features", "drift_features",
                "price_volume_features", "volatility_features",
                "rare_event_archive_features", "analyst_estimate_features",
                "options_implied_skew"
            ],
            "downstream_targets": [
                "ensemble_alpha_model", "risk_model",
                "dividend_safety_model", "execution_algo_router"
            ],
            "collision_logic": "timestamp_precedence"
        },
        "ensemble_interface": {
            "canonical_feature_vector": "event_features + quant_features + regime_features + sector_features + drift_features + sentiment_context",
            "model_outputs": [
                "score_buy", "score_hold", "score_reduce", "risk_score",
                "uncertainty", "expected_tracking_error",
                "liquidity_adjusted_alpha"
            ],
            "optimization_targets": [
                "sharpe_ratio", "max_drawdown_minimization"
            ]
        },
        "shap_logging": {
            "enabled": True,
            "log_path": "C:/NeuralNet/ShapLogs",
            "store_top_features": 20,
            "store_event_vs_quant_attribution": True,
            "periodic_summary_days": 7,
            "anomaly_detection": {
                "alert_on_top_feature_flip": True,
                "shap_value_drift_threshold": 0.15
            }
        },
        "rare_event_archive": {
            "enabled": True,
            "severity_threshold": 0.80,
            "post_event_window_days": 60,
            "stored_fields": [
                "event_type", "event_severity", "impact_credit",
                "impact_liquidity", "pre_post_returns",
                "drawdown_outcomes", "recovery_time_days",
                "volatility_spike_magnitude"
            ],
            "retrieval_mechanism": "k_nearest_neighbors_on_embeddings"
        },
        "promotion_gate": {
            "shadow_mode_enabled": True,
            "compare_metrics": [
                "calibration", "drift", "decision_stability",
                "feature_importance_shift", "precision_recall_auc",
                "information_coefficient"
            ],
            "promotion_condition": "no_regime_mismatch AND stable_metrics_30d AND tracking_error_within_bounds",
            "failure_modes_monitored": [
                "data_source_latency_spike",
                "event_type_misclassification_cluster",
                "api_rate_limit_saturation"
            ]
        },
        "architectural_enhancements": {
            "feedback_loop": {
                "manual_override_logging": True,
                "pnl_attribution_to_event_type": "enabled"
            },
            "contingency_handling": {
                "sec_edgar_down_policy": "fallback_to_secondary_aggregators",
                "stale_data_cutoff_hours": 24
            }
        }
    }

    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)

    logger.info(f"[CONFIG] Created default configuration: {config_path}")
    return str(config_path)


if __name__ == "__main__":
    # Create default config and test initialization
    config_path = create_default_config()

    # Test module initialization
    hybrid = EventQuantHybrid(config_path)
    stats = hybrid.get_statistics()

    print(f"\n[EVENT_QUANT_HYBRID] Module Statistics:")
    print(json.dumps(stats, indent=2, default=str))
