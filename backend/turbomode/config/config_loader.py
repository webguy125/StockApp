"""
TurboMode Config Loader
Loads and validates the TurboMode training configuration

Architecture: MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
Config File: turbomode_training_config.json

This module is the ONLY way to access training configuration.
All training rules come from JSON files, NOT from databases.
"""

import json
import os
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger('config_loader')


class TurboModeConfigLoader:
    """
    Load and validate TurboMode training configuration
    Ensures config integrity and provides typed access to settings
    """

    def __init__(self, config_path: str = None):
        """
        Initialize config loader

        Args:
            config_path: Path to turbomode_training_config.json
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                'turbomode_training_config.json'
            )

        self.config_path = config_path
        self.config = None
        self._load_config()

    def _load_config(self):
        """Load config from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            logger.info(f"[OK] Config loaded: version {self.config['config_metadata']['version']}")

        except FileNotFoundError:
            logger.error(f"[ERROR] Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Invalid JSON in config file: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate config structure and values

        Returns:
            True if valid, raises exception otherwise
        """
        required_sections = [
            'config_metadata',
            'rare_event_archive',
            'regime_labeling_rules',
            'balanced_sampling_ratios',
            'validation_set_definitions',
            'regime_weighted_loss',
            'drift_detection_thresholds',
            'model_promotion_gate_rules',
            'training_orchestrator_steps',
            'data_sources'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate data sources
        master_db_path = self.config['data_sources']['master_market_data_db']['path']
        if not os.path.exists(master_db_path):
            logger.warning(f"[WARNING] Master Market Data DB not found: {master_db_path}")

        turbomode_db_path = self.config['data_sources']['turbomode_db']['path']
        if not os.path.exists(turbomode_db_path):
            logger.warning(f"[WARNING] TurboMode DB not found: {turbomode_db_path}")

        logger.info("[OK] Config validation passed")
        return True

    def get_version(self) -> str:
        """Get config version"""
        return self.config['config_metadata']['version']

    def get_regime_rules(self) -> Dict[str, Any]:
        """Get regime labeling rules"""
        return self.config['regime_labeling_rules']

    def get_sampling_ratios(self) -> Dict[str, Any]:
        """Get balanced sampling ratios"""
        return self.config['balanced_sampling_ratios']

    def get_drift_thresholds(self) -> Dict[str, Any]:
        """Get drift detection thresholds"""
        return self.config['drift_detection_thresholds']

    def get_promotion_gate_rules(self) -> Dict[str, Any]:
        """Get model promotion gate rules"""
        return self.config['model_promotion_gate_rules']

    def get_orchestrator_steps(self) -> list:
        """Get training orchestrator steps"""
        return self.config['training_orchestrator_steps']['steps']

    def get_parallelizable_steps(self) -> list:
        """Get steps that can be parallelized"""
        return self.config['training_orchestrator_steps']['parallelizable_steps']

    def get_ensemble_config(self) -> Dict[str, Any]:
        """Get ensemble configuration"""
        return self.config['ensemble_configuration']

    def get_data_sources(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return self.config['data_sources']

    def get_master_db_path(self) -> str:
        """Get Master Market Data DB path"""
        return self.config['data_sources']['master_market_data_db']['path']

    def get_turbomode_db_path(self) -> str:
        """Get TurboMode DB path"""
        return self.config['data_sources']['turbomode_db']['path']

    def get_shap_settings(self) -> Dict[str, Any]:
        """Get SHAP analysis settings"""
        return self.config['shap_analysis_settings']

    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config['feature_engineering']

    def get_validation_strategy(self) -> Dict[str, Any]:
        """Get validation set definitions"""
        return self.config['validation_set_definitions']

    def get_regime_weighted_loss(self) -> Dict[str, Any]:
        """Get regime-weighted loss configuration"""
        return self.config['regime_weighted_loss']

    def get_error_replay_buffer_rules(self) -> Dict[str, Any]:
        """Get error replay buffer configuration"""
        return self.config['error_replay_buffer_rules']

    def get_rare_event_archive_rules(self) -> Dict[str, Any]:
        """Get rare event archive configuration"""
        return self.config['rare_event_archive']

    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self.config

    def reload(self):
        """Reload config from disk (for dynamic updates)"""
        self._load_config()
        logger.info("[OK] Config reloaded")


# Singleton instance for easy import
_config_loader = None


def get_config() -> TurboModeConfigLoader:
    """
    Get singleton config loader instance

    Returns:
        TurboModeConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = TurboModeConfigLoader()
        _config_loader.validate()
    return _config_loader


if __name__ == '__main__':
    # Test config loader
    print("=" * 80)
    print("TURBOMODE CONFIG LOADER - TEST")
    print("=" * 80)

    loader = TurboModeConfigLoader()

    print(f"\nConfig Version: {loader.get_version()}")
    print(f"Master DB Path: {loader.get_master_db_path()}")
    print(f"TurboMode DB Path: {loader.get_turbomode_db_path()}")

    print("\nOrchestrator Steps:")
    for i, step in enumerate(loader.get_orchestrator_steps(), 1):
        parallel = " (parallelizable)" if step in loader.get_parallelizable_steps() else ""
        print(f"  {i}. {step}{parallel}")

    print("\nPromotion Gate Rules:")
    gate_rules = loader.get_promotion_gate_rules()
    print(f"  Min Accuracy: {gate_rules['min_accuracy']:.1%}")
    print(f"  Min Precision: {gate_rules['min_precision']:.1%}")
    print(f"  Min Recall: {gate_rules['min_recall']:.1%}")

    print("\nDrift Thresholds:")
    drift = loader.get_drift_thresholds()
    print(f"  PSI Threshold: {drift['psi_threshold']}")
    print(f"  KL Divergence Threshold: {drift['kl_divergence_threshold']}")

    loader.validate()

    print("\n" + "=" * 80)
    print("CONFIG LOADER TEST COMPLETE")
    print("=" * 80)
