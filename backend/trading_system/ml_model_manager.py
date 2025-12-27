"""
ML Model Manager
Manages multiple ML model configurations, training, and switching
Each configuration can use different analyzers and criteria
"""

import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class MLModelManager:
    """
    Manages multiple ML model configurations
    - Create/delete configurations
    - Save/load trained models
    - Switch active model
    - Track performance per model
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'data', 'ml_models')

        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.state_file = os.path.join(os.path.dirname(self.models_dir), 'ml_model_state.json')

    def create_configuration(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Create a new model configuration

        Args:
            name: Model name (will be sanitized for filesystem)
            config: Configuration dict with:
                - analysis_type: 'price_action', 'volume_profile', 'raw_ohlcv', 'market_structure'
                - philosophy: List of philosophy choices
                - win_criteria: 'price_movement', 'quality_of_move', 'speed_of_move', 'risk_adjusted'
                - hold_period_days: int
                - win_threshold_pct: float
                - loss_threshold_pct: float

        Returns:
            True if created successfully
        """
        # Sanitize model name for filesystem
        safe_name = self._sanitize_name(name)

        # Create model directory
        model_dir = os.path.join(self.models_dir, safe_name)

        if os.path.exists(model_dir):
            return False  # Model already exists

        os.makedirs(model_dir, exist_ok=True)

        # Add metadata
        full_config = {
            'name': name,
            'safe_name': safe_name,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'active': False,
            'trades_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.0,
            **config
        }

        # Save configuration
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)

        # Initialize performance tracking
        performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0,
            'total_profit_loss': 0.0,
            'training_history': []
        }

        perf_file = os.path.join(model_dir, 'performance.json')
        with open(perf_file, 'w') as f:
            json.dump(performance, f, indent=2)

        print(f"[OK] Created model configuration: {name}")
        return True

    def get_all_configurations(self) -> List[Dict[str, Any]]:
        """Get list of all model configurations"""
        configs = []

        if not os.path.exists(self.models_dir):
            return configs

        for model_dir in os.listdir(self.models_dir):
            config_file = os.path.join(self.models_dir, model_dir, 'config.json')

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load performance data
                perf_file = os.path.join(self.models_dir, model_dir, 'performance.json')
                if os.path.exists(perf_file):
                    with open(perf_file, 'r') as f:
                        perf = json.load(f)
                    config.update(perf)

                configs.append(config)

        # Sort by creation date (newest first)
        configs.sort(key=lambda x: x.get('created', ''), reverse=True)

        return configs

    def get_configuration(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific model configuration"""
        safe_name = self._sanitize_name(name)
        config_file = os.path.join(self.models_dir, safe_name, 'config.json')

        if not os.path.exists(config_file):
            return None

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load performance
        perf_file = os.path.join(self.models_dir, safe_name, 'performance.json')
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                perf = json.load(f)
            config.update(perf)

        return config

    def activate_model(self, name: str) -> bool:
        """
        Activate a model (make it the active one)
        Deactivates all other models
        """
        safe_name = self._sanitize_name(name)
        config_file = os.path.join(self.models_dir, safe_name, 'config.json')

        if not os.path.exists(config_file):
            return False

        # Deactivate all models
        for model_dir in os.listdir(self.models_dir):
            model_config_file = os.path.join(self.models_dir, model_dir, 'config.json')
            if os.path.exists(model_config_file):
                with open(model_config_file, 'r') as f:
                    cfg = json.load(f)
                cfg['active'] = False
                with open(model_config_file, 'w') as f:
                    json.dump(cfg, f, indent=2)

        # Activate target model
        with open(config_file, 'r') as f:
            config = json.load(f)

        config['active'] = True
        config['modified'] = datetime.now().isoformat()

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Update state file
        self._update_state(safe_name)

        print(f"[OK] Activated model: {name}")
        return True

    def delete_configuration(self, name: str) -> bool:
        """Delete a model configuration and its data"""
        safe_name = self._sanitize_name(name)
        model_dir = os.path.join(self.models_dir, safe_name)

        if not os.path.exists(model_dir):
            return False

        # Don't delete if active
        config_file = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if config.get('active', False):
                print(f"[ERROR] Cannot delete active model: {name}")
                return False

        # Delete directory
        shutil.rmtree(model_dir)

        print(f"[OK] Deleted model configuration: {name}")
        return True

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get the currently active model configuration"""
        configs = self.get_all_configurations()

        for config in configs:
            if config.get('active', False):
                return config

        return None

    def save_model(self, name: str, model_obj: Any, scaler_obj: Any = None):
        """Save trained model and scaler"""
        safe_name = self._sanitize_name(name)
        model_dir = os.path.join(self.models_dir, safe_name)

        if not os.path.exists(model_dir):
            return False

        # Save model
        model_file = os.path.join(model_dir, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model_obj, f)

        # Save scaler if provided
        if scaler_obj is not None:
            scaler_file = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler_obj, f)

        print(f"[OK] Saved model: {name}")
        return True

    def load_model(self, name: str) -> Optional[tuple]:
        """
        Load trained model and scaler

        Returns:
            (model, scaler) or None if not found
        """
        safe_name = self._sanitize_name(name)
        model_dir = os.path.join(self.models_dir, safe_name)

        model_file = os.path.join(model_dir, 'model.pkl')
        scaler_file = os.path.join(model_dir, 'scaler.pkl')

        if not os.path.exists(model_file):
            return None

        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        scaler = None
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)

        return (model, scaler)

    def update_performance(self, name: str, stats: Dict[str, Any]):
        """Update performance statistics for a model"""
        safe_name = self._sanitize_name(name)
        perf_file = os.path.join(self.models_dir, safe_name, 'performance.json')

        if not os.path.exists(perf_file):
            return False

        with open(perf_file, 'r') as f:
            perf = json.load(f)

        # Update stats
        perf.update(stats)
        perf['last_updated'] = datetime.now().isoformat()

        with open(perf_file, 'w') as f:
            json.dump(perf, f, indent=2)

        # Also update config
        config_file = os.path.join(self.models_dir, safe_name, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)

            config['trades_count'] = stats.get('total_trades', 0)
            config['win_count'] = stats.get('wins', 0)
            config['loss_count'] = stats.get('losses', 0)
            config['win_rate'] = stats.get('win_rate', 0.0)
            config['modified'] = datetime.now().isoformat()

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

        return True

    def _sanitize_name(self, name: str) -> str:
        """Convert model name to filesystem-safe name"""
        # Replace spaces and special chars with underscores
        safe = name.lower().replace(' ', '_')
        safe = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe)
        # Remove consecutive underscores
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_')

    def _update_state(self, active_model_name: str):
        """Update global state file with active model"""
        state = {
            'active_model': active_model_name,
            'last_updated': datetime.now().isoformat()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_model_path(self, name: str) -> str:
        """Get filesystem path for a model"""
        safe_name = self._sanitize_name(name)
        return os.path.join(self.models_dir, safe_name)
