"""
Test ML Model Configuration System
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from trading_system.ml_model_manager import MLModelManager

def test_model_creation():
    """Test creating, activating, and switching between models"""
    print("=" * 60)
    print("ML MODEL CONFIGURATION TEST")
    print("=" * 60)

    manager = MLModelManager()

    # Test 1: Create a price action model
    print("\n[TEST 1] Creating Price Action Model...")
    config1 = {
        'name': 'Pure Price Action',
        'analysis_type': 'price_action',
        'philosophy': ['indicators_lag', 'ml_discover', 'simpler'],
        'win_criteria': 'price_movement',
        'hold_period_days': 14,
        'win_threshold_pct': 10.0,
        'loss_threshold_pct': -5.0
    }

    success = manager.create_configuration('Pure Price Action', config1)
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Test 2: Create a volume profile model
    print("\n[TEST 2] Creating Volume Profile Model...")
    config2 = {
        'name': 'Volume Beast',
        'analysis_type': 'volume_profile',
        'philosophy': ['ml_discover'],
        'win_criteria': 'quality_of_move',
        'hold_period_days': 14,
        'win_threshold_pct': 10.0,
        'loss_threshold_pct': -5.0
    }

    success = manager.create_configuration('Volume Beast', config2)
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Test 3: List all configurations
    print("\n[TEST 3] Listing all configurations...")
    configs = manager.get_all_configurations()
    print(f"   Found {len(configs)} configurations:")
    for cfg in configs:
        print(f"      - {cfg['name']} ({cfg['analysis_type']}) - Active: {cfg['active']}")

    # Test 4: Activate price action model
    print("\n[TEST 4] Activating 'Pure Price Action' model...")
    success = manager.activate_model('Pure Price Action')
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Test 5: Verify active model
    print("\n[TEST 5] Checking active model...")
    active = manager.get_active_model()
    if active:
        print(f"   Active model: {active['name']}")
        print(f"   Analysis type: {active['analysis_type']}")
    else:
        print("   ERROR: No active model!")

    # Test 6: Initialize TradingSystem with active model
    print("\n[TEST 6] Initializing TradingSystem with active model...")
    from trading_system.core.trading_system import TradingSystem
    ts = TradingSystem()
    print(f"   Active model name: {ts.active_model_name}")
    print(f"   Analyzers loaded: {ts.registry.get_enabled_count()}")
    print(f"   Analyzer types: {ts.registry.list_all()}")

    # Test 7: Switch to volume model
    print("\n[TEST 7] Switching to 'Volume Beast' model...")
    success = manager.activate_model('Volume Beast')
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Test 8: Create new TradingSystem instance (should load volume analyzer)
    print("\n[TEST 8] Creating new TradingSystem with Volume Beast...")
    ts2 = TradingSystem()
    print(f"   Active model name: {ts2.active_model_name}")
    print(f"   Analyzers loaded: {ts2.registry.get_enabled_count()}")
    print(f"   Analyzer types: {ts2.registry.list_all()}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    test_model_creation()
