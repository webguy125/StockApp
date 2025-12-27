"""
Test ML Trading System Imports
Quick test to verify all imports work before running through Flask
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print("Testing ML Trading System imports...")
print("=" * 60)

try:
    print("\n1. Testing TradingSystem import...")
    from trading_system.core.trading_system import TradingSystem
    print("   ✅ TradingSystem imported successfully")

    print("\n2. Testing analyzer imports...")
    from trading_system.analyzers.rsi_analyzer import RSIAnalyzer
    from trading_system.analyzers.macd_analyzer import MACDAnalyzer
    from trading_system.analyzers.volume_analyzer import VolumeAnalyzer
    from trading_system.analyzers.trend_analyzer import TrendAnalyzer
    print("   ✅ All analyzers imported successfully")

    print("\n3. Testing core components...")
    from trading_system.core.stock_scanner import StockScanner
    from trading_system.core.analyzer_registry import AnalyzerRegistry
    from trading_system.core.feature_extractor import FeatureExtractor
    from trading_system.core.trade_tracker import TradeTracker
    from trading_system.models.simple_trading_model import SimpleTradingModel
    print("   ✅ All core components imported successfully")

    print("\n4. Initializing TradingSystem...")
    system = TradingSystem()
    print(f"   ✅ TradingSystem initialized: {system}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe ML Trading System is ready to use.")
    print("You can now:")
    print("  1. Run Flask server")
    print("  2. Go to http://127.0.0.1:5000/ml-trading")
    print("  3. Click 'Run Scan'")
    print("\n" + "=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("Fix the error above before running the ML system.")
