"""Quick script to check raw predictions for specific stocks"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from turbomode.overnight_scanner import OvernightScanner

# Test a few stocks
test_symbols = ['AAPL', 'TSLA', 'AMD', 'NVDA', 'META', 'GOOGL', 'MSFT', 'AMZN']

scanner = OvernightScanner()

print("\n" + "=" * 90)
print("RAW MODEL PREDICTIONS")
print("=" * 90)
print(f"{'Symbol':<8} {'Prediction':<12} {'Confidence':<12} {'BUY Prob':<12} {'SELL Prob':<12}")
print("-" * 90)

for symbol in test_symbols:
    try:
        features = scanner.extract_features(symbol)
        if features is None:
            print(f"{symbol:<8} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        pred = scanner.get_prediction(features)

        print(f"{symbol:<8} {pred['prediction']:<12} {pred['confidence']:<12.4f} "
              f"{pred['buy_prob']:<12.4f} {pred['sell_prob']:<12.4f}")

    except Exception as e:
        print(f"{symbol:<8} ERROR: {e}")

print("\n")
