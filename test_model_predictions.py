"""
Quick diagnostic to check model predictions for all 80 stocks
"""
import sys
sys.path.insert(0, 'C:\\StockApp')

from backend.turbomode.overnight_scanner import OvernightScanner

scanner = OvernightScanner()

# Load models
print("Loading models...")
scanner._load_models()

# Get all symbols
from backend.advanced_ml.config.core_symbols import get_all_core_symbols

SYMBOLS = get_all_core_symbols()
print(f"\nTesting predictions for {len(SYMBOLS)} stocks...\n")

buy_count = 0
sell_count = 0
hold_count = 0

buy_list = []
sell_list = []

for symbol in SYMBOLS:
    try:
        # Extract features
        features = scanner.extract_features(symbol)
        if features is None:
            print(f"{symbol}: Could not extract features")
            continue

        # Get prediction
        pred = scanner.get_prediction(features)

        if pred['prediction'] == 'buy':
            buy_count += 1
            buy_list.append((symbol, pred['confidence']))
        elif pred['prediction'] == 'sell':
            sell_count += 1
            sell_list.append((symbol, pred['confidence']))
        else:
            hold_count += 1

    except Exception as e:
        print(f"{symbol}: Error - {e}")

print(f"\n{'='*60}")
print(f"PREDICTION SUMMARY")
print(f"{'='*60}")
print(f"BUY predictions:  {buy_count} ({buy_count/len(SYMBOLS)*100:.1f}%)")
print(f"SELL predictions: {sell_count} ({sell_count/len(SYMBOLS)*100:.1f}%)")
print(f"HOLD predictions: {hold_count} ({hold_count/len(SYMBOLS)*100:.1f}%)")

print(f"\n{'='*60}")
print(f"BUY SIGNALS (confidence >= 65%)")
print(f"{'='*60}")
buy_signals = [b for b in buy_list if b[1] >= 0.65]
buy_signals.sort(key=lambda x: x[1], reverse=True)
for symbol, conf in buy_signals[:20]:
    print(f"{symbol}: {conf:.1%}")

print(f"\n{'='*60}")
print(f"SELL SIGNALS (confidence >= 65%)")
print(f"{'='*60}")
sell_signals = [s for s in sell_list if s[1] >= 0.65]
sell_signals.sort(key=lambda x: x[1], reverse=True)
if sell_signals:
    for symbol, conf in sell_signals[:20]:
        print(f"{symbol}: {conf:.1%}")
else:
    print("NO SELL SIGNALS WITH >= 65% CONFIDENCE")

print(f"\nTop 5 SELL predictions (any confidence):")
sell_list.sort(key=lambda x: x[1], reverse=True)
for symbol, conf in sell_list[:5]:
    print(f"{symbol}: {conf:.1%}")
