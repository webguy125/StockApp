"""Quick test for TMDX and NVDA predictions"""
import sys
sys.path.insert(0, 'C:\\StockApp')

from backend.turbomode.overnight_scanner import OvernightScanner

scanner = OvernightScanner()
scanner._load_models()

for symbol in ['TMDX', 'NVDA']:
    print(f"\n{'='*50}")
    print(f"Testing {symbol}")
    print(f"{'='*50}")

    # Get current price
    price = scanner.get_current_price(symbol)
    print(f"Current price: ${price:.2f}")

    # Extract features
    features = scanner.extract_features(symbol)
    if features is None:
        print(f"ERROR: Could not extract features for {symbol}")
        continue

    # Get prediction
    pred = scanner.get_prediction(features)
    print(f"Prediction: {pred['prediction'].upper()}")
    print(f"Confidence: {pred['confidence']:.1%}")
    print(f"Meets 65% threshold: {'YES' if pred['confidence'] >= 0.65 else 'NO'}")

    if pred['prediction'] == 'buy' and pred['confidence'] >= 0.65:
        print("✅ Would generate BUY signal")
    elif pred['prediction'] == 'sell' and pred['confidence'] >= 0.65:
        print("✅ Would generate SELL signal")
    else:
        print("❌ Would NOT generate signal (HOLD or low confidence)")
