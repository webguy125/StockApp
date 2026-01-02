"""
Diagnostic script to check raw model predictions
Shows buy_prob, sell_prob, and final prediction for all 80+ stocks
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from turbomode.overnight_scanner import OvernightScanner

def main():
    scanner = OvernightScanner()

    # Get curated stock list
    stocks = scanner._load_curated_stocks()

    print("\n" + "=" * 90)
    print("RAW MODEL PREDICTIONS FOR ALL STOCKS")
    print("=" * 90)
    print(f"{'Symbol':<8} {'BUY Prob':<10} {'SELL Prob':<10} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 90)

    buy_count = 0
    sell_count = 0
    hold_count = 0

    buy_probs = []
    sell_probs = []

    for symbol in stocks[:20]:  # Check first 20 stocks
        try:
            # Extract features
            features = scanner.extract_features(symbol)
            if features is None:
                continue

            # Get prediction
            pred = scanner.get_prediction(features)

            buy_probs.append(pred['buy_prob'])
            sell_probs.append(pred['sell_prob'])

            if pred['prediction'] == 'buy':
                buy_count += 1
            elif pred['prediction'] == 'sell':
                sell_count += 1
            else:
                hold_count += 1

            print(f"{symbol:<8} {pred['buy_prob']:<10.4f} {pred['sell_prob']:<10.4f} "
                  f"{pred['prediction']:<12} {pred['confidence']:<12.4f}")

        except Exception as e:
            print(f"{symbol:<8} ERROR: {e}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"BUY predictions: {buy_count}")
    print(f"SELL predictions: {sell_count}")
    print(f"HOLD predictions: {hold_count}")

    if buy_probs:
        print(f"\nAverage BUY probability: {sum(buy_probs)/len(buy_probs):.4f}")
    if sell_probs:
        print(f"Average SELL probability: {sum(sell_probs)/len(sell_probs):.4f}")

if __name__ == "__main__":
    main()
