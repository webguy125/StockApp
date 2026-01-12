"""
On-the-Fly Options Predictor
Generates ML-powered options recommendations for ANY symbol (not just Top 30)
Can be called from CLI or API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

# Import options analyzer
from turbomode.options_api import OptionsAnalyzer

def get_basic_ml_signal(symbol: str, ticker: yf.Ticker) -> Optional[Dict]:
    """
    Generate basic ML signal for any symbol using price momentum and volatility
    Fallback when no TurboMode prediction exists
    """
    try:
        # Get historical data (30 days)
        hist = ticker.history(period="30d")

        if len(hist) < 20:
            return None

        # Calculate momentum indicators
        current_price = float(hist['Close'].iloc[-1])
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_5 = hist['Close'].rolling(5).mean().iloc[-1]

        # Price change metrics
        pct_change_5d = (current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
        pct_change_20d = (current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]

        # Volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)

        # Determine signal
        bullish_score = 0

        # Trend indicators
        if current_price > sma_20:
            bullish_score += 1
        if sma_5 > sma_20:
            bullish_score += 1
        if pct_change_5d > 0:
            bullish_score += 1
        if pct_change_20d > 0:
            bullish_score += 1

        # Generate signal
        if bullish_score >= 3:
            signal_type = 'BUY'
            confidence = 0.55 + (bullish_score - 3) * 0.05  # 55-65%
        elif bullish_score <= 1:
            signal_type = 'SELL'
            confidence = 0.55 + (1 - bullish_score) * 0.05  # 55-60%
        else:
            signal_type = 'BUY'  # Neutral bias toward calls
            confidence = 0.50

        # Calculate target price (based on volatility and momentum)
        expected_move_pct = volatility * np.sqrt(14/252) * (confidence - 0.5) * 4  # 14-day move

        if signal_type == 'BUY':
            target_price = current_price * (1 + abs(expected_move_pct))
        else:
            target_price = current_price * (1 - abs(expected_move_pct))

        return {
            'signal': signal_type,
            'confidence': min(confidence, 0.75),  # Cap at 75% for basic signals
            'entry_price': current_price,
            'target_price': target_price,
            'source': 'momentum_based'
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate basic signal: {e}")
        return None

def predict_options_for_any_symbol(symbol: str, verbose: bool = True) -> Dict:
    """
    Main function to get options prediction for ANY symbol

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA', 'NVDA')
        verbose: Print progress messages

    Returns:
        Dictionary with options analysis or error
    """
    symbol = symbol.upper()

    if verbose:
        print(f"\n{'='*80}")
        print(f"ON-THE-FLY OPTIONS PREDICTOR - {symbol}")
        print(f"{'='*80}\n")

    try:
        # Initialize analyzer
        analyzer = OptionsAnalyzer()

        # Get ticker
        ticker = yf.Ticker(symbol)

        # Check if TurboMode has prediction
        ml_pred = analyzer.get_ml_prediction(symbol)

        if ml_pred:
            if verbose:
                print(f"[INFO] Found TurboMode prediction for {symbol}")
                print(f"  Signal: {ml_pred['signal']}")
                print(f"  Confidence: {ml_pred['confidence']:.1%}")

            # Use existing TurboMode prediction
            result = analyzer.analyze_options_chain(symbol)

        else:
            if verbose:
                print(f"[INFO] No TurboMode prediction found for {symbol}")
                print(f"[INFO] Generating momentum-based signal...")

            # Generate basic signal
            basic_signal = get_basic_ml_signal(symbol, ticker)

            if not basic_signal:
                return {
                    'error': f'Unable to generate signal for {symbol}',
                    'reason': 'Insufficient historical data'
                }

            if verbose:
                print(f"[OK] Generated signal: {basic_signal['signal']} (confidence: {basic_signal['confidence']:.1%})")

            # Temporarily inject signal into analyzer for options analysis
            # (Monkey-patch get_ml_prediction method)
            original_method = analyzer.get_ml_prediction
            analyzer.get_ml_prediction = lambda sym: basic_signal if sym == symbol else original_method(sym)

            # Analyze options chain
            result = analyzer.analyze_options_chain(symbol)

            # Restore original method
            analyzer.get_ml_prediction = original_method

            # Add source info to result
            if 'ml_prediction' in result:
                result['ml_prediction']['source'] = 'momentum_based'

        if 'error' in result:
            return result

        # Success
        if verbose:
            print(f"\n[SUCCESS] Options analysis complete for {symbol}")
            print(f"  Recommended: {result['recommended_option']['type']} ${result['recommended_option']['strike']}")
            print(f"  Expiration: {result['recommended_option']['expiration']} ({result['recommended_option']['dte']} DTE)")
            print(f"  Hybrid Score: {result['recommended_option']['hybrid_score']:.1f}/100")
            print(f"  Premium: ${result['recommended_option']['premium']}")
            print(f"  Take Profit: ${result['profit_targets']['take_profit_price']:.2f} (+{result['profit_targets']['take_profit_pnl']:.2f})")
            print(f"  Stop Loss: ${result['profit_targets']['stop_loss_price']:.2f} ({result['profit_targets']['stop_loss_pnl']:.2f})")

        return result

    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to analyze {symbol}: {e}")
            import traceback
            traceback.print_exc()

        return {
            'error': str(e),
            'symbol': symbol
        }

def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python onthefly_predictor.py <SYMBOL>")
        print("Example: python onthefly_predictor.py NVDA")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    # Get prediction
    result = predict_options_for_any_symbol(symbol, verbose=True)

    # Save to JSON file
    if 'error' not in result:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_predictions')
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'{symbol}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[SAVED] Prediction saved to: {output_file}")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
