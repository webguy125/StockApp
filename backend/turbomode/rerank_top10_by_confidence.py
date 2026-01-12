"""
Reorganize Top 10 stocks by strongest BUY signal confidence
Uses latest predictions from all_predictions.json
"""

import json
import os
from datetime import datetime

def rerank_top10_by_confidence():
    """
    Reorder the Top 10 stocks in stock_rankings.json by current BUY confidence
    from all_predictions.json (strongest signals first)
    """

    # File paths
    rankings_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_rankings.json')
    predictions_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'all_predictions.json')

    # Load current rankings
    print("Loading stock_rankings.json...")
    with open(rankings_file, 'r') as f:
        rankings_data = json.load(f)

    # Load latest predictions
    print("Loading all_predictions.json...")
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    # Create prediction lookup by symbol
    predictions_by_symbol = {}
    for pred in predictions_data['predictions']:
        predictions_by_symbol[pred['symbol']] = {
            'prediction': pred['prediction'],
            'confidence': pred['confidence'],
            'current_price': pred['current_price'],
            'sector': pred['sector']
        }

    # Get current Top 10
    current_top10 = rankings_data['top_10']

    print(f"\nCurrent Top 10 order (by composite score):")
    for i, stock in enumerate(current_top10, 1):
        symbol = stock['symbol']
        pred = predictions_by_symbol.get(symbol, {})
        conf = pred.get('confidence', 0.0) * 100
        print(f"  {i}. {symbol:6s} - Composite: {stock['composite_score']:.4f}, "
              f"Signal: {pred.get('prediction', 'N/A'):4s} ({conf:.1f}%)")

    # Enrich Top 10 with current confidence
    enriched_top10 = []
    for stock in current_top10:
        symbol = stock['symbol']
        pred = predictions_by_symbol.get(symbol, {})

        stock_data = stock.copy()
        stock_data['current_confidence'] = pred.get('confidence', 0.0)
        stock_data['current_signal'] = pred.get('prediction', 'hold')
        stock_data['current_price'] = pred.get('current_price', 0.0)

        enriched_top10.append(stock_data)

    # Sort by confidence (highest first)
    enriched_top10.sort(key=lambda x: x['current_confidence'], reverse=True)

    print(f"\n{'='*70}")
    print("NEW Top 10 order (by BUY signal confidence - STRONGEST FIRST):")
    print(f"{'='*70}")
    for i, stock in enumerate(enriched_top10, 1):
        symbol = stock['symbol']
        conf = stock['current_confidence'] * 100
        signal = stock['current_signal'].upper()

        print(f"  {i}. {symbol:6s} - {signal} ({conf:.1f}%) | "
              f"Composite: {stock['composite_score']:.4f} | "
              f"Win Rate 30d: {stock['win_rate_30d']*100:.0f}%")

    # Remove enrichment fields before saving
    final_top10 = []
    for stock in enriched_top10:
        stock_clean = {k: v for k, v in stock.items()
                      if k not in ['current_confidence', 'current_signal', 'current_price']}
        final_top10.append(stock_clean)

    # Update rankings data
    rankings_data['top_10'] = final_top10
    rankings_data['timestamp'] = datetime.now().isoformat()
    rankings_data['last_rerank'] = {
        'timestamp': datetime.now().isoformat(),
        'method': 'confidence_based',
        'note': 'Reranked by current BUY signal confidence'
    }

    # Backup original file
    backup_file = rankings_file.replace('.json', '_backup_before_rerank.json')
    print(f"\n[OK] Creating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(rankings_data, f, indent=2)

    # Save updated rankings
    print(f"[OK] Saving updated rankings to: {rankings_file}")
    with open(rankings_file, 'w') as f:
        json.dump(rankings_data, f, indent=2)

    print(f"\n{'='*70}")
    print("RERANKING COMPLETE!")
    print(f"{'='*70}")
    print(f"Top 10 now ordered by strongest BUY signals")
    print(f"Backup saved to: {backup_file}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    rerank_top10_by_confidence()
