"""
Check probability values and ratios in active signals
"""
import sqlite3
from pathlib import Path
import numpy as np

db_path = Path(r"C:\StockApp\backend\data\turbomode.db")

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all active signals with probabilities
cursor.execute("""
    SELECT
        symbol,
        signal_type,
        confidence,
        prob_buy,
        prob_sell,
        prob_hold
    FROM active_signals
    WHERE status = 'ACTIVE' AND prob_buy IS NOT NULL AND prob_sell IS NOT NULL
    ORDER BY signal_type, confidence DESC;
""")

signals = cursor.fetchall()

print("=" * 80)
print("PROBABILITY ANALYSIS")
print("=" * 80)
print(f"Total signals with probabilities: {len(signals)}\n")

# Analyze by signal type
for signal_type in ['BUY', 'SELL', 'HOLD']:
    type_signals = [s for s in signals if s['signal_type'] == signal_type]

    if not type_signals:
        print(f"\n{signal_type} SIGNALS: 0")
        continue

    print(f"\n{signal_type} SIGNALS: {len(type_signals)}")
    print("-" * 80)

    # Calculate statistics
    prob_buys = [s['prob_buy'] for s in type_signals]
    prob_sells = [s['prob_sell'] for s in type_signals]
    prob_holds = [s['prob_hold'] for s in type_signals if s['prob_hold'] is not None]

    print(f"  prob_buy:  min={min(prob_buys):.3f}, max={max(prob_buys):.3f}, avg={np.mean(prob_buys):.3f}")
    print(f"  prob_sell: min={min(prob_sells):.3f}, max={max(prob_sells):.3f}, avg={np.mean(prob_sells):.3f}")
    if prob_holds:
        print(f"  prob_hold: min={min(prob_holds):.3f}, max={max(prob_holds):.3f}, avg={np.mean(prob_holds):.3f}")

    # Calculate |prob_buy - prob_sell| differences
    diffs = [abs(s['prob_buy'] - s['prob_sell']) for s in type_signals]
    print(f"\n  |prob_buy - prob_sell|:")
    print(f"    min={min(diffs):.3f}, max={max(diffs):.3f}, avg={np.mean(diffs):.3f}")

    # Calculate model_std for each
    model_stds = []
    for s in type_signals:
        ph = s['prob_hold'] if s['prob_hold'] is not None else 0.0
        model_std = np.std([s['prob_buy'], s['prob_sell'], ph])
        model_stds.append(model_std)

    print(f"\n  model_std (std of [prob_buy, prob_sell, prob_hold]):")
    print(f"    min={min(model_stds):.3f}, max={max(model_stds):.3f}, avg={np.mean(model_stds):.3f}")

    # Calculate neutrality band thresholds
    band_075 = [0.75 * std for std in model_stds]
    band_100 = [1.0 * std for std in model_stds]
    band_150 = [1.5 * std for std in model_stds]

    print(f"\n  Neutrality Band Thresholds:")
    print(f"    0.75x std: min={min(band_075):.3f}, max={max(band_075):.3f}, avg={np.mean(band_075):.3f}")
    print(f"    1.0x std:  min={min(band_100):.3f}, max={max(band_100):.3f}, avg={np.mean(band_100):.3f}")
    print(f"    1.5x std:  min={min(band_150):.3f}, max={max(band_150):.3f}, avg={np.mean(band_150):.3f}")

    # Show how many would be HOLD with different bands
    would_be_hold_075 = sum(1 for i, s in enumerate(type_signals) if diffs[i] < band_075[i])
    would_be_hold_100 = sum(1 for i, s in enumerate(type_signals) if diffs[i] < band_100[i])
    would_be_hold_150 = sum(1 for i, s in enumerate(type_signals) if diffs[i] < band_150[i])

    print(f"\n  Would be classified as HOLD:")
    print(f"    0.75x band: {would_be_hold_075}/{len(type_signals)} ({100*would_be_hold_075/len(type_signals):.1f}%)")
    print(f"    1.0x band:  {would_be_hold_100}/{len(type_signals)} ({100*would_be_hold_100/len(type_signals):.1f}%)")
    print(f"    1.5x band:  {would_be_hold_150}/{len(type_signals)} ({100*would_be_hold_150/len(type_signals):.1f}%)")

    # Show top 5 examples
    print(f"\n  Top 5 {signal_type} signals by confidence:")
    for s in type_signals[:5]:
        ph = s['prob_hold'] if s['prob_hold'] is not None else 0.0
        diff = abs(s['prob_buy'] - s['prob_sell'])
        model_std = np.std([s['prob_buy'], s['prob_sell'], ph])
        print(f"    {s['symbol']:6s} | BUY={s['prob_buy']:.3f} SELL={s['prob_sell']:.3f} HOLD={ph:.3f} | diff={diff:.3f} | std={model_std:.3f} | 1.5x={1.5*model_std:.3f}")

conn.close()
print("\n" + "=" * 80)
