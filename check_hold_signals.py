"""
Quick script to check HOLD signals in turbomode.db
"""
import sqlite3
from pathlib import Path

db_path = Path(r"C:\StockApp\backend\data\turbomode.db")

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Count signals by type
cursor.execute("""
    SELECT
        COUNT(*) as total_signals,
        SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_signals,
        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals
    FROM active_signals
    WHERE status = 'ACTIVE';
""")

result = cursor.fetchone()
print("=" * 80)
print("ACTIVE SIGNALS SUMMARY")
print("=" * 80)
print(f"Total Active Signals: {result['total_signals']}")
print(f"  BUY signals:  {result['buy_signals'] or 0}")
print(f"  SELL signals: {result['sell_signals'] or 0}")
print(f"  HOLD signals: {result['hold_signals'] or 0}")
print()

# Get sample HOLD signals if they exist
if result['hold_signals'] and result['hold_signals'] > 0:
    print("=" * 80)
    print(f"SAMPLE HOLD SIGNALS (showing up to 10)")
    print("=" * 80)

    cursor.execute("""
        SELECT
            symbol,
            confidence,
            entry_price,
            current_price,
            stop_upper,
            stop_lower,
            prob_buy,
            prob_sell,
            prob_hold,
            sector,
            age_days
        FROM active_signals
        WHERE status = 'ACTIVE' AND signal_type = 'HOLD'
        ORDER BY confidence DESC
        LIMIT 10;
    """)

    holds = cursor.fetchall()
    for hold in holds:
        print(f"\n{hold['symbol']} ({hold['sector']})")
        print(f"  Confidence: {hold['confidence']:.3f}")
        print(f"  Entry Price: ${hold['entry_price']:.2f}")
        print(f"  Current Price: ${hold['current_price']:.2f}")
        print(f"  Stop Upper: ${hold['stop_upper']:.2f}" if hold['stop_upper'] else "  Stop Upper: None")
        print(f"  Stop Lower: ${hold['stop_lower']:.2f}" if hold['stop_lower'] else "  Stop Lower: None")
        print(f"  Probabilities: BUY={hold['prob_buy']:.3f}, SELL={hold['prob_sell']:.3f}, HOLD={hold['prob_hold']:.3f}")
        print(f"  Age: {hold['age_days']} days")
else:
    print("No HOLD signals found in database.")

conn.close()
print("\n" + "=" * 80)
