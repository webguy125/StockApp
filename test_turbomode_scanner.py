"""
Quick test of TurboMode scanner with just 5 symbols
"""

import sys
sys.path.insert(0, 'backend')

from backend.turbomode.overnight_scanner import OvernightScanner
from backend.turbomode.database_schema import TurboModeDB

print("\n" + "=" * 70)
print("TURBOMODE SCANNER - QUICK TEST")
print("=" * 70)

# Initialize database (will use test database)
db = TurboModeDB(db_path="backend/data/turbomode_test.db")

# Clear test data
print("\n[SETUP] Clearing test database...")
db.clear_all_data()

# Initialize scanner
print("\n[SETUP] Initializing scanner...")
scanner = OvernightScanner(
    db_path="backend/data/turbomode_test.db",
    ml_db_path="backend/backend/data/advanced_ml_system.db"
)

# Test with just 5 symbols
test_symbols = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'XOM']

print(f"\n[TEST] Scanning {len(test_symbols)} symbols...")
print(f"Symbols: {test_symbols}")

signals = {'buy_signals': [], 'sell_signals': []}

for i, symbol in enumerate(test_symbols, 1):
    print(f"\n[{i}/{len(test_symbols)}] Scanning {symbol}...")

    signal = scanner.scan_symbol(symbol)

    if signal:
        print(f"  [OK] {signal['signal_type']} signal - Confidence: {signal['confidence']:.2%}")
        print(f"      Entry: ${signal['entry_price']:.2f}")
        print(f"      Target: ${signal['target_price']:.2f}")
        print(f"      Stop: ${signal['stop_price']:.2f}")

        if signal['signal_type'] == 'BUY':
            signals['buy_signals'].append(signal)
        else:
            signals['sell_signals'].append(signal)
    else:
        print(f"  - No signal (HOLD or low confidence)")

# Save signals to database
print(f"\n[SAVE] Saving signals to database...")
for signal in signals['buy_signals'] + signals['sell_signals']:
    success = db.add_signal(signal)
    status = '[OK] Saved' if success else '[ERROR] Failed'
    print(f"  {signal['symbol']} ({signal['signal_type']}): {status}")

# Check database stats
print(f"\n[STATS] Database statistics...")
stats = db.get_stats()
print(f"  Active signals: {stats['active_signals']}")

# Get active signals
print(f"\n[RESULTS] Active signals in database:")
active_buy = db.get_active_signals(signal_type='BUY', limit=10)
active_sell = db.get_active_signals(signal_type='SELL', limit=10)

if active_buy:
    print(f"\n  BUY Signals ({len(active_buy)}):")
    for sig in active_buy:
        print(f"    {sig['symbol']}: {sig['confidence']:.2%} confidence")

if active_sell:
    print(f"\n  SELL Signals ({len(active_sell)}):")
    for sig in active_sell:
        print(f"    {sig['symbol']}: {sig['confidence']:.2%} confidence")

print("\n" + "=" * 70)
print("[OK] Scanner test complete!")
print("=" * 70)
