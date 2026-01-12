"""
Quick check: Do turbomode.db samples have features?
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "turbomode.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if entry_features_json is populated
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN entry_features_json IS NULL THEN 1 ELSE 0 END) as null_features,
        SUM(CASE WHEN entry_features_json IS NOT NULL THEN 1 ELSE 0 END) as with_features
    FROM trades
    WHERE trade_type = 'backtest'
""")

result = cursor.fetchone()
total, null_features, with_features = result

print("=" * 70)
print("TURBOMODE.DB FEATURE STATUS")
print("=" * 70)
print(f"Total backtest samples: {total:,}")
print(f"Samples WITH features:  {with_features:,} ({with_features/total*100:.1f}%)")
print(f"Samples WITHOUT features: {null_features:,} ({null_features/total*100:.1f}%)")
print()

if with_features == 0:
    print("[CRITICAL] NO FEATURES IN DATABASE!")
    print("This explains why prepare_training_data() will fail.")
    print()
    print("ROOT CAUSE:")
    print("  turbomode_backtest.py only generates labels (outcome)")
    print("  It does NOT extract 179 features from price data")
    print()
    print("SOLUTION:")
    print("  Need to create TurboMode-native feature extraction pipeline")
    print("  that populates entry_features_json during backtest generation")
elif with_features < total:
    print(f"[WARNING] {null_features:,} samples missing features")
    print("Partial feature extraction - may cause training issues")
else:
    print("[OK] All samples have features")
    # Show sample
    cursor.execute("""
        SELECT entry_features_json
        FROM trades
        WHERE trade_type = 'backtest' AND entry_features_json IS NOT NULL
        LIMIT 1
    """)
    sample = cursor.fetchone()
    if sample:
        import json
        features = json.loads(sample[0])
        print(f"\nSample feature count: {len(features)} keys")

conn.close()
