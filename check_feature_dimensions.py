import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

print("Checking feature dimensions across all samples...")
cursor.execute('SELECT id, symbol, entry_date, entry_features_json FROM trades')
rows = cursor.fetchall()

feature_counts = {}
samples_by_count = {}

for row in rows:
    trade_id, symbol, date, features_json = row
    features = json.loads(features_json)
    count = len(features)

    if count not in feature_counts:
        feature_counts[count] = 0
        samples_by_count[count] = []

    feature_counts[count] += 1
    if len(samples_by_count[count]) < 3:  # Store first 3 examples
        samples_by_count[count].append((trade_id[:8], symbol, date))

print(f"\nTotal samples: {len(rows)}")
print(f"\nFeature count distribution:")
for count in sorted(feature_counts.keys()):
    print(f"  {count} features: {feature_counts[count]:5} samples", end="")
    if len(samples_by_count[count]) > 0:
        examples = samples_by_count[count][:2]
        print(f"  (e.g., {examples[0][1]} on {examples[0][2]})")
    else:
        print()

# If we have multiple feature counts, investigate
if len(feature_counts) > 1:
    print(f"\n[WARNING] Found {len(feature_counts)} different feature counts!")
    print("\nInvestigating differences...")

    counts = sorted(feature_counts.keys())
    min_count = counts[0]
    max_count = counts[-1]

    # Get one sample of each
    cursor.execute('SELECT entry_features_json FROM trades LIMIT 1 OFFSET ?',
                   (list(feature_counts.values())[0] - 1,))
    sample_min = json.loads(cursor.fetchone()[0])

    cursor.execute('SELECT entry_features_json FROM trades WHERE LENGTH(entry_features_json) = (SELECT MAX(LENGTH(entry_features_json)) FROM trades) LIMIT 1')
    sample_max = json.loads(cursor.fetchone()[0])

    print(f"\nSample with {len(sample_min)} features has keys:")
    print(f"  {sorted(sample_min.keys())[:10]}... (showing first 10)")

    print(f"\nSample with {len(sample_max)} features has keys:")
    print(f"  {sorted(sample_max.keys())[:10]}... (showing first 10)")

    # Check for NaN values
    cursor.execute('SELECT entry_features_json FROM trades LIMIT 100')
    samples = cursor.fetchall()
    nan_counts = []
    for s in samples:
        features = json.loads(s[0])
        nan_count = sum(1 for v in features.values() if v is None or (isinstance(v, float) and str(v) == 'nan'))
        nan_counts.append(nan_count)

    print(f"\nNaN/None values in first 100 samples:")
    print(f"  Min: {min(nan_counts)}, Max: {max(nan_counts)}, Avg: {sum(nan_counts)/len(nan_counts):.1f}")

conn.close()
