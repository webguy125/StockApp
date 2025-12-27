import sqlite3
import json

# The 23 missing event features with default values
EVENT_FEATURES = {
    'event_cross_source_confirmation_flag': 0.0,
    'event_event_count_dividend_30d': 0.0,
    'event_event_count_dividend_7d': 0.0,
    'event_event_count_dividend_90d': 0.0,
    'event_event_count_litigation_30d': 0.0,
    'event_event_count_litigation_7d': 0.0,
    'event_event_count_litigation_90d': 0.0,
    'event_event_count_negative_news_30d': 0.0,
    'event_event_count_negative_news_7d': 0.0,
    'event_event_count_negative_news_90d': 0.0,
    'event_event_count_refinancing_30d': 0.0,
    'event_event_count_refinancing_7d': 0.0,
    'event_event_count_refinancing_90d': 0.0,
    'event_event_intensity_acceleration_ratio': 0.0,
    'event_filing_complexity_index': 0.0,
    'event_information_asymmetry_proxy_score': 0.0,
    'event_max_event_severity_30d': 0.0,
    'event_news_sentiment_mean_7d': 0.0,
    'event_news_sentiment_min_7d': 0.0,
    'event_sum_impact_credit_90d': 0.0,
    'event_sum_impact_dividend_90d': 0.0,
    'event_sum_impact_liquidity_90d': 0.0,
    'event_time_since_last_high_severity_event': 999.0  # High value = long time since event
}

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

print("Fixing BA samples - adding 23 missing event features...")

# Get all BA trades
cursor.execute('SELECT id, entry_features_json FROM trades WHERE symbol="BA"')
ba_trades = cursor.fetchall()

print(f"Found {len(ba_trades)} BA samples to fix")

updated = 0
for trade_id, features_json in ba_trades:
    features = json.loads(features_json)

    # Check if already has event features
    if 'event_cross_source_confirmation_flag' in features:
        print(f"  Trade {trade_id[:8]}... already has event features, skipping")
        continue

    # Add event features
    features.update(EVENT_FEATURES)

    # Update database
    cursor.execute('UPDATE trades SET entry_features_json=? WHERE id=?',
                   (json.dumps(features), trade_id))
    updated += 1

conn.commit()
print(f"\nUpdated {updated} BA samples")

# Verify fix
cursor.execute('SELECT entry_features_json FROM trades WHERE symbol="BA" LIMIT 1')
sample = json.loads(cursor.fetchone()[0])
print(f"BA now has {len(sample)} features ✓")

# Check overall distribution
cursor.execute('SELECT entry_features_json FROM trades')
rows = cursor.fetchall()
feature_counts = {}
for row in rows:
    count = len(json.loads(row[0]))
    feature_counts[count] = feature_counts.get(count, 0) + 1

print(f"\nFinal feature count distribution:")
for count in sorted(feature_counts.keys()):
    print(f"  {count} features: {feature_counts[count]:5} samples")

if len(feature_counts) == 1:
    print("\n✓ SUCCESS! All samples now have consistent feature dimensions!")
else:
    print(f"\n✗ WARNING: Still have {len(feature_counts)} different feature counts")

conn.close()
