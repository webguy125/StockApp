import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

# Check events table
cursor.execute('SELECT COUNT(*) FROM events')
event_count = cursor.fetchone()[0]
print(f"Events in database: {event_count}")

# Check a sample trade's event features
cursor.execute('SELECT entry_features_json FROM trades LIMIT 1')
features = json.loads(cursor.fetchone()[0])

event_features = {k:v for k,v in features.items() if k.startswith('event_')}
print(f"\nEvent features in sample trade ({len(event_features)}):")
for k,v in event_features.items():
    print(f"  {k}: {v}")

# Check if they're all the same value
unique_values = set(event_features.values())
print(f"\nUnique values across event features: {unique_values}")

if len(unique_values) <= 2:
    print("\n⚠️  WARNING: Event features are mostly/all the same value (likely all zeros)!")
    print("   These are adding NOISE, not signal!")

conn.close()
