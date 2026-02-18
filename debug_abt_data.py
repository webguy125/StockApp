"""Debug script to see what Tradier returns for ABT"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / "backend" / ".env")

# Import Tradier client
from tradier_websocket.tradier_unified_scheduler_rest_client import get_tradier_scheduler_client

client = get_tradier_scheduler_client()

print("Fetching ABT data from Tradier...")
df = client.get_historical_data('ABT', interval='daily')

print(f"\nTotal rows: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

# Check for null values
print(f"\nNull value counts:")
print(df.isnull().sum())

# Check specific problematic rows
if df.isnull().any().any():
    print(f"\nRows with null values:")
    null_rows = df[df.isnull().any(axis=1)]
    print(null_rows)
