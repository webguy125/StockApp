"""
Test date formatting issue
"""
from datetime import datetime

# Check what JavaScript's new Date().toISOString().split('T')[0] would produce
today = datetime.now()
today_iso = today.isoformat().split('T')[0]
print(f"Today's date: {today_iso}")
print(f"Year: {today.year}, Month: {today.month}, Day: {today.day}")

# The data from Coinbase shows dates like "2025-10-22"
# But today is actually "2024-10-23"
print("\nCoinbase returned dates in 2025!")
print("That's why the candles are overlapping - wrong year!")