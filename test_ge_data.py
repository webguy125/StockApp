import yfinance as yf
import pandas as pd

# Test fetching GE data
data = yf.download('GE', period='5d')

print("=" * 60)
print("ORIGINAL DATA:")
print("=" * 60)
print("Columns:", data.columns)
print("\nIs MultiIndex?", isinstance(data.columns, pd.MultiIndex))

if isinstance(data.columns, pd.MultiIndex):
    print("\nMultiIndex levels:")
    for i, level in enumerate(data.columns.levels):
        print(f"  Level {i}: {list(level)}")

    print("\nGetting level 0:")
    data.columns = data.columns.get_level_values(0)
    print("New columns:", data.columns)

print("\n" + "=" * 60)
print("PROCESSED DATA:")
print("=" * 60)
print("\nLast 3 rows:")
print(data.tail(3))

print("\nLast Close value:")
print(data['Close'].iloc[-1])
print(f"Type: {type(data['Close'].iloc[-1])}")

# Test the exact same logic as api_server.py
data.reset_index(inplace=True)
data = data.rename(columns={'Datetime': 'Date'} if 'Datetime' in data.columns else {'Date': 'Date'})
data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
result = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].fillna("").to_dict(orient="records")

print("\n" + "=" * 60)
print("FINAL JSON OUTPUT (last 2 records):")
print("=" * 60)
for record in result[-2:]:
    print(record)
