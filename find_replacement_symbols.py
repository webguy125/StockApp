"""
Find replacement symbols for BMBL, PLTR, SNOW
Need:
- 2 Technology Mid-Cap stocks (replace PLTR, SNOW)
- 1 Communication Services Small-Cap stock (replace BMBL)
"""

import yfinance as yf
from datetime import datetime, timedelta
import time

# Define candidates
TECH_MID_CAP_CANDIDATES = [
    'ANET',   # Arista Networks - Networking
    'PANW',   # Palo Alto Networks - Cybersecurity
    'ADSK',   # Autodesk - Software
    'CDNS',   # Cadence Design - Software
    'SNPS',   # Synopsys - Software
    'TEAM',   # Atlassian - Software
    'DDOG',   # Datadog - Monitoring
    'NET',    # Cloudflare - Cloud/Security
    'ZS',     # Zscaler - Cybersecurity
    'FTNT',   # Fortinet - Cybersecurity
    'MRVL',   # Marvell - Semiconductors
    'MU',     # Micron - Memory/Semiconductors
    'AMAT',   # Applied Materials - Semiconductor Equipment
    'LRCX',   # Lam Research - Semiconductor Equipment
    'KLAC',   # KLA Corporation - Semiconductor Equipment
    'ADI',    # Analog Devices - Semiconductors
    'TXN',    # Texas Instruments - Semiconductors
    'QCOM',   # Qualcomm - Semiconductors
]

COMM_SERVICES_SMALL_CAP_CANDIDATES = [
    'SIRI',   # SiriusXM - Audio Entertainment
    'MSG',    # Madison Square Garden Sports
    'MSGE',   # Madison Square Garden Entertainment
    'MSGS',   # Madison Square Garden Sports
    'GETY',   # Getty Images
    'SSTK',   # Shutterstock - Stock Media
    'ZD',     # Ziff Davis - Digital Media
    'QUOT',   # Quotient Technology - Digital Marketing
    'IHRT',   # iHeartMedia - Radio/Audio
    'STGW',   # Stagwell - Marketing Services
    'FUBO',   # FuboTV - Streaming
    'WBD',    # Warner Bros Discovery - Entertainment
]

def check_symbol_history(symbol: str, min_years: float = 6.5):
    """Check if symbol has enough history"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(7 * 365 + 30))

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(df) == 0:
            return None, None, "No data"

        first_date = df.index[0]
        first_date_naive = first_date.replace(tzinfo=None) if hasattr(first_date, 'tzinfo') else first_date
        years_available = (end_date - first_date_naive).days / 365.25

        # Get current market cap
        info = ticker.info
        market_cap = info.get('marketCap', 0)

        return years_available, market_cap, len(df)

    except Exception as e:
        return None, None, str(e)[:50]

print("=" * 80)
print("FINDING REPLACEMENT SYMBOLS")
print("=" * 80)

# Find Tech Mid-Cap replacements
print("\n" + "=" * 80)
print("TECHNOLOGY MID-CAP CANDIDATES (need 2 replacements for PLTR, SNOW)")
print("=" * 80)
print("Looking for: $10B-$50B market cap, 6.5+ years history, liquid stocks")
print()

tech_valid = []
for symbol in TECH_MID_CAP_CANDIDATES:
    print(f"{symbol:6s} ... ", end="", flush=True)
    years, market_cap, days = check_symbol_history(symbol)

    if years is None:
        print(f"SKIP - {days}")
        continue

    if years >= 6.5:
        cap_billions = market_cap / 1_000_000_000 if market_cap else 0
        print(f"OK - {years:.1f} years, ${cap_billions:.1f}B cap, {days:,} days")
        tech_valid.append((symbol, years, cap_billions, days))
    else:
        print(f"SHORT - Only {years:.1f} years")

    time.sleep(0.3)  # Rate limiting

print(f"\nFound {len(tech_valid)} valid tech mid-cap candidates")

# Find Comm Services Small-Cap replacement
print("\n" + "=" * 80)
print("COMMUNICATION SERVICES SMALL-CAP CANDIDATES (need 1 replacement for BMBL)")
print("=" * 80)
print("Looking for: $2B-$10B market cap, 6.5+ years history, liquid stocks")
print()

comm_valid = []
for symbol in COMM_SERVICES_SMALL_CAP_CANDIDATES:
    print(f"{symbol:6s} ... ", end="", flush=True)
    years, market_cap, days = check_symbol_history(symbol)

    if years is None:
        print(f"SKIP - {days}")
        continue

    if years >= 6.5:
        cap_billions = market_cap / 1_000_000_000 if market_cap else 0
        print(f"OK - {years:.1f} years, ${cap_billions:.1f}B cap, {days:,} days")
        comm_valid.append((symbol, years, cap_billions, days))
    else:
        print(f"SHORT - Only {years:.1f} years")

    time.sleep(0.3)  # Rate limiting

print(f"\nFound {len(comm_valid)} valid comm services small-cap candidates")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\nTechnology Mid-Cap (pick 2):")
if tech_valid:
    # Sort by years of history (more is better)
    tech_valid.sort(key=lambda x: x[1], reverse=True)
    for symbol, years, cap, days in tech_valid[:5]:  # Show top 5
        print(f"  {symbol:6s} - {years:.1f} years, ${cap:.1f}B cap, {days:,} days")
else:
    print("  No valid candidates found")

print("\nCommunication Services Small-Cap (pick 1):")
if comm_valid:
    # Sort by years of history
    comm_valid.sort(key=lambda x: x[1], reverse=True)
    for symbol, years, cap, days in comm_valid[:5]:  # Show top 5
        print(f"  {symbol:6s} - {years:.1f} years, ${cap:.1f}B cap, {days:,} days")
else:
    print("  No valid candidates found")

print("\n" + "=" * 80)
print("SUGGESTED REPLACEMENTS")
print("=" * 80)
if tech_valid and comm_valid:
    print(f"\nReplace PLTR with: {tech_valid[0][0]} (Technology Mid-Cap)")
    print(f"Replace SNOW with: {tech_valid[1][0]} (Technology Mid-Cap)")
    print(f"Replace BMBL with: {comm_valid[0][0]} (Communication Services Small-Cap)")
else:
    print("\nNot enough valid candidates found in all categories")
