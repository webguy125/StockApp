# Yahoo Gap-Filling Strategy
## Temporary Help for Missing Tradier Bars

**Version:** 1.0
**Date:** February 5, 2026
**Author:** TurboMode System

---

## Overview

The Yahoo Gap-Filling Strategy is an enhancement to the 3-tier fallback system that allows Yahoo Finance to "help" fill in missing bars from Tradier without completely switching to Yahoo as the data source.

### Key Concept

**"Mind you not switch to yahoo just give yahoo a chance to help"** - User requirement

This means:
- ✅ Tradier remains the primary source (Tier 1)
- ✅ When Tradier has missing/invalid bars (NaN values), track those specific timestamps
- ✅ Make a secondary fetch from Yahoo to get just those missing bars
- ✅ Insert only the missing timestamps from Yahoo
- ❌ Do NOT switch entirely to Yahoo for the symbol

---

## Problem Being Solved

### Issue: Tradier Returns NaN Values

Tradier API sometimes returns bars with NaN (Not a Number) values in OHLCV fields:
- Market holidays (e.g., timestamp 1741219200)
- Data gaps during trading halts
- Corporate actions (splits, special dividends)
- API data quality issues

### Database Constraint

The database schema requires NOT NULL for OHLCV fields:

```sql
CREATE TABLE ohlcv (
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,      -- Cannot be NULL
    high REAL NOT NULL,      -- Cannot be NULL
    low REAL NOT NULL,       -- Cannot be NULL
    close REAL NOT NULL,     -- Cannot be NULL
    volume REAL NOT NULL,    -- Cannot be NULL
    PRIMARY KEY (symbol, timestamp)
)
```

### Previous Behavior

Before gap-filling:
1. Tradier returns 252 bars (250 valid + 2 with NaN)
2. Ingestion skips 2 invalid bars
3. **Result:** Database has 250 bars (missing 2 timestamps)

### New Behavior with Gap-Filling

With gap-filling:
1. Tradier returns 252 bars (250 valid + 2 with NaN)
2. Ingestion inserts 250 valid bars, tracks 2 skipped timestamps
3. **Yahoo attempts to fill:** Fetches same period from Yahoo
4. If Yahoo has valid data for those 2 timestamps, insert them
5. **Result:** Database has 252 bars (no missing data)

---

## Implementation

### File Location

`C:\StockApp\backend\turbomode\core_engine\ingest_master_market_data.py`

### Function Modified

`ingest_symbol_ohlcv()` - Lines 315-360

### Algorithm

```
1. Fetch data from Tradier (primary source)
2. For each bar:
   a. Normalize keys (Open -> open, etc.)
   b. Check for None or NaN in OHLCV fields
   c. If valid: INSERT into database
   d. If invalid: Track timestamp in skipped_timestamps[]

3. If skipped_timestamps is not empty:
   a. Log: "[GAP-FILL] {symbol}: Trying Yahoo for {count} missing bars"
   b. Temporarily disable Tradier (fetcher.use_tradier = False)
   c. Fetch same period from Yahoo (triggers Tier 3 fallback)
   d. For each Yahoo bar:
      i. Check if timestamp is in skipped_timestamps
      ii. If yes AND bar is valid: INSERT into database
      iii. Count successfully filled bars
   e. Restore Tradier setting (fetcher.use_tradier = True)
   f. Log results: "[GAP-FILL] {symbol}: Yahoo filled {filled} missing bars"

4. Commit all changes to database
```

### Code Implementation

```python
def ingest_symbol_ohlcv(
    conn: sqlite3.Connection,
    fetcher: HybridDataFetcher,
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
) -> int:
    """
    Fetch and upsert OHLCV data for a single symbol.
    Returns number of rows ingested.

    Strategy: Try Tradier first, then use Yahoo to fill gaps if Tradier has missing bars.
    """
    import math

    rows = fetch_ohlcv_for_symbol(fetcher, symbol, start, end)
    if not rows:
        return 0

    # Track skipped timestamps for gap-filling with Yahoo
    skipped_timestamps = []

    cur = conn.cursor()
    inserted = 0
    for row in rows:
        # Normalize Tradier keys (Open/High/Low/Close/Volume -> open/high/low/close/volume)
        row = {k.lower(): v for k, v in row.items()}

        # Skip invalid or incomplete bars (check for None and NaN)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [row.get('open'), row.get('high'), row.get('low'), row.get('close'), row.get('volume')]):
            logger.warning(f"[SKIP] Invalid bar for {symbol} on {row.get('timestamp')} (missing OHLCV) - will try Yahoo")
            skipped_timestamps.append(row.get('timestamp'))
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO ohlcv (
                symbol, timestamp, open, high, low, close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                int(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ),
        )
        inserted += 1

    # Gap-filling: If we skipped any bars, try Yahoo to fill them in
    if skipped_timestamps:
        logger.info(f"[GAP-FILL] {symbol}: Trying Yahoo for {len(skipped_timestamps)} missing bars from Tradier")

        # Temporarily disable Tradier to force Yahoo fallback
        original_use_tradier = fetcher.use_tradier
        fetcher.use_tradier = False

        try:
            yahoo_rows = fetch_ohlcv_for_symbol(fetcher, symbol, start, end)
            if yahoo_rows:
                filled = 0
                for row in yahoo_rows:
                    row = {k.lower(): v for k, v in row.items()}

                    # Only insert if this timestamp was skipped from Tradier
                    if row.get('timestamp') in skipped_timestamps:
                        # Check if Yahoo data is valid
                        if not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [row.get('open'), row.get('high'), row.get('low'), row.get('close'), row.get('volume')]):
                            cur.execute(
                                """
                                INSERT OR REPLACE INTO ohlcv (
                                    symbol, timestamp, open, high, low, close, volume
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    symbol,
                                    int(row["timestamp"]),
                                    float(row["open"]),
                                    float(row["high"]),
                                    float(row["low"]),
                                    float(row["close"]),
                                    float(row["volume"]),
                                ),
                            )
                            filled += 1
                            inserted += 1

                if filled > 0:
                    logger.info(f"[GAP-FILL] {symbol}: Yahoo filled {filled} missing bars")
                else:
                    logger.warning(f"[GAP-FILL] {symbol}: Yahoo could not fill any missing bars")
        finally:
            # Restore Tradier setting
            fetcher.use_tradier = original_use_tradier

    conn.commit()
    return inserted
```

---

## Log Examples

### Successful Gap-Filling

```
[1/233] AAPL: Fetching data...
[TRADIER REST] Fetched 252 rows for AAPL (daily)
[HYBRID] [TIER 1] AAPL fetched from Tradier (252 rows)
[SKIP] Invalid bar for AAPL on 1741219200 (missing OHLCV) - will try Yahoo
[GAP-FILL] AAPL: Trying Yahoo for 1 missing bars from Tradier
[HYBRID] [TIER 3] AAPL fetched from Yahoo (252 rows)
[GAP-FILL] AAPL: Yahoo filled 1 missing bars
[1/233] AAPL: 252 rows
```

### Yahoo Cannot Fill Gap

```
[50/233] XYZ: Fetching data...
[TRADIER REST] Fetched 250 rows for XYZ (daily)
[HYBRID] [TIER 1] XYZ fetched from Tradier (250 rows)
[SKIP] Invalid bar for XYZ on 1741219200 (missing OHLCV) - will try Yahoo
[GAP-FILL] XYZ: Trying Yahoo for 1 missing bars from Tradier
[HYBRID] [TIER 3] XYZ fetched from Yahoo (250 rows)
[GAP-FILL] XYZ: Yahoo could not fill any missing bars
[50/233] XYZ: 250 rows
```

### No Missing Bars (No Gap-Filling Needed)

```
[100/233] MSFT: Fetching data...
[TRADIER REST] Fetched 252 rows for MSFT (daily)
[HYBRID] [TIER 1] MSFT fetched from Tradier (252 rows)
[100/233] MSFT: 252 rows
```

---

## Benefits

### Data Completeness
- Maximizes data coverage by combining Tradier + Yahoo
- Reduces missing bars in the database
- Improves backtest accuracy

### Data Quality Prioritization
- Still uses Tradier as primary source (higher quality)
- Yahoo only fills specific gaps (not full dataset)
- Maintains 3-tier fallback priority

### Performance
- Only makes Yahoo request when necessary (has missing bars)
- Doesn't slow down ingestion for symbols with complete Tradier data
- Efficient timestamp-based filtering

### Logging Transparency
- Clear log messages show which bars were skipped
- Shows how many bars Yahoo filled
- Easy to audit data source for each bar

---

## Limitations

### Yahoo May Also Have Missing Data

Yahoo Finance is not guaranteed to have data for bars that Tradier is missing. Possible scenarios:
1. Market holiday: Both Tradier and Yahoo will have no bar
2. Trading halt: Both may have gaps
3. Symbol delisted: Yahoo may have stale data

### Performance Impact

For symbols with many missing bars:
- Extra fetch from Yahoo adds latency (~1-2 seconds per symbol)
- Rate limiting: Yahoo has ~1 req/sec soft limit
- For 233 symbols with 10 missing bars each: adds ~233 seconds to ingestion

### Timestamp Precision

Must match timestamps exactly between Tradier and Yahoo:
- Both use Unix epoch seconds
- Both use midnight UTC for daily bars
- Timestamp drift (even 1 second) will cause gap-filling to fail

---

## Testing

### Test Case 1: Symbol with Missing Bars

**Setup:**
- Symbol: ABT (known to have missing bars from Tradier)
- Period: 5 days
- Expected: Tradier missing 1 bar, Yahoo fills it

**Run:**
```bash
cd C:\StockApp
python backend/turbomode/core_engine/ingest_master_market_data.py --period 5d --symbols ABT
```

**Expected Output:**
```
[SKIP] Invalid bar for ABT on 1741219200 (missing OHLCV) - will try Yahoo
[GAP-FILL] ABT: Trying Yahoo for 1 missing bars from Tradier
[GAP-FILL] ABT: Yahoo filled 1 missing bars
```

### Test Case 2: Symbol with Complete Data

**Setup:**
- Symbol: AAPL (usually has complete Tradier data)
- Period: 5 days
- Expected: No gap-filling needed

**Run:**
```bash
cd C:\StockApp
python backend/turbomode/core_engine/ingest_master_market_data.py --period 5d --symbols AAPL
```

**Expected Output:**
```
[1/1] AAPL: 5 rows
```
(No [GAP-FILL] messages)

### Test Case 3: Full CORE_230 Ingestion

**Setup:**
- Symbols: All 233 symbols from CORE_230.json
- Period: 5 days
- Expected: Multiple symbols trigger gap-filling

**Run:**
```bash
cd C:\StockApp
python backend/turbomode/core_engine/ingest_master_market_data.py --period 5d
```

**Verify:**
```bash
# Count gap-fill attempts
findstr /C:"[GAP-FILL]" backend\logs\task_1_master_market_data_ingestion.log

# Check success rate
findstr /C:"Yahoo filled" backend\logs\task_1_master_market_data_ingestion.log
```

---

## Monitoring

### Key Metrics

1. **Gap-Fill Rate**: Percentage of symbols requiring gap-filling
2. **Gap-Fill Success Rate**: Percentage of missing bars Yahoo successfully filled
3. **Performance Impact**: Additional time spent on gap-filling

### Log Analysis

```bash
# Count symbols with missing bars
findstr /C:"[SKIP]" task_1_master_market_data_ingestion.log | find /C "Invalid bar"

# Count gap-fill attempts
findstr /C:"[GAP-FILL]" task_1_master_market_data_ingestion.log | find /C "Trying Yahoo"

# Count successful fills
findstr /C:"[GAP-FILL]" task_1_master_market_data_ingestion.log | find /C "Yahoo filled"

# Count failed fills
findstr /C:"[GAP-FILL]" task_1_master_market_data_ingestion.log | find /C "could not fill"
```

### Database Verification

```sql
-- Check completeness for a symbol
SELECT
    symbol,
    COUNT(*) as bar_count,
    MIN(timestamp) as first_bar,
    MAX(timestamp) as last_bar
FROM ohlcv
WHERE symbol = 'ABT'
GROUP BY symbol;

-- Find symbols with fewer bars than expected
SELECT
    symbol,
    COUNT(*) as bar_count
FROM ohlcv
WHERE timestamp >= (strftime('%s', 'now') - 5*86400)  -- Last 5 days
GROUP BY symbol
HAVING bar_count < 5;
```

---

## Troubleshooting

### Issue: Gap-Filling Not Triggering

**Symptoms:**
- No [GAP-FILL] messages in logs
- Database still has missing bars

**Possible Causes:**
1. Tradier not returning any NaN values (expected behavior)
2. NaN check not detecting invalid bars
3. skipped_timestamps list not being populated

**Solution:**
Check the [SKIP] messages. If you see [SKIP] but no [GAP-FILL], there's a logic error.

---

### Issue: Yahoo Cannot Fill Gaps

**Symptoms:**
- [GAP-FILL] messages show "could not fill any missing bars"
- Database still has missing bars

**Possible Causes:**
1. Yahoo also missing data for those timestamps (market holiday)
2. Timestamp mismatch between Tradier and Yahoo
3. Yahoo data also has NaN values

**Solution:**
This is expected for market holidays. Yahoo cannot provide data that doesn't exist.

---

### Issue: Duplicate Bars

**Symptoms:**
- Database has duplicate timestamps for same symbol
- Constraint violation errors

**Possible Causes:**
1. INSERT OR REPLACE not working correctly
2. Timestamp format mismatch (int vs float)

**Solution:**
The schema has PRIMARY KEY (symbol, timestamp), so duplicates should be impossible. Check if timestamps are being cast to int correctly.

---

## Future Enhancements

### Potential Improvements

1. **Smart Gap Detection**: Identify market holidays vs real data gaps
2. **Multiple Fallback Sources**: Try IBKR before Yahoo for gap-filling
3. **Caching**: Store Yahoo data for common missing timestamps
4. **Async Gap-Filling**: Fill gaps in background after main ingestion
5. **Gap-Fill Report**: Generate daily report of filled vs unfilled gaps

### Performance Optimization

1. **Batch Gap-Filling**: Collect all missing timestamps across symbols, fetch from Yahoo once
2. **Parallel Gap-Filling**: Use threading to fill gaps for multiple symbols simultaneously
3. **Conditional Gap-Filling**: Only try Yahoo if missing bar is recent (not old data)

---

## Summary

The Yahoo Gap-Filling Strategy provides a "best of both worlds" approach:
- ✅ Tradier provides high-quality real-time data (Tier 1)
- ✅ Yahoo fills in specific missing bars (temporary helper)
- ✅ Maintains 3-tier fallback architecture
- ✅ Maximizes data completeness
- ✅ Transparent logging and monitoring

**Key Principle:** "Mind you not switch to yahoo just give yahoo a chance to help"

Yahoo is a helper, not a replacement. Tradier remains the primary source.

---

**Version History:**
- v1.0 (2026-02-05): Initial implementation

**Author:** TurboMode System
**Last Updated:** February 5, 2026
