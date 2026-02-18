# EODHD Historical Options Ingestion

## Overview

Optimized system for ingesting historical options chains from EOD Historical Data (EODHD) API.

**Performance**: 43,984 records in 56 seconds (786 records/sec) for 3 symbols over 90 days.

## Architecture

### Key Optimizations

1. **Symbol-Level Parallelism**
   - 4-8 concurrent workers processing symbols in parallel
   - Each worker gets exclusive access to a subset of symbols
   - Deterministic processing order (symbols sorted alphabetically)

2. **Batched Database Writes**
   - Insert 100-2000 records per transaction
   - Short-lived transactions to minimize lock contention
   - SQLite WAL mode for concurrent writes

3. **Rate Limiting**
   - Global token bucket limiter at 5 requests/second
   - Prevents API throttling across all workers
   - Thread-safe implementation

4. **Retry Logic**
   - Exponential backoff on 5xx errors (1s, 2s, 4s)
   - Max 3 retries per request
   - Hard fail on 4xx errors (400, 401, 403, 404)

5. **Date-Driven Processing**
   - Process full options chain for each trading date
   - Skip weekends automatically
   - Insert all expirations/strikes for each date in batch

## Scripts

### ingest_historical_options_parallel.py

**Production-ready parallel ingestion script**

```bash
# Test mode: 3 symbols, 90 days, 4 workers
python ingest_historical_options_parallel.py --test

# Full ingestion: all 233 symbols, 90 days, 6 workers
python ingest_historical_options_parallel.py --workers 6

# Custom lookback: 180 days
python ingest_historical_options_parallel.py --lookback 180 --workers 8
```

**Arguments:**
- `--test`: Test mode (first 3 symbols only)
- `--workers N`: Number of parallel workers (4-8, default: 4)
- `--lookback N`: Days to look back (default: 90)

**Features:**
- Symbol-level parallelism (4-8 workers)
- Date-driven batched processing
- Rate limiting at 5 req/sec
- SQLite WAL mode
- Exponential backoff retry
- Real-time progress logging

**Output:**
```
[2026-02-07 18:49:24] [INFO] [SpawnProcess-1] [WORKER_1] AAPL 2025-09-10: 388 records (cumulative: 388)
[2026-02-07 18:49:25] [INFO] [SpawnProcess-2] [WORKER_2] ABBV 2025-09-11: 207 records (cumulative: 375)
...
Runtime: 55.90 seconds (0.93 minutes)
Total records: 43,984
Records/second: 786.8
```

### ingest_historical_options.py

**Legacy single-threaded script** (superseded by parallel version)

Simpler implementation without parallelism. Use for debugging or single-symbol ingestion.

```bash
python ingest_historical_options.py
```

### ingest_historical_options_optimized.py

**Experimental expiration-driven approach** (not recommended)

Attempted to optimize by fetching expiration ladder first, then filtering. Didn't provide benefit over date-driven approach.

## Database Schema

### historical_options_chains

```sql
CREATE TABLE historical_options_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    contract_name TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,  -- 'call' or 'put'

    -- Pricing
    last_price REAL,
    bid REAL,
    ask REAL,

    -- Volume/Interest
    volume INTEGER,
    open_interest INTEGER,

    -- Greeks
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,

    -- Derived Fields
    underlying_price REAL,
    days_to_expiration INTEGER,
    moneyness REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, snapshot_date, contract_name)
);
```

**Indexes:**
```sql
CREATE INDEX idx_symbol_date ON historical_options_chains(symbol, snapshot_date);
CREATE INDEX idx_symbol_expiration ON historical_options_chains(symbol, expiration_date);
CREATE INDEX idx_snapshot_date ON historical_options_chains(snapshot_date);
```

### ingestion_metadata

Tracks ingestion progress for checkpoint/resume (legacy, not used by parallel script).

## Configuration

### API Keys

Set EODHD API key in `config/api_keys.json`:

```json
{
  "EODHD_API_KEY": "your_key_here"
}
```

Or set environment variable:
```bash
export EODHD_API_KEY=your_key_here
```

### Symbol Universe

Edit `config/symbols/CORE_230.json` to customize symbol list:

```json
[
  {"ticker": "AAPL", "name": "Apple Inc."},
  {"ticker": "MSFT", "name": "Microsoft Corporation"},
  ...
]
```

Symbols are processed in alphabetical order for determinism.

### Date Range

Default: Last 90 days (ending 60 days ago to account for EODHD data delay)

- Start date: `now - 150 days`
- End date: `now - 60 days`

Customize with `--lookback` parameter.

## Performance Tuning

### Workers

- **4 workers**: Good for most use cases, balanced CPU/network
- **6 workers**: Faster for large symbol lists, higher network usage
- **8 workers**: Maximum parallelism, may hit rate limits

### Batch Size

Default: 2000 records per transaction

- Larger batches = fewer commits = faster inserts
- Smaller batches = less lock contention = better concurrency

Adjust `BATCH_SIZE_MAX` in script if needed.

### Rate Limiting

Default: 5 requests/second (global across all workers)

- Higher rate may trigger API throttling
- Lower rate = slower ingestion but safer

Adjust `RATE_LIMIT_RPS` in script if needed.

### SQLite Tuning

Current settings:
```sql
PRAGMA journal_mode=WAL;       -- Write-Ahead Logging for concurrency
PRAGMA synchronous=NORMAL;     -- Balance safety/speed
PRAGMA cache_size=-64000;      -- 64MB cache
PRAGMA temp_store=MEMORY;      -- In-memory temp tables
```

## Monitoring

### Real-time Progress

Each worker logs:
- Symbol being processed
- Date being processed
- Records inserted for that date
- Cumulative record count

Example:
```
[2026-02-07 18:49:24] [INFO] [SpawnProcess-1] [WORKER_1] AAPL 2025-09-10: 388 records (cumulative: 388)
```

### Final Summary

At completion:
```
Runtime: 55.90 seconds (0.93 minutes)
Symbols processed: 3
Success: 3
Errors: 0
Total records: 43,984
Records/second: 786.8
```

### Database Verification

```sql
-- Total records
SELECT COUNT(*) FROM historical_options_chains;

-- Records by symbol
SELECT symbol, COUNT(*) as records
FROM historical_options_chains
GROUP BY symbol
ORDER BY records DESC;

-- Date coverage
SELECT symbol,
       MIN(snapshot_date) as first_date,
       MAX(snapshot_date) as last_date,
       COUNT(DISTINCT snapshot_date) as trading_days
FROM historical_options_chains
GROUP BY symbol;

-- Expirations per symbol
SELECT symbol, COUNT(DISTINCT expiration_date) as expirations
FROM historical_options_chains
GROUP BY symbol;
```

## Troubleshooting

### No records inserted

**Symptom**: Script completes but 0 records inserted

**Causes**:
1. EODHD API key invalid or expired
2. Date range too recent (EODHD has ~60 day delay)
3. Symbol not found in EODHD database

**Fix**:
- Verify API key in `config/api_keys.json`
- Check date range (must be historical, ending 60+ days ago)
- Test with known symbol (AAPL, MSFT)

### Rate limit errors

**Symptom**: 429 Too Many Requests errors

**Causes**:
1. Too many workers
2. Rate limit set too high
3. Other processes using same API key

**Fix**:
- Reduce workers to 4
- Lower `RATE_LIMIT_RPS` to 3-4
- Check for other EODHD API usage

### Database locked errors

**Symptom**: "database is locked" errors

**Causes**:
1. Too many concurrent writes
2. WAL mode not enabled
3. Long-running transactions

**Fix**:
- Verify WAL mode: `PRAGMA journal_mode;` should return `wal`
- Reduce batch size
- Increase timeout in `sqlite3.connect(timeout=30.0)`

### Memory errors

**Symptom**: Out of memory or slow performance

**Causes**:
1. Too many workers
2. Large batches accumulating in memory
3. SQLite cache too large

**Fix**:
- Reduce workers to 4
- Reduce `BATCH_SIZE_MAX` to 1000
- Lower cache size: `PRAGMA cache_size=-32000;` (32MB)

## Future Enhancements

1. **Delta Ingestion**: Only fetch new dates since last run
2. **Error Recovery**: Automatic retry on transient failures
3. **Data Validation**: Check for gaps, outliers, corrupt data
4. **Compression**: Store older data in compressed format
5. **Partitioning**: Split database by date ranges for faster queries

## References

- [EODHD API Documentation](https://eodhd.com/financial-apis/stock-options-data/)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [Python multiprocessing](https://docs.python.org/3/library/concurrent.futures.html)
