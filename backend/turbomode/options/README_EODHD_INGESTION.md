# EODHD Historical Options Ingestion

## Overview

Standalone one-time ingestion script that downloads 1-2 years of historical options chains from **EOD Historical Data (EODHD)** for all 233 symbols in `CORE_230.json`.

## Key Features

✅ **EODHD ONLY** - No Yahoo, no IBKR, no dependencies on live scanner
✅ **Standalone** - Runs independently, can be executed manually anytime
✅ **Append-Only** - Never re-downloads existing data
✅ **Resumable** - Checkpoint system allows resuming from interruptions
✅ **Deduplication** - Automatically skips duplicate records
✅ **Progress Tracking** - Metadata table tracks ingestion status per symbol

## Architecture

### Data Flow
```
CORE_230.json (233 symbols)
    ↓
EODHD API (historical options chains)
    ↓
options_universe.db (raw chains)
    ↓
options_training_history.db (outcomes for meta-learner)
```

### Databases

1. **options_universe.db** - Historical options chains
   - Table: `historical_options_chains` (strikes, Greeks, IV, volume, OI)
   - Table: `ingestion_metadata` (progress tracking)

2. **options_training_history.db** - Training outcomes
   - Table: `options_training_outcomes` (P&L, regime, features)
   - Table: `training_statistics` (aggregated stats)

## Prerequisites

### 1. EODHD API Key

Set the environment variable:

**Windows:**
```cmd
set EODHD_API_KEY=your_eodhd_api_key_here
```

**Linux/Mac:**
```bash
export EODHD_API_KEY=your_eodhd_api_key_here
```

### 2. Python Dependencies

Ensure you have:
- `requests` (for API calls)
- `sqlite3` (built-in)

## Usage

### Test Mode (Recommended First)

Test with 1 symbol and 7 days of data:

```bash
python C:\StockApp\test_eodhd_ingestion.py
```

This will:
- Download 7 days of AAPL options data
- Verify EODHD API key works
- Confirm database writes succeed

### Full Production Run

Download 1 year of data for all 233 symbols:

```bash
python C:\StockApp\backend\turbomode\Options\Ingestion\ingest_historical_options.py
```

**Expected Runtime:** ~2-4 hours (depending on API rate limits)

### Custom Configuration

Edit the script and modify:

```python
run_ingestion(
    lookback_days=365,   # Default: 1 year (set to 730 for 2 years)
    symbols_limit=None   # None = all symbols, or set to N for testing
)
```

## What Gets Stored

### Historical Options Chains Table

Each row represents a single option contract on a specific date:

| Field | Description |
|-------|-------------|
| `symbol` | Stock ticker (e.g., 'AAPL') |
| `snapshot_date` | Date this chain snapshot was taken |
| `contract_name` | Option contract name (e.g., 'AAPL260214C00150000') |
| `expiration_date` | Expiration date of the option |
| `strike` | Strike price |
| `option_type` | 'call' or 'put' |
| `last_price` | Last traded price |
| `bid` / `ask` | Bid/Ask prices |
| `volume` | Daily volume |
| `open_interest` | Open interest |
| `implied_volatility` | IV from EODHD |
| `delta` / `gamma` / `theta` / `vega` | Greeks |
| `underlying_price` | Stock price at snapshot time |
| `days_to_expiration` | Calculated DTE |
| `moneyness` | strike / underlying_price |

### Ingestion Metadata Table

Tracks progress for each symbol:

| Field | Description |
|-------|-------------|
| `symbol` | Stock ticker |
| `start_date` | Ingestion start date |
| `end_date` | Ingestion end date |
| `last_ingested_date` | Last successfully ingested date (checkpoint) |
| `total_records` | Total options records ingested |
| `status` | 'pending', 'in_progress', 'completed', 'failed' |
| `error_message` | Error details if failed |

## Resumability

If the ingestion is interrupted (Ctrl+C, network error, etc.):

1. **Automatic Resume:** Just re-run the script
2. **Checkpoint System:** Resumes from `last_ingested_date`
3. **No Re-downloads:** Already ingested dates are skipped

Example:
```
[INGEST] Resuming AAPL from 2026-01-15
```

## Rate Limiting

- Default: 1 second delay between API calls
- Configurable via `RATE_LIMIT_DELAY` in script
- Adjust if you hit EODHD rate limits

## Monitoring Progress

### Check Database Size
```bash
ls -lh C:\StockApp\backend\turbomode\Options\Data\options_universe.db
```

### Query Progress
```sql
SELECT symbol, status, total_records, last_ingested_date
FROM ingestion_metadata
WHERE status = 'completed'
ORDER BY total_records DESC;
```

### Count Total Records
```sql
SELECT COUNT(*) FROM historical_options_chains;
```

## Expected Output

### Successful Run
```
[2026-02-07 16:30:00] [INIT] Initializing databases...
[2026-02-07 16:30:01] [SYMBOLS] Loaded 233 symbols from CORE_230.json
[2026-02-07 16:30:01] [CONFIG] Date range: 2025-02-07 to 2026-02-07 (365 days)
[2026-02-07 16:30:01] [CONFIG] Symbols: 233
[2026-02-07 16:30:01] [PROGRESS] Symbol 1/233: AAPL
[2026-02-07 16:30:02] [INGEST] Fetching AAPL options for 2025-02-07
[2026-02-07 16:30:03] [INGEST] AAPL 2025-02-07: inserted 1250 options (total: 1250)
...
[2026-02-07 18:45:00] [INGEST] AAPL completed: 456,000 total records
[2026-02-07 18:45:01] [PROGRESS] Symbol 2/233: MSFT
...
```

### Error Handling
```
[2026-02-07 16:30:05] [INGEST] ERROR fetching AAPL 2025-02-10: 429 Rate Limit Exceeded
[2026-02-07 16:30:05] [INGEST] AAPL failed
```

## Troubleshooting

### Issue: "No EODHD_API_KEY environment variable"
**Solution:** Set `EODHD_API_KEY` before running

### Issue: "Rate limit exceeded"
**Solution:** Increase `RATE_LIMIT_DELAY` in script (e.g., 2.0 seconds)

### Issue: "Database locked"
**Solution:** Close any SQLite viewers/tools that have the database open

### Issue: Script crashes mid-run
**Solution:** Just re-run - it will resume from checkpoint

## Post-Ingestion

After successful ingestion, you have:

1. **options_universe.db** - Ready for meta-learner training
2. **Checkpoint metadata** - Shows what was ingested
3. **Full historical chains** - Greeks, IV, volume, OI for 1-2 years

## Next Steps

1. **Run Scanner:** Live Tradier scanner uses this historical data for intelligence
2. **Train Meta-Learner:** Use `options_training_outcomes` table
3. **Backtest Strategies:** Query historical chains for simulations

## Files Created

```
C:\StockApp\backend\eodhd_client.py
C:\StockApp\backend\turbomode\Options\Data\schema_options_universe.sql
C:\StockApp\backend\turbomode\Options\Data\schema_options_training_history.sql
C:\StockApp\backend\turbomode\Options\Data\options_universe.db (created on first run)
C:\StockApp\backend\turbomode\Options\Data\options_training_history.db (created on first run)
C:\StockApp\backend\turbomode\Options\Ingestion\ingest_historical_options.py
C:\StockApp\test_eodhd_ingestion.py
```

---

**Ready to run!** Start with `test_eodhd_ingestion.py` to verify everything works.
