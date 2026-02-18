# Quick Start: EODHD Historical Options Ingestion

## 📋 Prerequisites
- EODHD API account with historical options data access
- Your EODHD API key

---

## 🚀 3 Simple Steps

### Step 1: Add Your API Key

Edit this file:
```
C:\StockApp\config\api_keys.json
```

Change this line:
```json
"EODHD_API_KEY": "your_eodhd_api_key_here"
```

To your actual key:
```json
"EODHD_API_KEY": "abc123xyz456..."
```

Save the file.

---

### Step 2: Run Test

Open terminal in `C:\StockApp` and run:

```bash
python test_eodhd_ingestion.py
```

**Expected output:**
```
✓ Databases created
✓ Loaded 233 symbols
✓ Testing with 1 symbol (AAPL)
✓ Downloaded 7 days of data
✓ Total records: 5000-10000
```

If you see errors, check `SETUP_API_KEYS.md` for troubleshooting.

---

### Step 3: Run Full Ingestion (Optional)

For production with all 233 symbols and 1 year of data:

```bash
python backend\turbomode\Options\Ingestion\ingest_historical_options.py
```

**Runtime:** 2-4 hours
**Expected data:** ~50-100 million option records
**Database size:** 5-10 GB

---

## ✅ What You Get

After successful ingestion:

- **options_universe.db** - Historical chains with Greeks, IV, volume, OI
- **options_training_history.db** - Training outcomes for meta-learner
- **Resumable** - Can interrupt and resume anytime
- **Append-only** - Never re-downloads existing data

---

## 🔍 Verify It Worked

Check your database:

```bash
python -c "import sqlite3; conn = sqlite3.connect('backend/turbomode/Options/Data/options_universe.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM historical_options_chains'); print(f'Total records: {cursor.fetchone()[0]:,}')"
```

---

## 📚 More Info

- **Full documentation:** `backend/turbomode/Options/README_EODHD_INGESTION.md`
- **API setup help:** `SETUP_API_KEYS.md`
- **Complete architecture:** `STEP_4_COMPLETE.md`

---

**That's it!** Just edit the config file and run the test.
