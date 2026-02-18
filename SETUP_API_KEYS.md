# API Keys Setup Guide

## Quick Setup (Recommended)

### 1. Edit the Config File

Open this file in a text editor:
```
C:\StockApp\config\api_keys.json
```

### 2. Add Your EODHD API Key

Replace `your_eodhd_api_key_here` with your actual key:

```json
{
  "EODHD_API_KEY": "YOUR_ACTUAL_KEY_HERE",
  "TRADIER_API_KEY": "pplYfsA91vM8AAFoSmLB4naoaDa5",
  "ALPHA_VANTAGE_API_KEY": "your_alpha_vantage_key_if_needed"
}
```

### 3. Save and Test

Run the test to verify it works:
```bash
python test_eodhd_ingestion.py
```

---

## What Gets Loaded

The system will automatically load API keys in this priority order:

1. **Config file** (recommended): `C:\StockApp\config\api_keys.json`
2. **Environment variable** (fallback): `EODHD_API_KEY`

---

## Security Notes

- ✅ `api_keys.json` should be in `.gitignore` (already done)
- ✅ Never commit API keys to git
- ✅ `api_keys.json.example` is safe to commit (contains placeholders)

---

## Testing

After setting up your key, test with:

```bash
# Simple test (1 symbol, 7 days)
python test_eodhd_ingestion.py

# Or the AAPL-specific test
python test_eodhd_aapl.py
```

If it works, you'll see:
```
Total options records: 5000+
Days with data: 5-7
Unique expirations: 15-25
```

---

## Full Production Run

Once testing works, run the full ingestion:

```bash
python backend\turbomode\Options\Ingestion\ingest_historical_options.py
```

This will download 1 year of historical options data for all 233 symbols (2-4 hours).

---

## Troubleshooting

### "No API key provided"
- Check that `api_keys.json` exists in `C:\StockApp\config\`
- Check that your key is correctly entered (no quotes errors, no extra spaces)
- Verify the JSON is valid (use a JSON validator)

### "Rate limit exceeded"
- EODHD has rate limits on their API plans
- Increase `RATE_LIMIT_DELAY` in the ingestion script (default: 1.0 seconds)

### "Invalid API key"
- Verify your EODHD subscription is active
- Check you're using the correct key from your EODHD account

---

**Ready!** Edit `config/api_keys.json` and run `python test_eodhd_ingestion.py`
