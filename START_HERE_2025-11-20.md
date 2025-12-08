# START HERE - November 20, 2025 Morning

## 📖 Files to Read First (In Order)

### 1. **SESSION_NOTES_2025-11-19.md** ⭐
- Read this first for complete context
- What we fixed yesterday
- Current system status
- Known issues

### 2. **TODO_2025-11-20.md** ⭐
- Your prioritized task list
- Quick wins section at bottom
- Estimated time for each task

### 3. **Check Midnight Run Log** 🔍
```bash
# Find the latest log file
dir C:\StockApp\agents\logs\scanner_2025112*.log /O-D

# Open most recent one and check for:
# ✅ "SCAN COMPLETE"
# ✅ "Fusion Agent completed successfully"
# ❌ Any error messages
```

## 🎯 Quick Morning Checklist (5 minutes)

```bash
# 1. Check if midnight scanner ran
dir C:\StockApp\agents\logs\scanner_20251120_*.log

# 2. Check file timestamps (should be around midnight)
powershell -Command "(Get-Item 'C:\StockApp\agents\repository\scanner_output.json').LastWriteTime"
powershell -Command "(Get-Item 'C:\StockApp\agents\repository\fusion_output.json').LastWriteTime"

# 3. Quick verify Flask is running
powershell -Command "(Invoke-WebRequest -Uri 'http://127.0.0.1:5000/heatmap-data' -UseBasicParsing).StatusCode"
```

**Expected Results:**
- Log file exists from ~00:00 timestamp
- JSON files updated around midnight
- Flask returns HTTP 200

## 📁 Key Files Reference

### Configuration Files
- `C:\StockApp\backend\.env` - API keys (Polygon needed)
- `C:\StockApp\requirements.txt` - Python dependencies

### Agent Data (Check Timestamps)
- `C:\StockApp\agents\repository\scanner_output.json` - Scanner results
- `C:\StockApp\agents\repository\fusion_output.json` - Heat map data
- `C:\StockApp\agents\repository\indicators_agent\*.json` - Technical indicators
- `C:\StockApp\agents\repository\volume_agent\*.json` - Volume analysis

### Automation Files
- `C:\StockApp\run_scanner_scheduled.bat` - What runs at midnight
- `C:\StockApp\agents\logs\scanner_*.log` - Execution logs

### Agent Scripts (May Need to Run)
- `C:\StockApp\agents\comprehensive_scanner.py` - Main scanner
- `C:\StockApp\agents\indicators_agent.py` - Calculate indicators ⚠️ Need to run
- `C:\StockApp\agents\volume_agent.py` - Volume analysis ⚠️ Need to run
- `C:\StockApp\agents\tick_agent.py` - Real-time ticks ⚠️ Need to run
- `C:\StockApp\agents\fusion_agent.py` - Combine signals

### Documentation Created Yesterday
- `C:\StockApp\SESSION_NOTES_2025-11-19.md` - Full session notes
- `C:\StockApp\TODO_2025-11-20.md` - Task list
- `C:\StockApp\AUTOMATION_SETUP.md` - Automation guide
- `C:\StockApp\SCANNER_README.md` - Scanner documentation

## 🚀 First Action Items

### Priority 1: Verify Automation (5 min)
1. Check midnight log file exists
2. Verify no errors in log
3. Check JSON file timestamps

### Priority 2: Quick Fix (10 min)
1. Install lxml: `venv\Scripts\pip.exe install lxml`
2. Update requirements.txt

### Priority 3: Main Task (45 min)
Run full agent pipeline to get all 105 symbols analyzed:
```bash
cd C:\StockApp\agents
..\venv\Scripts\python.exe indicators_agent.py
..\venv\Scripts\python.exe volume_agent.py
..\venv\Scripts\python.exe fusion_agent.py
```

## 🔴 Known Issues from Yesterday

### 1. Limited Fusion Coverage
- Scanner found 105 candidates
- Only 12 have fusion data
- **Fix:** Run indicators_agent.py and volume_agent.py

### 2. Missing lxml Package
- Can't parse full S&P 500 list from Wikipedia
- Using fallback 55 stocks
- **Fix:** `pip install lxml`

### 3. Crypto Historical Data
- Many cryptos can't get Yahoo Finance data
- Need CoinGecko integration in indicators_agent
- **Fix:** Later priority

### 4. Polygon API Not Configured
- Using slower Yahoo Finance fallback
- **Fix:** Get API key from polygon.io

## 📊 Current Numbers (Last Night 10:04 PM)

- **Symbols Scanned:** 155 (55 stocks + 100 cryptos)
- **Passed Filters:** 105 candidates
- **Symbols with Fusion Data:** 12 (11.4%)
- **Strong Buy Signals:** 1 (XOM)
- **Hold Signals:** 2 (AMD, CVX)
- **Sell Signals:** 2 (META, LTC)
- **Strong Sell Signals:** 7 (ETH, SOL, DOGE, etc.)

**Goal Today:** Increase fusion data coverage from 12 → 100+ symbols

## ⚙️ System Status

- ✅ Windows Task Scheduler: "StockApp Scanner" created
- ✅ Dependencies installed: pycoingecko, polygon-api-client, ta, schedule
- ✅ Flask server: Running on port 5000
- ✅ Heat map page: http://127.0.0.1:5000/heatmap
- ⚠️ Full agent pipeline: Not automated yet

## 🎯 Success Criteria for Today

1. ✅ Midnight run completed without errors
2. ✅ lxml installed
3. ✅ All 105 candidates have indicator/volume data
4. ✅ Heat maps show 80+ symbols (instead of 12)
5. ✅ Automation updated to run full pipeline
6. ⏸️ (Stretch) Crypto historical data fix started

## 💡 Quick Commands Reference

### Start Flask Server
```bash
cd C:\StockApp\backend
..\venv\Scripts\python.exe api_server.py
```

### Run Scanner Manually
```bash
cd C:\StockApp\agents
..\venv\Scripts\python.exe comprehensive_scanner.py
```

### Run Full Agent Pipeline
```bash
cd C:\StockApp\agents
..\venv\Scripts\python.exe comprehensive_scanner.py
..\venv\Scripts\python.exe indicators_agent.py
..\venv\Scripts\python.exe volume_agent.py
..\venv\Scripts\python.exe tick_agent.py
..\venv\Scripts\python.exe fusion_agent.py
```

### Check Task Scheduler
```bash
schtasks /query /tn "StockApp Scanner" /fo LIST /v
```

### View Logs
```bash
# List all logs
dir C:\StockApp\agents\logs

# Read latest log
powershell -Command "Get-Content (Get-ChildItem C:\StockApp\agents\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName"
```

---

**Last Updated:** November 19, 2025 at 10:15 PM
**Next Session:** November 20, 2025 morning
**Estimated Time to Get Up to Speed:** 10-15 minutes reading
**Estimated Time for Priority Tasks:** 1-2 hours
