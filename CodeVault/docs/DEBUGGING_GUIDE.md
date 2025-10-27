# 🔍 Debugging Guide for "Only 1 Candle" Issue

## Issue Overview
When loading charts with 1 year / 1 day settings, sometimes only 1 candle appears instead of ~252 trading days.

---

## ✅ Fixes Applied

### **Frontend Logging Added**
**File**: `frontend/js/tos-app.js`

Added comprehensive logging in the `reloadChart()` method:

```javascript
// Logs the exact URL being fetched
console.log(`Fetching chart data: ${fetchUrl}`);

// Logs how many data points were received
console.log(`Received ${data.length} data points for ${this.currentSymbol}`);

// Warns if only 1 candle received
if (data.length === 1) {
  console.warn('Only 1 candle received. Period:', period, 'Interval:', interval);
}
```

### **Backend Logging Added**
**File**: `backend/api_server.py`

Added logging to the `/data/<symbol>` endpoint:

```python
# Logs what's being requested from yfinance
print(f"📊 Fetching {symbol}: period={period}, interval={interval}, kwargs={kwargs}")

# Logs how many rows yfinance returned
print(f"✅ Downloaded {len(data)} rows for {symbol}")

# Logs how many candles are being sent to frontend
print(f"📤 Returning {len(result)} candles for {symbol}")
```

---

## 🔍 How to Debug

### **Step 1: Check Browser Console**
1. Open Developer Tools (F12)
2. Go to Console tab
3. Load a stock symbol
4. Look for these log messages:

```
Fetching chart data: /data/AAPL?interval=1d&period=1y
Received 1 data points for AAPL     ← This tells you the problem!
Only 1 candle received. Period: 1y Interval: 1d
```

### **Step 2: Check Flask Server Console**
In your terminal where Flask is running, you'll see:

```
📊 Fetching AAPL: period=1y, interval=1d, kwargs={'interval': '1d', 'period': '1y'}
✅ Downloaded 252 rows for AAPL     ← If yfinance got 252 rows but frontend got 1...
📤 Returning 252 candles for AAPL   ← ...there's a network or JSON parsing issue
```

**OR**

```
📊 Fetching AAPL: period=1y, interval=1d, kwargs={'interval': '1d', 'period': '1y'}
✅ Downloaded 1 rows for AAPL       ← If yfinance only returned 1 row, it's a yfinance issue
📤 Returning 1 candles for AAPL
```

---

## 🐛 Possible Root Causes

### **Scenario A: Backend Returns 252, Frontend Gets 1**
**Symptoms:**
- Backend logs show "Downloaded 252 rows"
- Frontend logs show "Received 1 data points"

**Likely Causes:**
1. JSON parsing error in frontend
2. Network response truncation
3. Flask response size limit
4. Browser memory issue

**How to Check:**
1. Open Browser DevTools → Network tab
2. Find the request to `/data/AAPL?interval=1d&period=1y`
3. Click on it and check the Response tab
4. See if the full JSON array is there

### **Scenario B: Backend Returns 1, Frontend Gets 1**
**Symptoms:**
- Backend logs show "Downloaded 1 rows"
- Frontend logs show "Received 1 data points"

**Likely Causes:**
1. **yfinance API issue** - Yahoo Finance sometimes rate limits or returns incomplete data
2. **Invalid period/interval combination** - Some combinations don't work
3. **Symbol not found** - Wrong ticker or delisted stock
4. **Network issue** - Can't reach Yahoo Finance servers

**Solutions:**
1. Wait a few minutes and try again (rate limiting)
2. Try a different symbol (MSFT, GOOGL, TSLA)
3. Try a different period (5d instead of 1y)
4. Update yfinance: `pip install --upgrade yfinance`

### **Scenario C: Works Sometimes, Fails Other Times**
**Symptoms:**
- Same symbol works fine one minute, then returns 1 candle the next
- Different symbols behave differently

**Likely Causes:**
1. **Yahoo Finance rate limiting** - Too many requests
2. **yfinance caching issue** - Stale cache data
3. **Market hours** - Trying to get today's data when market hasn't opened

**Solutions:**
1. Clear yfinance cache:
   ```python
   import yfinance as yf
   yf.download("AAPL", period="1y", interval="1d", progress=False, ignore_tz=True)
   ```
2. Add retry logic to backend
3. Implement better caching strategy

---

## 🔧 Quick Fixes to Try

### **Fix #1: Restart Flask Server**
```bash
# Stop the server (Ctrl+C)
# Start it again
venv\Scripts\python.exe backend\api_server.py
```

After restart, the new logging will be active.

### **Fix #2: Update yfinance**
```bash
venv\Scripts\pip install --upgrade yfinance
```

yfinance gets updated frequently to fix Yahoo Finance API changes.

### **Fix #3: Clear Browser Cache**
Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)

### **Fix #4: Try Different Symbols**
- AAPL (Apple) - Usually very reliable
- MSFT (Microsoft) - Usually very reliable
- SPY (S&P 500 ETF) - Always has data
- Avoid: Penny stocks, recently delisted stocks, foreign stocks with weird symbols

---

## 📝 Expected Behavior

For `period=1y, interval=1d`:
- **Expected rows**: ~252 (trading days in a year)
- **Backend should log**: "Downloaded 252 rows" (give or take for holidays)
- **Frontend should log**: "Received 252 data points"

For `period=5d, interval=1d`:
- **Expected rows**: ~5 (recent trading days)
- **Backend should log**: "Downloaded 5 rows"
- **Frontend should log**: "Received 5 data points"

---

## 🚀 Next Steps

1. **Restart Flask server** to pick up the new logging
2. **Open browser console** (F12)
3. **Load a stock symbol**
4. **Compare logs**:
   - Flask terminal: How many rows did yfinance return?
   - Browser console: How many data points did frontend receive?
5. **Report findings**:
   - If backend=252, frontend=1 → Data transfer issue
   - If backend=1, frontend=1 → yfinance/Yahoo Finance issue
   - If it varies → Rate limiting or caching issue

---

## 📊 Test Cases

Try these and note which work:

- [ ] AAPL, 1y, 1d
- [ ] AAPL, 5d, 1d
- [ ] AAPL, 1mo, 1d
- [ ] MSFT, 1y, 1d
- [ ] SPY, 1y, 1d
- [ ] TSLA, 1y, 1d

If some work and some don't, it helps narrow down the issue!

---

**Status**: Logging added, ready to debug!
**Action**: Restart Flask server and try loading stocks
