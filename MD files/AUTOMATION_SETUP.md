# Scanner Automation Setup Guide

## Quick Start - Choose Your Method

### Option 1: Python Scheduler (Easiest - For Testing)
### Option 2: Windows Task Scheduler (Best - For Production)

---

## Option 1: Python Scheduler (Simple)

### What It Does
Keeps a Python process running 24/7 that triggers the scanner at midnight UTC.

### Pros
✅ Easy to start/stop
✅ See real-time output in console
✅ Good for testing

### Cons
❌ Stops if you close the console
❌ Stops if computer restarts
❌ Requires leaving a console window open

### Setup Steps

1. **Double-click** `start_scheduler.bat`

   Or manually:
   ```bash
   cd C:\StockApp\agents
   ..\venv\Scripts\activate
   python schedule_scanner.py
   ```

2. **Leave the console window open**
   - The scheduler will run continuously
   - You'll see: "Next run: 2025-11-20 00:00:00"

3. **Stop the scheduler**
   - Press Ctrl+C in the console window

### Test Immediately

Instead of waiting for midnight, run a scan now:

```bash
cd C:\StockApp\agents
..\venv\Scripts\activate
python schedule_scanner.py --now
```

Or just double-click: `run_scanner.bat`

---

## Option 2: Windows Task Scheduler (Recommended)

### What It Does
Windows automatically runs the scanner at midnight, even if you're not logged in.

### Pros
✅ Runs automatically without user intervention
✅ Survives computer restarts
✅ Runs even when logged out
✅ Creates log files of each run

### Cons
❌ Slightly more setup
❌ Need to check logs to see output

### Setup Steps

#### Method A: Use PowerShell (Fastest)

1. **Open PowerShell as Administrator**
   - Press Windows key
   - Type "PowerShell"
   - Right-click "Windows PowerShell"
   - Select "Run as administrator"

2. **Run this command:**
   ```powershell
   schtasks /create /tn "StockApp Scanner" /tr "C:\StockApp\run_scanner_scheduled.bat" /sc daily /st 00:00 /rl limited /f
   ```

3. **Verify it was created:**
   ```powershell
   schtasks /query /tn "StockApp Scanner"
   ```

4. **Test it immediately:**
   ```powershell
   schtasks /run /tn "StockApp Scanner"
   ```

#### Method B: Use Task Scheduler GUI (Visual)

1. **Open Task Scheduler**
   - Press Windows + R
   - Type: `taskschd.msc`
   - Press Enter

2. **Create New Task**
   - Click "Create Task..." (not "Create Basic Task")
   - Name: `StockApp Scanner`
   - Description: `Runs comprehensive market scanner nightly`
   - Select: "Run whether user is logged on or not"
   - Check: "Do not store password"

3. **Set Trigger**
   - Go to "Triggers" tab
   - Click "New..."
   - Begin the task: "On a schedule"
   - Settings: Daily
   - Start time: 12:00:00 AM (midnight)
   - Recur every: 1 days
   - Click OK

4. **Set Action**
   - Go to "Actions" tab
   - Click "New..."
   - Action: "Start a program"
   - Program/script: `C:\StockApp\run_scanner_scheduled.bat`
   - Start in: `C:\StockApp`
   - Click OK

5. **Configure Settings**
   - Go to "Settings" tab
   - Check: "Allow task to be run on demand"
   - Check: "Run task as soon as possible after a scheduled start is missed"
   - Check: "If the task fails, restart every: 10 minutes"
   - Attempt to restart up to: 3 times
   - Uncheck: "Stop the task if it runs longer than: 3 days"
   - Click OK

6. **Test the Task**
   - Find your task in the Task Scheduler Library
   - Right-click → "Run"
   - Check: `C:\StockApp\agents\logs\` for log files

#### Method C: Import XML Template

1. **Open Task Scheduler**
   - Press Windows + R
   - Type: `taskschd.msc`
   - Press Enter

2. **Import Task**
   - Click "Import Task..." in the right panel
   - Browse to: `C:\StockApp\agents\TaskScheduler_Template.xml`
   - Click "Open"
   - Review settings
   - Click "OK"

3. **Test It**
   - Right-click the task → "Run"

---

## Verify Automation is Working

### Check Log Files

After a scheduled run, check:

```
C:\StockApp\agents\logs\scanner_YYYYMMDD_HHMMSS.log
```

You should see:
```
========================================
Scheduled Scanner Run
Started: 11/20/2025 12:00:00 AM
========================================
🚀 COMPREHENSIVE SCANNER - S&P 500 + Top 100 Cryptos
...
✅ SCAN COMPLETE
```

### Check Output File

```
C:\StockApp\agents\repository\scanner_output.json
```

Check the timestamp - it should match the last run time.

### Windows Task Scheduler Status

1. Open Task Scheduler
2. Find "StockApp Scanner"
3. Check "Last Run Time" and "Last Run Result"
4. Should show: "The operation completed successfully. (0x0)"

---

## Change Schedule Time

### Option 1: Python Scheduler

Edit `agents/schedule_scanner.py`, line ~70:

```python
# Change from midnight to 2 AM:
schedule.every().day.at("02:00").do(run_comprehensive_scan)
```

### Option 2: Windows Task Scheduler

1. Open Task Scheduler
2. Find "StockApp Scanner"
3. Right-click → Properties
4. Go to "Triggers" tab
5. Double-click the trigger
6. Change "Start" time
7. Click OK

---

## Troubleshooting

### Scheduler Not Running

**Check Python Scheduler:**
```bash
# Make sure it's running
tasklist | findstr python
```

**Check Windows Task:**
```powershell
schtasks /query /tn "StockApp Scanner" /v
```

### No Log Files Created

**Manually create logs directory:**
```bash
cd C:\StockApp\agents
mkdir logs
```

### Task Shows as "Failed"

1. Check log file for error messages
2. Verify paths in batch file are correct
3. Test batch file manually:
   ```bash
   C:\StockApp\run_scanner_scheduled.bat
   ```

### Scanner Takes Too Long

- Polygon free tier is slow (5 requests/min)
- Consider reducing stock count or upgrading Polygon plan
- Scanner timeout is 2 hours (should be plenty)

---

## Stop/Disable Automation

### Python Scheduler
Just close the console window or press Ctrl+C

### Windows Task Scheduler

**Disable:**
```powershell
schtasks /change /tn "StockApp Scanner" /disable
```

**Enable:**
```powershell
schtasks /change /tn "StockApp Scanner" /enable
```

**Delete:**
```powershell
schtasks /delete /tn "StockApp Scanner" /f
```

Or use Task Scheduler GUI and right-click the task.

---

## Recommended Setup

**For Daily Use:**
1. Use Windows Task Scheduler
2. Set to run at midnight (when you're asleep)
3. Check logs in the morning
4. View heat maps: http://127.0.0.1:5000/heatmap

**For Testing:**
1. Use `run_scanner.bat` for immediate runs
2. Use Python scheduler to test timing

**Production:**
- Windows Task Scheduler
- Daily at midnight
- Logs enabled
- Auto-restart on failure

---

## Next Steps

1. **Choose your automation method** (Task Scheduler recommended)
2. **Test it**: Run manually to verify it works
3. **Check output**: Look at logs and `scanner_output.json`
4. **View results**: Visit heat map page
5. **Monitor**: Check logs daily for the first week

**You're all set!** 🚀 The scanner will run automatically every night.
