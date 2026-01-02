# How to Disable Scheduler Before Running Backtest

## ⚠️ IMPORTANT: Disable scheduler BEFORE running the backtest!

You have TWO schedulers that might interfere:

### 1. **TurboMode Scheduler** (11 PM nightly)
- Location: `backend/turbomode/turbomode_scheduler.py`
- Schedule: Runs at 11 PM every night
- State file: `backend/data/turbomode_scheduler_state.json`

### 2. **Agents Scheduler** (Midnight UTC)
- Location: `agents/schedule_scanner.py`
- Launcher: `start_scheduler.bat`
- Schedule: Runs at midnight UTC

## How to Check if Schedulers Are Running

```bash
# Check for running Python processes
tasklist | findstr /i "python"

# Look for processes running:
# - schedule_scanner.py
# - turbomode_scheduler.py
# - start_scheduler.bat
```

## How to Stop Schedulers

### Option 1: Kill Python Processes (Quick)
```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Then verify they're stopped
tasklist | findstr /i "python"
```

### Option 2: Disable TurboMode Scheduler State
```bash
# Edit the state file to disable it
cd backend/data
# If turbomode_scheduler_state.json exists, set "enabled": false
```

### Option 3: Just Don't Start Them
- Don't run `start_scheduler.bat`
- Don't run the scheduler Python scripts
- They only run if you manually start them

## Recommendation for Tomorrow

**SAFEST APPROACH:**

1. When you wake up, check for running schedulers:
   ```bash
   tasklist | findstr /i "python"
   ```

2. If any scheduler is running, kill it:
   ```bash
   taskkill /F /IM python.exe
   ```

3. Immediately start the backtest (takes 25-30 min):
   ```bash
   cd backend/turbomode
   rm -f backtest_checkpoint.json
   ../../venv/Scripts/python.exe generate_backtest_data.py
   ```

4. After backtest completes, train models (~10-15 min)

5. Then run scanner with production models

6. **ONLY AFTER** production models are trained, you can re-enable schedulers

## Good News

Based on the code I found, the schedulers appear to be **manually started** (via `start_scheduler.bat`), not Windows Task Scheduler or systemd.

This means: **If you haven't manually run `start_scheduler.bat` or the Python scheduler scripts, nothing is scheduled!**

You're safe to run the backtest anytime. Just make sure no Python processes are running first.
