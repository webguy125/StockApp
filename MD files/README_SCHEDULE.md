# TurboMode Phase 2 - Unified Scheduler Documentation

**Version:** 1.0
**Date:** 2026-01-06
**Architecture:** MASTER_MARKET_DATA_ARCHITECTURE.json v1.1
**Configuration:** scheduler_config.json v1.0

---

## Overview

The Unified Scheduler implements a complete automation pipeline for TurboMode Phase 2, managing six scheduled tasks that run daily/weekly to maintain the Master Market Data DB and TurboMode ML models.

### Key Features:

- **Config-Driven**: All schedules defined in `backend/scheduler_config.json`
- **Flask-Integrated**: Runs inside Flask application using APScheduler
- **API-Controllable**: All tasks can be triggered manually via REST endpoints
- **Dependency-Aware**: Tasks respect dependencies (e.g., training waits for ingestion)
- **Idempotent**: All tasks are safe to re-run without side effects
- **Audit Trail**: Complete logging of all task executions

---

## Architecture Compliance

Per your JSON specification, the scheduler enforces these global rules:

1. ✅ **All scheduled jobs are Python functions** inside the Flask app
2. ✅ **Each job has a corresponding Flask endpoint** for manual triggering
3. ✅ **Only Task 1 calls external data sources** (IBKR/yfinance)
4. ✅ **All other tasks read from Master Market Data DB** (read-only)
5. ✅ **TurboMode.db is write-only** for model outputs, predictions, drift logs, SHAP logs, metadata
6. ✅ **All jobs log start, finish, and error states**
7. ✅ **All jobs are idempotent** and safe to re-run
8. ✅ **APScheduler uses BackgroundScheduler** with deterministic ordering
9. ✅ **Jobs respect dependency ordering**

---

## Scheduled Tasks

### Task 1: Master Market Data Ingestion
**Function:** `run_ingestion()`
**Schedule:** Daily at 22:45 (10:45 PM)
**Endpoint:** `POST /scheduler/run_ingestion`
**Dependencies:** None

**Responsibilities:**
- Fetch daily OHLCV, fundamentals, splits, dividends, and metadata
- Write updates to Master Market Data DB
- Does NOT write to TurboMode.db
- Logs ingestion results

**Data Source:** IBKR Gateway (primary) + yfinance (fallback)
**Timeout:** 30 minutes
**Retry:** Yes (max 3 retries)

---

### Task 2: TurboMode Training Orchestrator
**Function:** `run_orchestrator()`
**Schedule:** Daily at 23:00 (11:00 PM)
**Endpoint:** `POST /scheduler/run_orchestrator`
**Dependencies:** Task 1 (waits for ingestion to complete)

**Responsibilities:**
- Load symbols from Master Market Data DB
- Generate features and training samples (GPU-accelerated)
- Train all 8 models + meta-learner
- Compute SHAP values and save logs
- Update model metadata and versioning in TurboMode.db

**Reads From:** Master Market Data DB (read-only)
**Writes To:** TurboMode.db (models, metadata, SHAP logs)
**Timeout:** 120 minutes
**Retry:** No (manual review required on failure)

---

### Task 3: Overnight Scanner
**Function:** `run_overnight_scanner()`
**Schedule:** Daily at 23:30 (11:30 PM)
**Endpoint:** `POST /scheduler/run_overnight_scanner`
**Dependencies:** Task 2 (waits for training to complete)

**Responsibilities:**
- Load the latest trained model from TurboMode.db
- Generate predictions for the next trading day
- Save signals and prediction metadata to TurboMode.db

**Reads From:** Master Market Data DB (OHLCV data) + TurboMode.db (trained models)
**Writes To:** TurboMode.db (predictions, signals)
**Timeout:** 60 minutes
**Retry:** Yes (max 2 retries)

---

### Task 4: Backtest Data Generator
**Function:** `run_backtest_generator()`
**Schedule:** Daily at 00:00 (midnight)
**Endpoint:** `POST /scheduler/run_backtest_generator`
**Dependencies:** Task 1 (waits for ingestion to complete)

**Responsibilities:**
- Use Master Market Data DB to generate updated backtest datasets
- Save backtest results and metadata to TurboMode.db
- Ensure backtests reflect the latest ingestion

**Reads From:** Master Market Data DB (read-only)
**Writes To:** TurboMode.db (backtest results, metadata)
**Timeout:** 90 minutes
**Retry:** No
**Status:** Placeholder implementation (ready for backtest logic)

---

### Task 5: Drift Monitoring System
**Function:** `run_drift_monitor()`
**Schedule:** Daily at 00:30 (12:30 AM)
**Endpoint:** `POST /scheduler/run_drift_monitor`
**Dependencies:** Task 2 (training) + Task 4 (backtest)

**Responsibilities:**
- Compare today's feature distributions to historical baselines
- Detect regime shifts using PSI, KL Divergence, KS Statistic
- Write drift logs to TurboMode.db
- Trigger retraining flags if drift exceeds thresholds

**Reads From:** TurboMode.db (historical features) + Master Market Data DB
**Writes To:** TurboMode.db (drift logs, retraining flags)
**Timeout:** 45 minutes
**Retry:** Yes (max 2 retries)
**Status:** Placeholder implementation (ready for drift detection)

---

### Task 6: Weekly Maintenance
**Function:** `run_weekly_maintenance()`
**Schedule:** Weekly on Sunday at 02:00 (2:00 AM)
**Endpoint:** `POST /scheduler/run_weekly_maintenance`
**Dependencies:** None

**Responsibilities:**
- VACUUM Master Market Data DB
- VACUUM TurboMode.db
- Clean temp directories
- Archive logs older than 30 days

**Reads From:** Filesystem
**Writes To:** Databases (VACUUM operation)
**Timeout:** 30 minutes
**Retry:** No

---

## API Endpoints

### Scheduler Control

#### Get Scheduler Status
```bash
GET /scheduler/status

# Response:
{
  "running": true,
  "version": "1.0",
  "jobs": [
    {
      "id": "task_1",
      "name": "Master Market Data Ingestion",
      "next_run": "2026-01-06T22:45:00-05:00"
    },
    ...
  ],
  "last_runs": {
    "1": "2026-01-05T22:45:23",
    "2": "2026-01-05T23:00:45",
    ...
  },
  "last_results": {
    "1": {
      "status": "success",
      "symbols_processed": 83,
      "candles_ingested": 41500
    },
    ...
  },
  "errors": {}
}
```

#### Start Scheduler
```bash
POST /scheduler/start

# Response:
{
  "success": true,
  "message": "Unified scheduler started"
}
```

#### Stop Scheduler
```bash
POST /scheduler/stop

# Response:
{
  "success": true,
  "message": "Unified scheduler stopped"
}
```

---

### Manual Task Execution

All tasks can be triggered manually (bypasses schedule):

```bash
# Task 1: Master Market Data Ingestion
POST /scheduler/run_ingestion

# Task 2: TurboMode Training Orchestrator
POST /scheduler/run_orchestrator

# Task 3: Overnight Scanner
POST /scheduler/run_overnight_scanner

# Task 4: Backtest Data Generator
POST /scheduler/run_backtest_generator

# Task 5: Drift Monitoring System
POST /scheduler/run_drift_monitor

# Task 6: Weekly Maintenance
POST /scheduler/run_weekly_maintenance

# Generic endpoint (task_id = 1-6)
POST /scheduler/run_task/<task_id>
```

**Example Response:**
```json
{
  "success": true,
  "task_id": 1,
  "results": {
    "total_symbols": 83,
    "successful": 83,
    "failed": 0,
    "total_candles": 41500
  }
}
```

---

## Daily Execution Timeline

Here's the default daily schedule (times in Eastern Time):

```
22:45 PM - Task 1: Master Market Data Ingestion (30 min)
           ↓
23:00 PM - Task 2: TurboMode Training Orchestrator (60-120 min)
           ↓
23:30 PM - Task 3: Overnight Scanner (30-60 min)

00:00 AM - Task 4: Backtest Data Generator (60-90 min)
           ↓
00:30 AM - Task 5: Drift Monitoring System (30-45 min)

SUNDAY ONLY:
02:00 AM - Task 6: Weekly Maintenance (10-30 min)
```

**Total Daily Runtime:** ~3-4 hours
**Total Weekly Runtime (Sunday):** ~3.5-4.5 hours

---

## Configuration

### scheduler_config.json Structure

```json
{
  "version": "1.0",
  "global_settings": {
    "timezone": "America/New_York",
    "max_concurrent_jobs": 1,
    "job_defaults": {
      "coalesce": true,
      "max_instances": 1,
      "misfire_grace_time": 300
    }
  },
  "scheduled_tasks": [
    {
      "task_id": 1,
      "name": "Master Market Data Ingestion",
      "function_name": "run_ingestion",
      "endpoint": "/scheduler/run_ingestion",
      "enabled": true,
      "schedule": {
        "type": "cron",
        "hour": 22,
        "minute": 45,
        "day_of_week": "mon-sun"
      },
      "timeout_minutes": 30,
      "retry_on_failure": true,
      "max_retries": 3
    },
    ...
  ]
}
```

### Modifying the Schedule

1. Edit `backend/scheduler_config.json`
2. Change `hour`, `minute`, or `day_of_week` for any task
3. Set `enabled: false` to disable a task
4. Restart Flask or call `POST /scheduler/stop` then `POST /scheduler/start`

**Example: Run ingestion at 11:00 PM instead of 10:45 PM:**
```json
{
  "task_id": 1,
  "schedule": {
    "hour": 23,  // Changed from 22
    "minute": 0  // Changed from 45
  }
}
```

---

## File Structure

```
C:\StockApp/
├── backend/
│   ├── scheduler_config.json              # Schedule configuration
│   ├── unified_scheduler.py               # Core scheduler logic (6 task functions)
│   ├── unified_scheduler_api.py           # Flask API endpoints
│   ├── api_server.py                      # Flask app (integrated)
│   │
│   ├── turbomode/
│   │   ├── training_orchestrator.py       # Task 2 implementation
│   │   ├── overnight_scanner.py           # Task 3 implementation
│   │   └── drift_monitor.py               # Task 5 implementation
│   │
│   └── data/
│       └── turbomode.db                   # TurboMode DB (private ML memory)
│
├── master_market_data/
│   ├── market_data.db                     # Master DB (shared, read-only)
│   ├── market_data_api.py                 # Read-only API
│   ├── ingest_via_ibkr.py                 # Task 1 implementation (IBKR)
│   └── ingest_market_data.py              # Task 1 implementation (yfinance)
│
└── README_SCHEDULE.md                     # This file
```

---

## Logging

All tasks log to Flask's standard output with this format:

```
[2026-01-06 22:45:00] unified_scheduler - INFO - ================================================================================
[2026-01-06 22:45:00] unified_scheduler - INFO - TASK 1: Master Market Data Ingestion
[2026-01-06 22:45:00] unified_scheduler - INFO - Started: 2026-01-06 22:45:00
[2026-01-06 22:45:00] unified_scheduler - INFO - ================================================================================
...
[2026-01-06 22:58:23] unified_scheduler - INFO - [SUCCESS] Task 1 completed
[2026-01-06 22:58:23] unified_scheduler - INFO -   Symbols processed: 83
[2026-01-06 22:58:23] unified_scheduler - INFO -   Candles ingested: 41,500
```

---

## Troubleshooting

### Scheduler Not Starting
```bash
# Check if scheduler is running
curl http://localhost:5000/scheduler/status

# If not running, start it manually
curl -X POST http://localhost:5000/scheduler/start
```

### Task Failed
```bash
# Check scheduler status for error details
curl http://localhost:5000/scheduler/status

# Look for errors in the "errors" field:
{
  "errors": {
    "1": {
      "timestamp": "2026-01-06T22:45:00",
      "error": "IBKR Gateway not available"
    }
  }
}

# Manually retry the task
curl -X POST http://localhost:5000/scheduler/run_ingestion
```

### Change Schedule
1. Edit `backend/scheduler_config.json`
2. Restart scheduler:
```bash
curl -X POST http://localhost:5000/scheduler/stop
curl -X POST http://localhost:5000/scheduler/start
```

### Disable a Task
1. Edit `backend/scheduler_config.json`
2. Set `"enabled": false` for the task
3. Restart scheduler

---

## Testing

### Test Individual Tasks

```bash
# Test Task 1 (Ingestion)
curl -X POST http://localhost:5000/scheduler/run_ingestion

# Test Task 2 (Training)
curl -X POST http://localhost:5000/scheduler/run_orchestrator

# Test Task 3 (Scanner)
curl -X POST http://localhost:5000/scheduler/run_overnight_scanner

# Test Task 4 (Backtest)
curl -X POST http://localhost:5000/scheduler/run_backtest_generator

# Test Task 5 (Drift)
curl -X POST http://localhost:5000/scheduler/run_drift_monitor

# Test Task 6 (Maintenance)
curl -X POST http://localhost:5000/scheduler/run_weekly_maintenance
```

### Run Scheduler in Test Mode

```bash
cd C:\StockApp\backend
python unified_scheduler.py

# Output:
# ================================================================================
# UNIFIED SCHEDULER - TEST MODE
# ================================================================================
#
# Scheduler Status:
#   Running: True
#   Version: 1.0
#   Active Jobs: 6
#
# Scheduled Jobs:
#   - Master Market Data Ingestion
#     Next run: 2026-01-06T22:45:00-05:00
#   - TurboMode Training Orchestrator
#     Next run: 2026-01-06T23:00:00-05:00
#   ...
```

---

## Maintenance

### Daily (Automated)
- ✅ Master Market Data ingestion
- ✅ Model training
- ✅ Overnight scanning
- ✅ Backtest generation
- ✅ Drift monitoring

### Weekly (Automated - Sunday 2 AM)
- ✅ Database VACUUM
- ✅ Temp file cleanup
- ✅ Log archiving (pending implementation)

### Monthly (Manual)
- Review drift_monitoring table for persistent alerts
- Review training_runs table for model performance trends
- Update scheduler_config.json if schedule changes needed

---

## Production Deployment

### Prerequisites
1. Flask app running: `python backend/api_server.py`
2. IBKR Gateway running on port 4002 (optional, will fallback to yfinance)
3. Master Market Data DB populated: `C:\StockApp\master_market_data\market_data.db`

### Deployment Steps
1. Scheduler auto-starts when Flask launches
2. Verify with: `curl http://localhost:5000/scheduler/status`
3. Monitor logs in Flask output
4. Tasks run automatically at scheduled times

### High Availability
- Scheduler persists state across Flask restarts
- All tasks are idempotent (safe to re-run)
- Missed jobs execute immediately upon scheduler restart (within grace period)

---

## Future Enhancements

Per your specification, these components are ready for full implementation:

1. **Task 4 (Backtest Generator)**: Currently placeholder - ready for backtest logic
2. **Task 5 (Drift Monitor)**: Currently placeholder - ready for drift detection with baseline data
3. **Task 6 (Log Archiving)**: Currently skipped - ready for archive logic

All scaffolding is in place - just implement the core logic in the respective task functions.

---

## Support

For issues or questions:
- Check Flask logs for detailed error messages
- Use `GET /scheduler/status` to see task execution history
- Manually trigger tasks via API endpoints for debugging
- Review `scheduler_config.json` for configuration issues

---

**Generated:** 2026-01-06
**By:** TurboMode Unified Scheduler System
**Version:** 1.0
