# Old Master Data Scheduler - Archived 2026-01-06

This directory contains the old single-task Master Data scheduler that was replaced by the Unified Scheduler System.

## Archived Files:

1. **master_data_scheduler.py** - Old scheduler that only handled Master Data ingestion
2. **master_data_api_extension.py** - Old Flask API integration for Master Data scheduler

## Why Archived:

These files were replaced by the new **Unified Scheduler System** which provides:

- **6 scheduled tasks** instead of just 1 (Master Data ingestion)
- **Config-driven schedules** via `scheduler_config.json`
- **Complete automation pipeline** for TurboMode Phase 2
- **Dependency-aware execution** (tasks wait for dependencies)
- **Manual triggering** via REST API endpoints
- **Better logging and error handling**

## New System Files:

- `backend/scheduler_config.json` - Schedule configuration
- `backend/unified_scheduler.py` - Core scheduler (6 tasks)
- `backend/unified_scheduler_api.py` - Flask API endpoints
- `README_SCHEDULE.md` - Complete documentation

## Migration Notes:

The old Master Data scheduler ran nightly at 10:45 PM. The new system:
- **Task 1** (Master Data Ingestion) still runs at 10:45 PM
- Plus 5 additional tasks for complete automation
- All controlled via `scheduler_config.json`

## Restoration (if needed):

If you need to restore the old scheduler:

1. Copy files back to original locations:
   - `master_data_scheduler.py` → `master_market_data/`
   - `master_data_api_extension.py` → `backend/`

2. Edit `backend/api_server.py` line ~3090:
   ```python
   # Change FROM:
   from backend.unified_scheduler_api import init_unified_scheduler_api

   # Change TO:
   from backend.master_data_api_extension import init_master_data_api
   ```

3. Restart Flask

**Note:** This is NOT recommended - the new unified scheduler is superior in every way.

---

**Archived:** 2026-01-06
**Reason:** Replaced by Unified Scheduler System
**By:** TurboMode Phase 2 Implementation
