# Deployment Checklist: Options Intelligence Caching System

## Pre-Deployment Verification

### Database Setup
- [ ] Verify `options_intel.db` exists at `backend/turbomode/Options/Data/`
- [ ] Run database rebuild script if needed:
  ```bash
  python C:\StockApp\backend\turbomode\Options\Data\rebuild_options_db.py
  ```
- [ ] Confirm database schema has 44+ fields (`enriched_options_signals` table)
- [ ] Verify database is writable (check file permissions)

### Scanner Validation
- [ ] Test scanner runs without errors:
  ```bash
  python C:\StockApp\backend\turbomode\Options\Scanner\options_scanner.py
  ```
- [ ] Verify scanner logs appear in `backend/turbomode/Options/Data/scanner.log`
- [ ] Confirm scanner successfully writes to `options_intel.db`
- [ ] Check scanner performance metrics (should complete in reasonable time)
- [ ] Verify error recovery works (scanner continues despite individual symbol failures)

### Scheduler Validation
- [ ] Test scheduler starts without errors:
  ```bash
  python C:\StockApp\backend\turbomode\Options\Scanner\options_scanner_scheduler.py
  ```
- [ ] Verify scheduler logs appear in `backend/turbomode/Options/Scanner/scheduler.log`
- [ ] Confirm market hours detection works correctly
- [ ] Verify scheduler runs scans every 5 minutes during market hours
- [ ] Test Ctrl+C graceful shutdown with final statistics display

### API Endpoint Verification
- [ ] Verify API server starts without errors
- [ ] Confirm OPTIONS_INTEL_API_AVAILABLE flag is True in startup logs
- [ ] Test `/turbomode/options_intel/all` endpoint returns data
- [ ] Test `/turbomode/options_intel/stats` endpoint returns cache statistics
- [ ] Test `/turbomode/options_intel/symbol/<SYMBOL>` endpoint works
- [ ] Verify API returns cached data (fast response < 1 second)

### Dashboard Verification
- [ ] Navigate to Signal Intelligence Dashboard (`frontend/turbomode/signal_intel_dashboard.html`)
- [ ] Verify dashboard loads cached data quickly (< 2 seconds)
- [ ] Confirm "Cache Updated" timestamp appears in stats bar
- [ ] Verify all 91+ signals display with full intelligence data
- [ ] Check signal strength, confirmation status, and other enriched fields display
- [ ] Verify dashboard auto-refreshes (30-second polling)
- [ ] Test error handling (disconnect API server and verify error message displays)

### Engine Placeholder Verification
- [ ] Confirm all engine files import without errors:
  - `expiration_selector.py`
  - `wing_selector.py`
  - `condor_pricing.py`
  - `analytics_engine.py`
  - `regime_engine.py`
  - `transition_engine.py`
  - `signal_intelligence.py`
- [ ] Verify engines return placeholder data (not None/errors)

### Integration Tests
- [ ] Run all test suites:
  ```bash
  # DB Writer tests
  python C:\StockApp\backend\turbomode\Options\Tests\test_db_writer.py

  # Scheduler tests
  python C:\StockApp\backend\turbomode\Options\Tests\test_scheduler.py

  # Scanner end-to-end tests
  python C:\StockApp\backend\turbomode\Options\Tests\test_scanner_end_to_end.py

  # API endpoint tests
  python C:\StockApp\backend\turbomode\Options\Tests\test_api_endpoint.py
  ```
- [ ] Verify all tests pass (check for [SUCCESS] or [PASS] messages)
- [ ] Review test output for any warnings

### Snapshot System Verification
- [ ] Verify `snapshot_writer.py` can write to database
- [ ] Verify `snapshot_loader.py` can read from database
- [ ] Test snapshot roundtrip (write then read same data)
- [ ] Confirm `get_previous_snapshot()` works for transition detection

### Performance Validation
- [ ] Measure dashboard load time (should be < 2 seconds for 91 signals)
- [ ] Measure scanner runtime (should complete all signals in < 5 minutes)
- [ ] Check database file size (should be reasonable, < 10MB for 100 signals)
- [ ] Monitor memory usage (no memory leaks during continuous operation)

### Error Handling Validation
- [ ] Test scanner with invalid symbols (should skip and continue)
- [ ] Test scheduler during market closed hours (should skip scan)
- [ ] Test API with empty database (should return empty array gracefully)
- [ ] Test dashboard with no cached data (should show "No signals" message)

## Deployment Steps

### 1. Stop Existing Services
- [ ] Stop API server if running
- [ ] Stop any existing scanner/scheduler processes

### 2. Deploy Code
- [ ] Pull latest code from repository
- [ ] Verify all new files are present:
  - API blueprint: `Options/api_options_intel.py`
  - Scanner: `Options/Scanner/options_scanner.py`
  - Scheduler: `Options/Scanner/options_scanner_scheduler.py`
  - Database: `Options/Data/options_intel.db`
  - Schema: `Options/Data/schema_options_intel.sql`
  - Engines: All 7 engine files
  - Tests: All 4 test files

### 3. Initialize Database
- [ ] Run database rebuild script
- [ ] Verify table structure matches schema
- [ ] Run initial scanner populate to cache data

### 4. Start Services
- [ ] Start API server:
  ```bash
  python C:\StockApp\backend\api_server.py
  ```
- [ ] Verify OPTIONS_INTEL_API_AVAILABLE in startup logs
- [ ] Start scheduler in background:
  ```bash
  # Linux/Mac
  nohup python C:\StockApp\backend\turbomode\Options\Scanner\options_scanner_scheduler.py &

  # Windows (use Task Scheduler or run in separate terminal)
  python C:\StockApp\backend\turbomode\Options\Scanner\options_scanner_scheduler.py
  ```

### 5. Post-Deployment Verification
- [ ] Check API server logs for errors
- [ ] Check scheduler logs for successful scans
- [ ] Verify database is being updated (check `last_updated` timestamps)
- [ ] Test dashboard loads and displays data
- [ ] Monitor for 30 minutes to ensure stability

## Monitoring & Maintenance

### Daily Checks
- [ ] Review scheduler logs for errors
- [ ] Check database size growth
- [ ] Verify dashboard loads correctly
- [ ] Monitor scan success rate (should be > 90%)

### Weekly Checks
- [ ] Review cache freshness (all signals should have recent `last_updated`)
- [ ] Check for database corruption
- [ ] Review error patterns in logs
- [ ] Performance audit (load times, scan times)

### Monthly Maintenance
- [ ] Archive old logs
- [ ] Database optimization (VACUUM if needed)
- [ ] Review and update engine placeholder logic
- [ ] Update documentation based on learnings

## Rollback Plan

If deployment fails:

1. [ ] Stop scheduler and API server
2. [ ] Restore previous code version
3. [ ] Delete `options_intel.db` if corrupted
4. [ ] Restart services with previous configuration
5. [ ] Dashboard will fall back to real-time calculation (slow but functional)

## Success Criteria

- ✅ Dashboard loads in < 2 seconds (vs 10-15 minutes previously)
- ✅ All 91+ signals display with full options intelligence
- ✅ Cache updates every 5 minutes during market hours
- ✅ Scheduler runs continuously without crashes
- ✅ API endpoints return 200 status with valid JSON
- ✅ All integration tests pass
- ✅ No critical errors in logs after 24 hours

## Support Contacts

- **Primary**: Claude Code Session (this implementation)
- **Database Issues**: Check `rebuild_options_db.py` script
- **Scanner Issues**: Review `scanner.log` and `options_scanner.py`
- **API Issues**: Check `api_server.py` startup logs
- **Dashboard Issues**: Browser console + Network tab

---

**Deployment Date**: _____________

**Deployed By**: _____________

**Sign-off**: _____________
