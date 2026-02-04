# Live Output Streaming Implementation Specification

**Date**: 2026-01-31
**Status**: Planning Phase
**Priority**: High
**Estimated Time**: 30-45 minutes

---

## Objective

Add real-time streaming output to the scheduler status page task execution modal, allowing users to see live progress updates while tasks are running instead of waiting for completion.

---

## Current Behavior

**Problem**: When a user clicks "▶ Run Task X" on the scheduler status page:
1. Modal opens with "Waiting for output..."
2. Task runs in background (can take minutes)
3. User sees no progress - just a blank screen
4. After task completes, final result appears
5. No indication of what's happening during execution

**User Experience Issue**: For long-running tasks (like VACUUM on 9.7GB database), users wait 5-10 minutes seeing only "Waiting for output..." with no feedback.

---

## Desired Behavior

**Solution**: Stream live log output from the task to the browser in real-time:
1. User clicks "▶ Run Task 7: Maintenance"
2. Modal opens immediately
3. **Live output starts appearing line-by-line**:
   ```
   [2026-01-31 09:08:33] Starting Weekly Maintenance...
   [2026-01-31 09:08:34] VACUUMing Master Market Data DB...
   [2026-01-31 09:10:15] Master DB VACUUM complete (1m 41s)
   [2026-01-31 09:10:15] VACUUMing TurboMode DB (9.7GB)...
   [2026-01-31 09:15:42] TurboMode DB VACUUM complete (5m 27s)
   [2026-01-31 09:15:42] Cleaning temp directories...
   [2026-01-31 09:15:43] Task completed successfully
   ```
4. User sees progress in real-time
5. When task finishes, status updates to "Completed Successfully"

---

## Technical Architecture

### Current Architecture
```
Browser (Modal)
    ↓ (HTTP POST)
Flask Endpoint (/scheduler/run_task/7)
    ↓ (Synchronous call)
Task Function (run_weekly_maintenance)
    ↓ (Writes to)
Log File (backend/logs/task_7.log)
    ↓ (After completion)
JSON Response → Browser (displays final result)
```

**Limitation**: Browser waits for entire task to complete before receiving ANY output.

### Proposed Architecture (Server-Sent Events)

```
Browser (Modal with EventSource)
    ↑ (SSE stream - real-time)
    |
Flask SSE Endpoint (/scheduler/stream_task/7)
    ↑ (Reads continuously)
    |
Log File (backend/logs/task_7.log)
    ↑ (Written to)
    |
Task Function (run_weekly_maintenance)
    ↑ (Triggered by)
    |
Flask Endpoint (/scheduler/run_task/7)
    ↑ (HTTP POST)
Browser (Modal)
```

**Flow**:
1. Browser POSTs to `/scheduler/run_task/7` (starts task in background thread)
2. Browser opens EventSource connection to `/scheduler/stream_task/7`
3. SSE endpoint reads log file and streams new lines as they appear
4. Browser appends each line to modal output in real-time
5. When task completes, SSE closes connection

---

## Implementation Plan

### Phase 1: Backend - Add SSE Endpoint

**File**: `backend/unified_scheduler_api.py`

**New Endpoint**:
```python
@app.route('/scheduler/stream_task/<int:task_id>')
def stream_task_output(task_id):
    """
    Server-Sent Events endpoint for streaming task output

    Continuously reads task log file and yields new lines
    """
    def generate():
        log_file = f'backend/logs/task_{task_id}.log'
        # Tail the log file and yield new lines
        # Stop when task completes

    return Response(generate(), mimetype='text/event-stream')
```

**Key Requirements**:
- Read log file in real-time (tail -f behavior)
- Detect when task completes
- Close stream gracefully
- Handle file not found errors

### Phase 2: Modify Task Execution to Run in Background

**File**: `backend/unified_scheduler_api.py`

**Current**:
```python
def manual_run_overnight_scanner():
    result = run_task_manually(task_id=3)  # BLOCKS until complete
    return jsonify(result)
```

**Modified**:
```python
import threading

def manual_run_overnight_scanner():
    # Start task in background thread
    thread = threading.Thread(
        target=run_task_manually,
        args=(3,)
    )
    thread.start()

    # Return immediately
    return jsonify({'success': True, 'message': 'Task started'})
```

**Key Requirements**:
- Task runs in separate thread
- Endpoint returns immediately
- Browser doesn't wait for task completion

### Phase 3: Frontend - Update Modal JavaScript

**File**: `backend/unified_scheduler_api.py` (HTML template section)

**Current JavaScript**:
```javascript
function runTask(taskId, taskName) {
    fetch('/scheduler/run_task/' + taskId)
        .then(response => response.json())
        .then(data => {
            // Display final result only
            outputContent.textContent = output;
        });
}
```

**Modified JavaScript**:
```javascript
function runTask(taskId, taskName) {
    // Start the task
    fetch('/scheduler/run_task/' + taskId, {method: 'POST'});

    // Open SSE connection for live output
    var eventSource = new EventSource('/scheduler/stream_task/' + taskId);

    eventSource.onmessage = function(event) {
        // Append each line as it arrives
        outputContent.textContent += event.data + '\n';

        // Auto-scroll to bottom
        outputContent.scrollTop = outputContent.scrollHeight;
    };

    eventSource.onerror = function() {
        // Task completed or error
        eventSource.close();
        taskStatus.textContent = 'Task Completed';
    };
}
```

**Key Requirements**:
- EventSource API for SSE
- Append lines as they arrive
- Auto-scroll to show latest output
- Close connection when done

---

## Files to Modify

### Primary Files (Must Change)
1. **backend/unified_scheduler_api.py**
   - Add `/scheduler/stream_task/<task_id>` endpoint
   - Modify task execution to use threading
   - Update JavaScript in HTML template

### Supporting Files (May Need Changes)
2. **backend/unified_scheduler.py**
   - Ensure tasks write to log files (already done)
   - No changes needed if logging works

### Files NOT Changed (Read-Only)
- ❌ Training orchestrators
- ❌ Scanner files
- ❌ Database schemas
- ❌ ML models
- ❌ Backtest generators
- ❌ Any production trading logic

---

## Technical Challenges & Solutions

### Challenge 1: Tailing Log Files in Python
**Problem**: Need "tail -f" behavior to read new lines as they're written

**Solution**: Use file seeking with polling
```python
def tail_log_file(filepath):
    with open(filepath, 'r') as f:
        f.seek(0, 2)  # Go to end of file
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(0.1)  # Poll every 100ms
                # Check if task is done
                if task_completed():
                    break
```

### Challenge 2: Detecting Task Completion
**Problem**: SSE needs to know when to stop streaming

**Solution**: Check task status from scheduler
```python
from backend.unified_scheduler import job_state

def task_completed(task_id):
    return task_id in job_state.get('last_runs', {})
```

### Challenge 3: Thread Safety
**Problem**: Multiple users might trigger same task

**Solution**: Use locks or check if task is already running
```python
running_tasks = set()

def run_task_manually(task_id):
    if task_id in running_tasks:
        return {'error': 'Task already running'}

    running_tasks.add(task_id)
    try:
        # Run task
        pass
    finally:
        running_tasks.remove(task_id)
```

### Challenge 4: Log File Location
**Problem**: Need to know where each task writes logs

**Solution**: Use consistent naming pattern
```python
log_file = f'backend/logs/task_{task_id}_<timestamp>.log'
# OR use the scheduler's log manager
from backend.scheduler_logger import get_logger_manager
logger_manager = get_logger_manager()
log_path = logger_manager.get_task_log_path(task_id)
```

---

## Success Criteria

### Functional Requirements
✅ User sees log output appearing line-by-line in real-time
✅ Output auto-scrolls to show latest lines
✅ Modal shows "Running..." status while task executes
✅ Status changes to "Completed" when task finishes
✅ Works for all 7 tasks
✅ No changes to scheduler task logic

### Performance Requirements
✅ Output appears within 100ms of being logged
✅ No significant CPU overhead from streaming
✅ Multiple users can stream different tasks simultaneously
✅ Memory usage stays reasonable (no log buffering in RAM)

### Error Handling
✅ Graceful handling if log file doesn't exist
✅ Connection closes properly on task completion
✅ Error messages shown if task fails
✅ No browser console errors

---

## Testing Plan

### Test Case 1: Quick Task
- Run Task 7 (Maintenance) - completes in ~10 seconds
- Verify output streams in real-time
- Verify modal closes after completion

### Test Case 2: Long Task
- Run Task 3 (Scanner) - takes 5-10 minutes
- Verify output continues streaming throughout
- Verify auto-scroll works
- Verify completion detection

### Test Case 3: Failed Task
- Run task with intentional error
- Verify error appears in output
- Verify modal shows failure status

### Test Case 4: Multiple Tasks
- Run Task 7 in one browser tab
- Run Task 6 in another browser tab simultaneously
- Verify both stream correctly without interference

---

## Rollback Plan

If live streaming doesn't work or causes issues:

1. **Keep the current modal** (works, just shows final result)
2. **Revert unified_scheduler_api.py** to previous version
3. **Document what didn't work** for future attempts

**Rollback Command**:
```bash
git checkout backend/unified_scheduler_api.py
```

---

## Implementation Checklist

- [ ] Add SSE endpoint `/scheduler/stream_task/<task_id>`
- [ ] Implement log file tailing function
- [ ] Add task completion detection
- [ ] Modify task execution endpoints to use threading
- [ ] Update modal JavaScript to use EventSource
- [ ] Add auto-scroll to output area
- [ ] Test with Task 7 (quick task)
- [ ] Test with Task 3 (long task)
- [ ] Test error scenarios
- [ ] Update session notes
- [ ] Commit changes to git

---

## Reference Links

- **Server-Sent Events (SSE)**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- **Flask Streaming**: https://flask.palletsprojects.com/en/2.3.x/patterns/streaming/
- **EventSource API**: https://developer.mozilla.org/en-US/docs/Web/API/EventSource
- **Python Threading**: https://docs.python.org/3/library/threading.html

---

## Notes for Implementation

1. **Start Small**: Get basic SSE working with a static file first
2. **Test Incrementally**: Test each phase before moving to next
3. **Keep Current Code**: Don't delete working modal until new version works
4. **Log Everything**: Add debug logging to troubleshoot SSE issues
5. **Browser Compatibility**: EventSource works in all modern browsers

---

## Expected Timeline

- **Phase 1** (Backend SSE): 15 minutes
- **Phase 2** (Threading): 10 minutes
- **Phase 3** (Frontend): 15 minutes
- **Testing**: 10 minutes
- **Bug Fixes**: 10 minutes buffer

**Total**: ~60 minutes (including testing and debugging)

---

## Post-Implementation

After live streaming works:

1. **Test tonight's automated run** (11:30 PM scanner)
2. **Monitor for any issues** with threaded execution
3. **Consider adding** pause/cancel buttons for running tasks
4. **Consider adding** progress bars for tasks with known steps
5. **Update documentation** with screenshots of live output

---

## Questions to Consider

1. Should we show elapsed time in the modal?
2. Should we add a "Cancel Task" button?
3. Should we limit output to last N lines (e.g., 1000) to prevent memory issues?
4. Should we highlight errors in red in the output?
5. Should we add a download button to save full log?

---

**END OF SPECIFICATION**

This document should be updated as implementation progresses.
