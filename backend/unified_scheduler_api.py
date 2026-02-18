"""
Flask API Extension for Unified Scheduler
Provides REST endpoints for scheduler control and manual task execution

All endpoints follow /scheduler/* pattern
"""

import logging
from flask import jsonify, request
from datetime import datetime
from backend.unified_scheduler import (
    start_unified_scheduler,
    stop_unified_scheduler,
    get_scheduler_status,
    run_task_manually,
    TASK_FUNCTIONS
)

logger = logging.getLogger('unified_scheduler_api')

# HTML template for human-friendly scheduler status page
SCHEDULER_STATUS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Scheduler Status</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }

        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            margin-top: 10px;
        }

        .status-running {
            background: #10b981;
            color: white;
        }

        .status-stopped {
            background: #ef4444;
            color: white;
        }

        .jobs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .job-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }

        .job-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        }

        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f3f4f6;
        }

        .job-id {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18px;
        }

        .job-name {
            flex: 1;
            margin-left: 15px;
            font-size: 18px;
            font-weight: 600;
            color: #1f2937;
        }

        .copy-btn {
            background: rgba(102, 126, 234, 0.1);
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.2s;
        }

        .copy-btn:hover {
            background: rgba(102, 126, 234, 0.2);
            transform: scale(1.1);
        }

        .copy-btn:active {
            transform: scale(0.95);
        }

        .job-info {
            margin-top: 15px;
        }

        .info-row {
            display: flex;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid #f3f4f6;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: #6b7280;
            font-weight: 500;
        }

        .info-value {
            color: #1f2937;
            font-weight: 600;
            text-align: right;
        }

        .next-run {
            color: #10b981;
        }

        .countdown {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            text-align: center;
        }

        .countdown-label {
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .countdown-time {
            font-size: 24px;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }

        .last-run {
            color: #3b82f6;
        }

        .error {
            color: #ef4444;
        }

        .success {
            color: #10b981;
        }

        .controls {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .controls h2 {
            color: #333;
            margin-bottom: 15px;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            margin-right: 10px;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .btn-start {
            background: #10b981;
            color: white;
        }

        .btn-start:hover {
            background: #059669;
        }

        .btn-stop {
            background: #ef4444;
            color: white;
        }

        .btn-stop:hover {
            background: #dc2626;
        }

        .btn-refresh {
            background: #3b82f6;
            color: white;
        }

        .btn-refresh:hover {
            background: #2563eb;
        }

        .btn-task {
            background: #8b5cf6;
            color: white;
        }

        .btn-task:hover {
            background: #7c3aed;
        }

        .timestamp {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 14px;
        }

        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .modal-content {
            background-color: white;
            margin: 5% auto;
            border-radius: 10px;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
            padding: 20px 30px;
            border-bottom: 2px solid #f3f4f6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h2 {
            margin: 0;
            color: #1f2937;
        }

        .close {
            color: #9ca3af;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.2s;
        }

        .close:hover {
            color: #ef4444;
        }

        .modal-body {
            padding: 20px 30px;
            overflow-y: auto;
            flex: 1;
        }

        .modal-footer {
            padding: 15px 30px;
            border-top: 2px solid #f3f4f6;
            text-align: right;
        }

        .task-status {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-weight: 600;
            text-align: center;
            font-size: 16px;
        }

        .status-running {
            background: #dbeafe;
            color: #1e40af;
        }

        .status-success-task {
            background: #d1fae5;
            color: #065f46;
        }

        .status-error-task {
            background: #fee2e2;
            color: #991b1b;
        }

        .output-content {
            background: #1f2937;
            color: #f3f4f6;
            padding: 20px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
        }

        .output-content:empty::before {
            content: "Waiting for output...";
            color: #9ca3af;
        }

        .no-jobs {
            background: white;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            color: #6b7280;
            font-size: 18px;
        }

        .notification-banner {
            background: white;
            border-radius: 10px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            border-left: 5px solid;
        }

        .notification-success {
            border-left-color: #10b981;
            background: linear-gradient(to right, #ecfdf5 0%, white 100%);
        }

        .notification-error {
            border-left-color: #ef4444;
            background: linear-gradient(to right, #fef2f2 0%, white 100%);
        }

        .notification-warning {
            border-left-color: #f59e0b;
            background: linear-gradient(to right, #fffbeb 0%, white 100%);
        }

        .notification-icon {
            font-size: 32px;
            margin-right: 20px;
        }

        .notification-content {
            flex: 1;
        }

        .notification-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .notification-message {
            font-size: 14px;
            color: #6b7280;
        }

        .notification-time {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Unified Scheduler Status</h1>
            <span class="status-badge {% if status.running %}status-running{% else %}status-stopped{% endif %}">
                {% if status.running %}RUNNING{% else %}STOPPED{% endif %}
            </span>
            <p style="color: #6b7280; margin-top: 10px;">Version: {{ status.version }}</p>
            <p style="color: #667eea; margin-top: 10px;">
                <a href="/scheduler/history" style="color: #667eea; text-decoration: none; font-weight: 500; margin-right: 20px;">📊 Task Execution History</a>
                <a href="/turbomode/performance_dashboard.html" style="color: #667eea; text-decoration: none; font-weight: 500;">💰 Performance Dashboard</a>
            </p>
        </div>

        {% set last_task_info = get_last_execution_info(status) %}
        {% if last_task_info %}
        <div class="notification-banner notification-{{ last_task_info.type }}">
            <div class="notification-icon">
                {% if last_task_info.type == 'success' %}
                    ✓
                {% elif last_task_info.type == 'error' %}
                    ✗
                {% else %}
                    ⚠
                {% endif %}
            </div>
            <div class="notification-content">
                <div class="notification-title" style="color: {% if last_task_info.type == 'success' %}#10b981{% elif last_task_info.type == 'error' %}#ef4444{% else %}#f59e0b{% endif %};">
                    {{ last_task_info.title }}
                </div>
                <div class="notification-message">
                    {{ last_task_info.message }}
                </div>
                <div class="notification-time">
                    {{ last_task_info.time }}
                </div>
            </div>
        </div>
        {% endif %}

        {% if status.jobs %}
        <div class="jobs-grid">
            {% for job in status.jobs %}
            <div class="job-card">
                <div class="job-header">
                    <div class="job-id">{{ job.id.replace('task_', '') }}</div>
                    <div class="job-name">
                        {% if '|' in job.name %}
                            {{ job.name.split('|')[0] }}<br>
                            <span style="font-size: 14px; color: #667eea;">{{ job.name.split('|')[1] }}</span>
                        {% else %}
                            {{ job.name }}
                        {% endif %}
                    </div>
                    <button class="copy-btn" onclick="copyTaskCommand('{{ job.id }}', '{{ job.name }}', event)" title="Copy script path">📋</button>
                </div>
                <div class="job-info">
                    <div class="info-row">
                        <span class="info-label">Next Run:</span>
                        <span class="info-value next-run" data-timestamp="{{ job.next_run if job.next_run else '' }}">
                            {% if job.next_run %}
                                Loading...
                            {% else %}
                                Not scheduled
                            {% endif %}
                        </span>
                    </div>
                    {% set task_id = job.id.replace('task_', '') %}
                    {% if status.last_runs.get(task_id|int) %}
                    <div class="info-row">
                        <span class="info-label">Last Run:</span>
                        <span class="info-value last-run">
                            {{ status.last_runs[task_id|int].split('T')[0] }} {{ status.last_runs[task_id|int].split('T')[1].split('.')[0] }}
                        </span>
                    </div>
                    {% endif %}
                    {% if status.last_results.get(task_id|int) %}
                    <div class="info-row">
                        <span class="info-label">Status:</span>
                        <span class="info-value {% if status.last_results[task_id|int].status == 'success' %}success{% else %}error{% endif %}">
                            {{ status.last_results[task_id|int].status|upper }}
                        </span>
                    </div>
                    {% endif %}
                    {% if status.errors.get(task_id|int) %}
                    <div class="info-row">
                        <span class="info-label">Error:</span>
                        <span class="info-value error">
                            {{ status.errors[task_id|int].error[:50] }}...
                        </span>
                    </div>
                    {% endif %}
                </div>
                {% if job.next_run %}
                <div class="countdown" id="countdown-{{ job.id }}" data-next-run="{{ job.next_run }}">
                    <div class="countdown-label">Next run in</div>
                    <div class="countdown-time">--:--:--</div>
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-jobs">
            No scheduled jobs found
        </div>
        {% endif %}

        <div class="timestamp">
            Last updated: {{ now() }}
        </div>
    </div>

    <script>
        function copyTaskCommand(taskId, taskName, event) {
            // Map task IDs to script paths
            const scriptMap = {
                '1': 'C:\\StockApp\\backend\\turbomode\\core_engine\\ingest_master_market_data.py',
                '2': 'C:\\StockApp\\backend\\turbomode\\core_engine\\train_all_sectors_fastmode_orchestrator.py',
                '3': 'C:\\StockApp\\backend\\turbomode\\core_engine\\overnight_scanner.py',
                '3A': 'C:\\StockApp\\backend\\turbomode\\core_engine\\overnight_scanner.py',
                '3B': 'C:\\StockApp\\backend\\turbomode\\core_engine\\overnight_scanner.py',
                '3C': 'C:\\StockApp\\backend\\turbomode\\core_engine\\overnight_scanner.py',
                '4': 'C:\\StockApp\\backend\\turbomode\\core_engine\\generate_backtest_data.py',
                '5': 'C:\\StockApp\\backend\\turbomode\\core_engine\\adaptive_stock_ranker.py',
                '6': 'No script path (runs inline)',
                '7': 'No script path (runs inline)'
            };

            const cleanTaskId = taskId.replace('task_', '');
            const path = scriptMap[cleanTaskId] || 'Unknown task';

            navigator.clipboard.writeText(path).then(() => {
                // Show a brief confirmation
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✓';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 1000);
            }).catch(err => {
                alert('Failed to copy: ' + err);
            });
        }

        function runTask(taskId, taskName) {
            // Open modal
            const modal = document.getElementById('outputModal');
            const modalTitle = document.getElementById('modalTitle');
            const taskStatus = document.getElementById('taskStatus');
            const outputContent = document.getElementById('outputContent');

            modalTitle.textContent = 'Running: ' + taskName;
            taskStatus.textContent = 'Starting Task ' + taskId + '...';
            taskStatus.className = 'task-status status-running';
            outputContent.textContent = '';

            modal.style.display = 'block';

            // Trigger task execution
            fetch('/scheduler/run_task/' + taskId, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    taskStatus.textContent = 'Task Completed Successfully';
                    taskStatus.className = 'task-status status-success-task';

                    var lines = [];
                    lines.push('TASK EXECUTION RESULT');
                    lines.push('============================================================');
                    lines.push('');
                    lines.push('Task ID: ' + data.task_id);
                    lines.push('Status: SUCCESS');
                    lines.push('');

                    if (data.signals_generated !== undefined) {
                        lines.push('Signals Generated: ' + data.signals_generated);
                    }
                    if (data.symbols_scanned !== undefined) {
                        lines.push('Symbols Scanned: ' + data.symbols_scanned);
                    }
                    if (data.total_samples !== undefined) {
                        lines.push('Samples Generated: ' + data.total_samples);
                    }
                    if (data.top_10_symbols !== undefined) {
                        lines.push('Top 10 Symbols: ' + data.top_10_symbols.join(', '));
                    }
                    if (data.total_analyzed !== undefined) {
                        lines.push('Total Analyzed: ' + data.total_analyzed);
                    }
                    if (data.training_type !== undefined) {
                        lines.push('Training Type: ' + data.training_type);
                        lines.push('Total Models: ' + data.total_models);
                        lines.push('Sectors Trained: ' + data.sectors_trained);
                    }

                    lines.push('');
                    lines.push('============================================================');
                    lines.push('Task completed at ' + new Date().toLocaleString());

                    outputContent.textContent = lines.join('\\n');

                    setTimeout(function() {
                        location.reload();
                    }, 3000);
                } else {
                    taskStatus.textContent = 'Task Failed';
                    taskStatus.className = 'task-status status-error-task';

                    var lines = [];
                    lines.push('TASK EXECUTION FAILED');
                    lines.push('============================================================');
                    lines.push('');
                    lines.push('Task ID: ' + data.task_id);
                    lines.push('Status: FAILED');
                    lines.push('');
                    lines.push('Error: ' + (data.error || 'Unknown error'));
                    lines.push('');
                    lines.push('============================================================');
                    lines.push('Failed at ' + new Date().toLocaleString());

                    outputContent.textContent = lines.join('\\n');
                }
            })
            .catch(error => {
                taskStatus.textContent = 'Request Failed';
                taskStatus.className = 'task-status status-error-task';
                outputContent.textContent = 'Network error: ' + error;
            });
        }

        function closeModal() {
            document.getElementById('outputModal').style.display = 'none';
        }

        // Close modal when clicking outside of it
        window.onclick = function(event) {
            const modal = document.getElementById('outputModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }

        // Format timestamp strings to human-friendly format
        function formatTimestamp(timestamp) {
            if (!timestamp) return '';
            // Format as "Month Day, Time"
            const date = new Date(timestamp);
            const month = date.toLocaleDateString('en-US', { month: 'long' });
            const day = date.getDate();
            const time = date.toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true
            });
            return `${month} ${day}, ${time}`;
        }

        // Format all next-run timestamps on page load
        document.querySelectorAll('.next-run').forEach(elem => {
            const timestamp = elem.getAttribute('data-timestamp');
            if (timestamp && elem.textContent === 'Loading...') {
                elem.textContent = formatTimestamp(timestamp);
            }
        });

        // Countdown timer logic
        function updateCountdowns() {
            const countdowns = document.querySelectorAll('.countdown');

            countdowns.forEach(countdown => {
                const nextRunStr = countdown.getAttribute('data-next-run');
                if (!nextRunStr) return;

                // Parse the next run time (ISO format)
                const nextRun = new Date(nextRunStr);
                const now = new Date();
                const diff = nextRun - now;

                const timeDisplay = countdown.querySelector('.countdown-time');

                if (diff <= 0) {
                    // Task should be running or just finished
                    timeDisplay.textContent = 'Running...';
                    timeDisplay.style.color = '#fbbf24';
                } else {
                    // Calculate hours, minutes, seconds
                    const hours = Math.floor(diff / (1000 * 60 * 60));
                    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                    const seconds = Math.floor((diff % (1000 * 60)) / 1000);

                    // Format as HH:MM:SS
                    const hoursStr = String(hours).padStart(2, '0');
                    const minutesStr = String(minutes).padStart(2, '0');
                    const secondsStr = String(seconds).padStart(2, '0');

                    timeDisplay.textContent = `${hoursStr}:${minutesStr}:${secondsStr}`;
                    timeDisplay.style.color = 'white';
                }
            });
        }

        // Update countdowns every second
        setInterval(updateCountdowns, 1000);
        updateCountdowns(); // Run immediately on load

        // AJAX refresh to update next_run times without page reload
        function refreshSchedulerData() {
            fetch('/scheduler/status?format=json')
                .then(response => response.json())
                .then(data => {
                    // Update next_run times for each job
                    data.jobs.forEach(job => {
                        const countdown = document.getElementById('countdown-' + job.id);
                        if (countdown && job.next_run) {
                            countdown.setAttribute('data-next-run', job.next_run);
                        }

                        // Update "Next Run" display
                        const nextRunElement = document.querySelector(`#countdown-${job.id}`).closest('.job-card').querySelector('.next-run');
                        if (nextRunElement && job.next_run) {
                            nextRunElement.textContent = formatTimestamp(job.next_run);
                            nextRunElement.setAttribute('data-timestamp', job.next_run);
                        }
                    });

                    // Update last run times and status
                    data.jobs.forEach(job => {
                        const taskId = job.id.replace('task_', '');
                        const taskIdInt = parseInt(taskId) || taskId;
                        const jobCard = document.querySelector(`#countdown-${job.id}`).closest('.job-card');

                        // Update last run if it exists in the data
                        if (data.last_runs && data.last_runs[taskIdInt]) {
                            const lastRunElement = jobCard.querySelector('.last-run');
                            if (lastRunElement) {
                                const parts = data.last_runs[taskIdInt].split('T');
                                const datePart = parts[0];
                                const timePart = parts[1] ? parts[1].split('.')[0] : '';
                                lastRunElement.textContent = `${datePart} ${timePart}`;
                            }
                        }
                    });

                    // Update timestamp
                    const timestampElement = document.querySelector('.timestamp');
                    if (timestampElement) {
                        const now = new Date();
                        const formatted = now.getFullYear() + '-' +
                            String(now.getMonth() + 1).padStart(2, '0') + '-' +
                            String(now.getDate()).padStart(2, '0') + ' ' +
                            String(now.getHours()).padStart(2, '0') + ':' +
                            String(now.getMinutes()).padStart(2, '0') + ':' +
                            String(now.getSeconds()).padStart(2, '0');
                        timestampElement.textContent = 'Last updated: ' + formatted;
                    }
                })
                .catch(err => {
                    console.error('Failed to refresh scheduler data:', err);
                });
        }

        // Auto-refresh data every 10 seconds via AJAX
        setInterval(refreshSchedulerData, 10000);
    </script>
</body>
</html>
"""

def now():
    """Helper function for template to get current timestamp"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_last_execution_info(status):
    """
    Analyze scheduler status and return info about the most recent task execution.

    Focuses on overnight tasks (Task 1 Ingestion and Task 3 Scanner) to show
    what happened last night.

    Args:
        status: Scheduler status dictionary

    Returns:
        Dictionary with notification info or None
    """
    from datetime import datetime, timedelta

    # Priority order: Check overnight tasks first (3=Scanner, 1=Ingestion)
    # Then check any other recently completed tasks
    priority_tasks = [3, 1, 5, 4, 6, 2, 7]

    last_runs = status.get('last_runs', {})
    last_results = status.get('last_results', {})
    errors = status.get('errors', {})

    # Task names for display
    task_names = {
        1: "Master Market Data Ingestion",
        2: "TurboMode Training Orchestrator",
        3: "Overnight Scanner",
        4: "Backtest Data Generator",
        5: "Adaptive Stock Ranking",
        6: "Drift Monitoring System",
        7: "Weekly Maintenance"
    }

    # Find the most recent execution from priority tasks
    most_recent = None
    most_recent_time = None

    for task_id in priority_tasks:
        task_id_str = str(task_id)
        task_id_int = int(task_id)

        # Check if task has run
        if task_id_int in last_runs or task_id_str in last_runs:
            run_time_str = last_runs.get(task_id_int) or last_runs.get(task_id_str)

            if run_time_str:
                try:
                    run_time = datetime.fromisoformat(run_time_str.replace('Z', '+00:00'))

                    # Only show tasks from last 24 hours
                    if (datetime.now() - run_time).total_seconds() < 86400:
                        if most_recent_time is None or run_time > most_recent_time:
                            most_recent_time = run_time
                            most_recent = task_id_int
                except:
                    pass

    # If no recent execution found, return None
    if most_recent is None:
        return None

    # Build notification info
    task_name = task_names.get(most_recent, f"Task {most_recent}")
    result = last_results.get(most_recent, {})
    error = errors.get(most_recent, {})

    # Determine notification type and message
    if error:
        # Task failed
        return {
            'type': 'error',
            'title': f'Last Night: {task_name} FAILED',
            'message': f'Error: {error.get("error", "Unknown error")[:100]}',
            'time': f'Failed at {most_recent_time.strftime("%Y-%m-%d %I:%M %p")}'
        }
    elif result.get('status') == 'success':
        # Task succeeded
        details = []

        # Add specific details based on task type
        if most_recent == 3:  # Scanner
            signals = result.get('signals_generated', 0)
            details.append(f'Generated {signals} trading signals')
        elif most_recent == 1:  # Ingestion
            symbols = result.get('symbols_processed', 0)
            details.append(f'Updated {symbols} symbols')
        elif most_recent == 5:  # Ranking
            top10 = result.get('top_10_count', 0)
            details.append(f'Ranked {top10} top stocks')

        message = ' • '.join(details) if details else 'Completed successfully'

        return {
            'type': 'success',
            'title': f'Last Night: {task_name} Completed Successfully',
            'message': message,
            'time': f'Completed at {most_recent_time.strftime("%Y-%m-%d %I:%M %p")}'
        }
    else:
        # Unknown status
        return {
            'type': 'warning',
            'title': f'Last Night: {task_name} Status Unknown',
            'message': 'Task may still be running or status unavailable',
            'time': f'Started at {most_recent_time.strftime("%Y-%m-%d %I:%M %p")}'
        }

    return None


def init_unified_scheduler_api(app):
    """
    Initialize unified scheduler API endpoints in Flask app

    Args:
        app: Flask application instance

    Returns:
        Flask app with scheduler endpoints registered
    """

    # ========================================================================
    # SCHEDULER CONTROL ENDPOINTS
    # ========================================================================

    @app.route('/scheduler/status', methods=['GET'])
    def scheduler_status():
        """
        GET /scheduler/status

        Get current scheduler status including all jobs and their next run times

        Returns:
            HTML page for browser, JSON for API calls
        """
        try:
            status = get_scheduler_status()

            # Check if request is from browser (wants HTML) or API (wants JSON)
            if request.headers.get('Accept', '').find('text/html') != -1:
                # Return HTML page for human-friendly viewing
                from flask import render_template_string
                return render_template_string(
                    SCHEDULER_STATUS_HTML,
                    status=status,
                    now=now,
                    get_last_execution_info=get_last_execution_info
                )
            else:
                # Return JSON for API calls
                return jsonify(status), 200
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/start', methods=['POST'])
    def scheduler_start():
        """
        POST /scheduler/start

        Start the unified scheduler (if not already running)

        Returns:
            JSON with success status
        """
        try:
            success = start_unified_scheduler()

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Unified scheduler started'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler already running'
                }), 200
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/stop', methods=['POST'])
    def scheduler_stop():
        """
        POST /scheduler/stop

        Stop the unified scheduler

        Returns:
            JSON with success status
        """
        try:
            success = stop_unified_scheduler()

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Unified scheduler stopped'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Scheduler not running'
                }), 200
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # ========================================================================
    # MANUAL TASK EXECUTION ENDPOINTS
    # ========================================================================

    @app.route('/scheduler/run_ingestion', methods=['POST'])
    def manual_run_ingestion():
        """
        POST /scheduler/run_ingestion

        Manually trigger Task 1: Master Market Data Ingestion

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=1)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running ingestion: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_orchestrator', methods=['POST'])
    def manual_run_orchestrator():
        """
        POST /scheduler/run_orchestrator

        Manually trigger Task 2: TurboMode Training Orchestrator

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=2)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_overnight_scanner', methods=['POST'])
    def manual_run_overnight_scanner():
        """
        POST /scheduler/run_overnight_scanner

        Manually trigger Task 3: Overnight Scanner

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=3)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running overnight scanner: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_backtest_generator', methods=['POST'])
    def manual_run_backtest_generator():
        """
        POST /scheduler/run_backtest_generator

        Manually trigger Task 4: Backtest Data Generator

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=4)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running backtest generator: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_drift_monitor', methods=['POST'])
    def manual_run_drift_monitor():
        """
        POST /scheduler/run_drift_monitor

        Manually trigger Task 5: Drift Monitoring System

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=5)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running drift monitor: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_weekly_maintenance', methods=['POST'])
    def manual_run_weekly_maintenance():
        """
        POST /scheduler/run_weekly_maintenance

        Manually trigger Task 6: Weekly Maintenance

        Returns:
            JSON with task results
        """
        try:
            result = run_task_manually(task_id=6)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running weekly maintenance: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/run_task/<int:task_id>', methods=['POST'])
    def manual_run_task_by_id(task_id):
        """
        POST /scheduler/run_task/<task_id>

        Manually trigger any task by ID

        Args:
            task_id: Task ID (1-6)

        Returns:
            JSON with task results
        """
        try:
            if task_id not in TASK_FUNCTIONS:
                return jsonify({
                    'success': False,
                    'error': f'Invalid task_id: {task_id}. Valid IDs: 1-6'
                }), 400

            result = run_task_manually(task_id=task_id)
            return jsonify(result), 200 if result.get('success') else 500
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/scheduler/history', methods=['GET'])
    def scheduler_history():
        """
        GET /scheduler/history

        Display task execution history in human-friendly format
        """
        from backend.unified_scheduler import load_execution_history
        from flask import render_template_string

        history = load_execution_history()

        # Sort by timestamp descending (most recent first)
        history_sorted = sorted(history, key=lambda x: x['timestamp'], reverse=True)

        # Limit to last 100 executions
        history_sorted = history_sorted[:100]

        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Execution History</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }

        .back-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
            margin-top: 10px;
        }

        .back-link:hover {
            text-decoration: underline;
        }

        .history-table {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }

        tr:hover {
            background: #f9fafb;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .status-success {
            background: #d1fae5;
            color: #065f46;
        }

        .status-failed {
            background: #fee2e2;
            color: #991b1b;
        }

        .status-timeout {
            background: #fef3c7;
            color: #92400e;
        }

        .task-id {
            font-weight: 600;
            color: #667eea;
        }

        .duration {
            color: #6b7280;
            font-size: 14px;
        }

        .error-text {
            color: #dc2626;
            font-size: 12px;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .timestamp {
            color: #6b7280;
            font-size: 14px;
        }

        .no-history {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Task Execution History</h1>
            <p style="color: #6b7280; margin-top: 10px;">Last 100 task executions (30-day retention)</p>
            <a href="/scheduler/status" class="back-link">← Back to Scheduler Status</a>
        </div>

        <div class="history-table">
            {% if history %}
            <table>
                <thead>
                    <tr>
                        <th>Task ID</th>
                        <th>Task Name</th>
                        <th>Status</th>
                        <th>Timestamp</th>
                        <th>Duration</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {% for entry in history %}
                    <tr>
                        <td class="task-id">{{ entry.task_id }}</td>
                        <td>{{ entry.task_name }}</td>
                        <td>
                            <span class="status-badge status-{{ entry.status }}">
                                {{ entry.status.upper() }}
                            </span>
                        </td>
                        <td class="timestamp">
                            {{ entry.timestamp.split('T')[0] }} {{ entry.timestamp.split('T')[1].split('.')[0] }}
                        </td>
                        <td class="duration">
                            {% if entry.duration_seconds %}
                                {{ "%.1f"|format(entry.duration_seconds) }}s
                            {% else %}
                                -
                            {% endif %}
                        </td>
                        <td class="error-text" title="{{ entry.error or '' }}">
                            {{ entry.error or '-' }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="no-history">
                <p>No execution history available</p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
        """

        return render_template_string(html, history=history_sorted)

    # ========================================================================
    # SCHEDULER AUTO-START ON FLASK INIT
    # ========================================================================

    logger.info("=" * 80)
    logger.info("UNIFIED SCHEDULER - INITIALIZING")
    logger.info("=" * 80)

    try:
        # Auto-start scheduler when Flask starts
        start_unified_scheduler()
        logger.info("[OK] Unified scheduler started automatically")
    except Exception as e:
        logger.error(f"[ERROR] Failed to auto-start scheduler: {e}")

    logger.info("=" * 80)

    return app
