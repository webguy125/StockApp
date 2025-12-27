/**
 * ML Trading System Page
 * Displays signals from the modular ML pipeline (separate from agent system)
 */

let currentTimeframe = 'all';
let mlSignals = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadMLSignals();
    loadTrainingProgress();  // Load training progress tracker
    checkActiveOnLoad();  // Resume scan progress if active
});

/**
 * Load ML signals from backend
 */
async function loadMLSignals() {
    showLoading(true);

    try {
        const response = await fetch('/ml-signals');
        const data = await response.json();

        if (data.status === 'no_data') {
            showEmptyState();
            return;
        }

        mlSignals = data;
        renderSignals();
        updateMetadata(data);
        showLoading(false);

    } catch (error) {
        console.error('Error loading ML signals:', error);
        showEmptyState();
    }
}

/**
 * Render signals based on current timeframe
 */
function renderSignals() {
    if (!mlSignals) return;

    // Update tab counts
    const allSignalsCount = (mlSignals.all_signals || mlSignals.signals || []).length;
    document.getElementById('count-intraday').textContent = mlSignals.intraday?.length || 0;
    document.getElementById('count-daily').textContent = mlSignals.daily?.length || 0;
    document.getElementById('count-monthly').textContent = mlSignals.monthly?.length || 0;

    // Update "All Signals" tab to show total count with scan info
    const allTab = document.querySelector('[data-timeframe="all"]');
    if (allTab && allSignalsCount > 0) {
        if (mlSignals.scan_metadata && mlSignals.scan_metadata.total_scanned) {
            const scanned = mlSignals.scan_metadata.total_scanned;
            allTab.innerHTML = `All Signals <span class="tab-count">${allSignalsCount} of ${scanned}</span>`;
        } else {
            allTab.innerHTML = `All Signals <span class="tab-count">${allSignalsCount}</span>`;
        }
    }

    // Clear all grids
    ['intraday', 'daily', 'monthly'].forEach(tf => {
        document.getElementById(`grid-${tf}`).innerHTML = '';
    });

    // Render based on selected timeframe
    if (currentTimeframe === 'all') {
        // Show ALL signals regardless of category
        const allSignals = mlSignals.all_signals || mlSignals.signals || [];

        if (allSignals.length > 0) {
            // Render all signals in intraday section
            renderTimeframeSignals('intraday', allSignals);
            document.getElementById(`section-intraday`).style.display = 'block';

            // Update header to show it's all signals
            const header = document.querySelector('#section-intraday .section-header h2');
            if (header) header.textContent = `All Signals (${allSignals.length})`;

            // Hide other sections
            document.getElementById(`section-daily`).style.display = 'none';
            document.getElementById(`section-monthly`).style.display = 'none';
        } else {
            // If no all_signals, try to show categorized
            renderTimeframeSignals('intraday', mlSignals.intraday || []);
            renderTimeframeSignals('daily', mlSignals.daily || []);
            renderTimeframeSignals('monthly', mlSignals.monthly || []);

            // Show all sections
            ['intraday', 'daily', 'monthly'].forEach(tf => {
                document.getElementById(`section-${tf}`).style.display = 'block';
            });
        }
    } else {
        // Hide other sections
        ['intraday', 'daily', 'monthly'].forEach(tf => {
            document.getElementById(`section-${tf}`).style.display = tf === currentTimeframe ? 'block' : 'none';
        });

        // Render only selected timeframe
        renderTimeframeSignals(currentTimeframe, mlSignals[currentTimeframe] || []);
    }

    // Show footer
    document.getElementById('heatmap-footer').style.display = 'flex';
}

/**
 * Render signals for a specific timeframe
 */
function renderTimeframeSignals(timeframe, signals) {
    const grid = document.getElementById(`grid-${timeframe}`);
    grid.innerHTML = '';

    if (!signals || signals.length === 0) {
        grid.innerHTML = '<div class="no-signals">No signals for this timeframe</div>';
        return;
    }

    signals.forEach(signal => {
        const card = createSignalCard(signal);
        grid.appendChild(card);
    });
}

/**
 * Create a signal card element
 */
function createSignalCard(signal) {
    const card = document.createElement('div');
    card.className = 'heatmap-card';

    // Calculate color based on score
    const scoreColor = getScoreColor(signal.score);
    card.style.borderLeft = `4px solid ${scoreColor}`;

    // Format direction
    const directionClass = signal.direction === 'bullish' ? 'bullish' : signal.direction === 'bearish' ? 'bearish' : 'neutral';
    const directionIcon = signal.direction === 'bullish' ? '📈' : signal.direction === 'bearish' ? '📉' : '📊';

    card.innerHTML = `
        <div class="card-header">
            <div class="card-symbol">${signal.symbol}</div>
            <div class="card-score" style="background: ${scoreColor};">
                ${(signal.score * 100).toFixed(0)}
            </div>
        </div>
        <div class="card-details">
            <div class="detail-row">
                <span>Direction:</span>
                <span class="${directionClass}">${directionIcon} ${signal.direction}</span>
            </div>
            <div class="detail-row">
                <span>Confidence:</span>
                <span>${(signal.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="detail-row">
                <span>ML Prediction:</span>
                <span class="ml-prediction">${signal.ml_prediction || 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span>ML Confidence:</span>
                <span>${signal.ml_confidence ? (signal.ml_confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span>Price:</span>
                <span>$${signal.price?.toFixed(2) || 'N/A'}</span>
            </div>
            <div class="detail-row">
                <span>Analyzers:</span>
                <span>${Object.keys(signal.analyzers || {}).length}</span>
            </div>
        </div>
        <div class="card-actions">
            <button class="card-btn" onclick="viewSignalDetails('${signal.symbol}')">
                View Details
            </button>
            <button class="card-btn" onclick="openChart('${signal.symbol}')">
                Open Chart
            </button>
        </div>
    `;

    return card;
}

/**
 * Get color based on score
 */
function getScoreColor(score) {
    if (score >= 0.7) return '#10b981'; // Green
    if (score >= 0.6) return '#3b82f6'; // Blue
    if (score >= 0.5) return '#f59e0b'; // Orange
    if (score >= 0.4) return '#ef4444'; // Red
    return '#6b7280'; // Gray
}

/**
 * Update metadata footer
 */
function updateMetadata(data) {
    document.getElementById('meta-system').textContent = 'ML Trading System';
    document.getElementById('meta-timestamp').textContent = data.timestamp ? new Date(data.timestamp).toLocaleString() : 'Never';

    // Show scan metadata if available
    if (data.scan_metadata) {
        const scanned = data.scan_metadata.total_scanned || 0;
        const displayed = data.scan_metadata.top_displayed || data.total_count || 0;
        const confidence = (data.scan_metadata.min_confidence_threshold * 100).toFixed(0);

        document.getElementById('meta-total').textContent = `${displayed} (scanned ${scanned})`;
        document.getElementById('meta-model').textContent = `Min ${confidence}% confidence`;
    } else {
        document.getElementById('meta-total').textContent = data.total_count || 0;
        document.getElementById('meta-model').textContent = data.total_count > 0 ? 'Trained' : 'Untrained';
    }
}

/**
 * Switch timeframe view
 */
function switchTimeframe(timeframe) {
    currentTimeframe = timeframe;

    // Update active tab
    document.querySelectorAll('.timeframe-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-timeframe="${timeframe}"]`).classList.add('active');

    // Re-render
    renderSignals();
}

/**
 * Show/hide loading state
 */
function showLoading(show) {
    document.getElementById('loading-state').style.display = show ? 'flex' : 'none';
    document.getElementById('empty-state').style.display = 'none';
    document.querySelector('.heatmap-content').style.display = show ? 'none' : 'block';
}

/**
 * Show empty state
 */
function showEmptyState() {
    document.getElementById('loading-state').style.display = 'none';
    document.getElementById('empty-state').style.display = 'flex';
    document.getElementById('heatmap-footer').style.display = 'none';
}

/**
 * Refresh signals
 */
async function refreshSignals() {
    const icon = document.getElementById('refresh-icon');
    icon.style.animation = 'spin 1s linear';

    await loadMLSignals();

    setTimeout(() => {
        icon.style.animation = '';
    }, 1000);
}

// Track current scan job
let currentScanJobId = null;
let progressPollInterval = null;

/**
 * Run ML scan with background progress tracking
 */
async function runMLScan() {
    const btn = document.getElementById('scan-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Starting scan...';

    try {
        // Start the scan
        const response = await fetch('/ml-scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                max_stocks: 500,
                include_crypto: true
            })
        });

        const data = await response.json();

        if (data.status === 'started') {
            // Save job ID and start polling
            currentScanJobId = data.job_id;
            startProgressPolling();
        } else {
            alert(`❌ Scan failed: ${data.error || 'Unknown error'}`);
            btn.disabled = false;
            btn.innerHTML = '🔍 Run Scan';
        }

    } catch (error) {
        console.error('Scan error:', error);
        alert(`Error: ${error.message}`);
        btn.disabled = false;
        btn.innerHTML = '🔍 Run Scan';
    }
}

/**
 * Start polling for scan progress
 */
function startProgressPolling() {
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
    }

    // Poll every 2 seconds
    progressPollInterval = setInterval(checkScanProgress, 2000);

    // Check immediately
    checkScanProgress();
}

/**
 * Check scan progress
 */
async function checkScanProgress() {
    if (!currentScanJobId) return;

    try {
        const response = await fetch(`/ml-scan-progress/${currentScanJobId}`);
        const data = await response.json();

        if (data.status === 'success' && data.job) {
            const job = data.job;
            const btn = document.getElementById('scan-btn');

            if (job.status === 'running') {
                // Update button with progress
                const percent = job.percentage.toFixed(0);
                const current = job.progress;
                const total = job.total;
                const symbol = job.current_symbol || '...';

                btn.innerHTML = `⏳ Scanning... ${percent}% (${current}/${total}) - ${symbol}`;

            } else if (job.status === 'completed') {
                // Scan complete!
                stopProgressPolling();

                const btn = document.getElementById('scan-btn');
                btn.innerHTML = '✅ Scan Complete!';

                // Reload signals
                await loadMLSignals();

                // Show success message
                setTimeout(() => {
                    alert(`✅ Scan complete!\n\nScanned ${job.total} symbols\nGenerated ${job.signals_count} signals\n\nResults are now visible below.`);
                    btn.disabled = false;
                    btn.innerHTML = '🔍 Run Scan';
                }, 500);

            } else if (job.status === 'failed') {
                // Scan failed
                stopProgressPolling();

                const btn = document.getElementById('scan-btn');
                btn.innerHTML = '❌ Scan Failed';
                btn.disabled = false;

                alert(`❌ Scan failed:\n\n${job.error || 'Unknown error'}`);

                setTimeout(() => {
                    btn.innerHTML = '🔍 Run Scan';
                }, 2000);
            }
        }

    } catch (error) {
        console.error('Progress check error:', error);
    }
}

/**
 * Stop progress polling
 */
function stopProgressPolling() {
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
    currentScanJobId = null;
}

/**
 * Check for active scan on page load
 */
async function checkActiveOnLoad() {
    try {
        const response = await fetch('/ml-scan-active');
        const data = await response.json();

        if (data.status === 'success' && data.active && data.job) {
            // Resume tracking active scan
            currentScanJobId = data.job.job_id;
            startProgressPolling();

            const btn = document.getElementById('scan-btn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Resuming scan...';
        }

    } catch (error) {
        console.error('Active check error:', error);
    }
}

/**
 * Show performance statistics
 */
async function showStats() {
    try {
        const response = await fetch('/ml-stats');
        const data = await response.json();

        if (data.status === 'success') {
            const stats = data.stats;

            // Update stat cards
            document.getElementById('stat-total-trades').textContent = stats.total_trades || 0;
            document.getElementById('stat-win-rate').textContent = (stats.win_rate || 0).toFixed(1) + '%';
            document.getElementById('stat-total-pl').textContent = '$' + (stats.total_profit_loss || 0).toFixed(2);
            document.getElementById('stat-profit-factor').textContent = (stats.profit_factor || 0).toFixed(2);

            // Apply color classes
            const plElement = document.getElementById('stat-total-pl');
            plElement.classList.remove('positive', 'negative');
            if (stats.total_profit_loss > 0) plElement.classList.add('positive');
            else if (stats.total_profit_loss < 0) plElement.classList.add('negative');

            // Show stats section
            document.getElementById('system-stats').style.display = 'flex';
        }

    } catch (error) {
        console.error('Stats error:', error);
    }
}

/**
 * View signal details
 */
function viewSignalDetails(symbol) {
    const signal = findSignal(symbol);
    if (!signal) return;

    // Create details modal (simple alert for now)
    const details = `
Symbol: ${signal.symbol}
Score: ${(signal.score * 100).toFixed(1)}%
Confidence: ${(signal.confidence * 100).toFixed(1)}%
Direction: ${signal.direction}
ML Prediction: ${signal.ml_prediction || 'N/A'}
ML Confidence: ${signal.ml_confidence ? (signal.ml_confidence * 100).toFixed(1) + '%' : 'N/A'}
Price: $${signal.price?.toFixed(2) || 'N/A'}

Analyzers Active: ${Object.keys(signal.analyzers || {}).join(', ')}
    `;

    alert(details);
}

/**
 * Open chart for symbol
 */
function openChart(symbol) {
    window.location.href = `/?symbol=${symbol}`;
}

/**
 * Find signal by symbol
 */
function findSignal(symbol) {
    if (!mlSignals) return null;

    const allSignals = [
        ...(mlSignals.intraday || []),
        ...(mlSignals.daily || []),
        ...(mlSignals.monthly || [])
    ];

    return allSignals.find(s => s.symbol === symbol);
}

/**
 * Run full ML cycle (scan + learner + retrain)
 */
async function runFullCycle() {
    const btn = document.getElementById('full-cycle-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Running Full Cycle...';

    try {
        const response = await fetch('/ml-run-full-cycle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert('✅ Full cycle completed!\n\n' +
                  'Scan → Simulate → Check Outcomes → Retrain\n\n' +
                  'Check the signals and stats for updated results.');
            await loadMLSignals();
            await showStats();
        } else {
            alert(`❌ Full cycle failed:\n\n${data.message || data.error}`);
        }

    } catch (error) {
        console.error('Full cycle error:', error);
        alert(`❌ Error running full cycle:\n\n${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🚀 Run Full Cycle';
    }
}

/**
 * Run learner only (uses last scan results)
 */
async function runLearnerOnly() {
    const btn = document.getElementById('learner-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Running Learner...';

    try {
        const response = await fetch('/ml-run-learner-only', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert('✅ Learner completed!\n\n' +
                  'Checked outcomes, simulated new trades, and retrained if needed.\n\n' +
                  'Check stats for updated performance.');
            await showStats();
        } else {
            alert(`❌ Learner failed:\n\n${data.message || data.error}`);
        }

    } catch (error) {
        console.error('Learner error:', error);
        alert(`❌ Error running learner:\n\n${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🤖 Run Learner';
    }
}

/**
 * Check automation status (built-in scheduler)
 */
async function checkAutomationStatus() {
    const btn = document.getElementById('automation-status-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳...';

    try {
        const response = await fetch('/ml-automation-status');
        const data = await response.json();

        let message = '';

        if (data.status === 'enabled' && data.enabled) {
            const nextRun = data.next_run ? new Date(data.next_run).toLocaleString() : 'Unknown';
            const lastRun = data.last_run ? new Date(data.last_run).toLocaleString() : 'Never';

            message = `✅ AUTOMATION ENABLED\n\n` +
                      `Schedule Time: ${data.schedule_time} (daily)\n` +
                      `Next Run: ${nextRun}\n` +
                      `Last Run: ${lastRun}\n\n` +
                      `The system runs automatically inside Flask.\n` +
                      `Works on any hosting platform!\n\n` +
                      `Click 'Stop Auto' to disable.`;
        } else if (data.status === 'disabled') {
            message = `⚠️ AUTOMATION DISABLED\n\n` +
                      `${data.message}\n\n` +
                      `Click '▶️ Start Auto' to enable daily automation.\n` +
                      `System will run every day at 6 PM automatically.`;
        } else if (data.status === 'unavailable') {
            message = `ℹ️ AUTOMATION UNAVAILABLE\n\n` +
                      `${data.message}\n\n` +
                      `Install APScheduler:\n` +
                      `pip install APScheduler`;
        } else {
            message = `❓ UNKNOWN STATUS\n\n${JSON.stringify(data, null, 2)}`;
        }

        alert(message);

    } catch (error) {
        console.error('Automation status error:', error);
        alert(`❌ Error checking automation status:\n\n${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚙️ Status';
    }
}

/**
 * Start automation (built-in scheduler)
 */
async function startAutomation() {
    const btn = document.getElementById('automation-start-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Starting...';

    try {
        // Prompt for schedule time (optional)
        const scheduleTime = prompt('Enter daily run time (24-hour format):', '18:00');

        if (!scheduleTime) {
            btn.disabled = false;
            btn.innerHTML = '▶️ Start Auto';
            return;
        }

        const response = await fetch('/ml-automation-start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ schedule_time: scheduleTime })
        });

        const data = await response.json();

        if (data.status === 'success') {
            const nextRun = data.next_run ? new Date(data.next_run).toLocaleString() : 'Unknown';

            alert(`✅ AUTOMATION STARTED!\n\n` +
                  `Schedule: Every day at ${data.schedule_time}\n` +
                  `Next Run: ${nextRun}\n\n` +
                  `The system will now run automatically.\n` +
                  `Runs inside Flask - works on any platform!\n\n` +
                  `NO MANUAL CLICKING NEEDED!`);
        } else {
            alert(`❌ Failed to start automation:\n\n${data.message || data.error}`);
        }

    } catch (error) {
        console.error('Start automation error:', error);
        alert(`❌ Error starting automation:\n\n${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '▶️ Start Auto';
    }
}

/**
 * Stop automation (built-in scheduler)
 */
async function stopAutomation() {
    const btn = document.getElementById('automation-stop-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Stopping...';

    try {
        if (!confirm('Stop daily automation?\n\nYou can still run cycles manually.')) {
            btn.disabled = false;
            btn.innerHTML = '⏸️ Stop Auto';
            return;
        }

        const response = await fetch('/ml-automation-stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`✅ AUTOMATION STOPPED\n\n` +
                  `Daily automation has been disabled.\n\n` +
                  `You can still run cycles manually using the buttons above.`);
        } else {
            alert(`❌ Failed to stop automation:\n\n${data.message || data.error}`);
        }

    } catch (error) {
        console.error('Stop automation error:', error);
        alert(`❌ Error stopping automation:\n\n${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⏸️ Stop Auto';
    }
}

/**
 * Load and update training progress tracker
 */
async function loadTrainingProgress() {
    try {
        const response = await fetch('/ml-training-progress');
        const data = await response.json();

        if (data.status === 'success') {
            updateTrainingProgress(data);
        }
    } catch (error) {
        console.error('Error loading training progress:', error);
    }
}

/**
 * Update training progress UI
 */
function updateTrainingProgress(data) {
    const progressContainer = document.getElementById('training-progress');

    // Show the tracker
    progressContainer.style.display = 'block';

    const totalTrades = data.total_trades || 0;
    const expertTrades = 840;
    const winRate = data.win_rate || 0;
    const progressPct = Math.min((totalTrades / expertTrades) * 100, 100);

    // Calculate current level
    let level = 'Untrained';
    if (totalTrades >= 840) level = '🧠 Expert';
    else if (totalTrades >= 420) level = '📈 Well Trained';
    else if (totalTrades >= 100) level = '🎓 Trained';
    else if (totalTrades >= 30) level = '📊 Learning';

    // Update progress stats
    document.getElementById('progress-trades').textContent = `${totalTrades} / ${expertTrades}`;
    document.getElementById('progress-level').textContent = level;
    document.getElementById('progress-winrate').textContent = `${winRate.toFixed(1)}%`;

    // Calculate days until expert
    const firstRunDate = new Date('2025-12-20'); // Tomorrow
    const expertDate = new Date(firstRunDate);
    expertDate.setDate(expertDate.getDate() + 43); // 43 days from start

    const today = new Date();
    const daysUntilExpert = Math.max(0, Math.ceil((expertDate - today) / (1000 * 60 * 60 * 24)));

    document.getElementById('progress-countdown').textContent =
        totalTrades >= expertTrades ? '✅ Complete!' : `${daysUntilExpert} days`;

    // Update progress bar
    document.getElementById('progress-bar').style.width = `${progressPct}%`;
    document.getElementById('progress-text').textContent = `${progressPct.toFixed(1)}% to Expert Level`;

    // Update milestone states
    updateMilestoneStates(totalTrades, firstRunDate);
}

/**
 * Update milestone states based on current progress
 */
function updateMilestoneStates(totalTrades, firstRunDate) {
    const today = new Date();

    const milestones = [
        { id: 'first-cycle', trades: 0, days: 1 },
        { id: 'first-closures', trades: 30, days: 15 },
        { id: 'first-training', trades: 100, days: 18 },
        { id: 'well-trained', trades: 420, days: 29 },
        { id: 'expert', trades: 840, days: 43 }
    ];

    milestones.forEach(milestone => {
        const element = document.getElementById(`milestone-${milestone.id}`);
        const milestoneDate = new Date(firstRunDate);
        milestoneDate.setDate(milestoneDate.getDate() + milestone.days - 1);

        // Remove existing classes
        element.classList.remove('completed', 'current');

        if (totalTrades >= milestone.trades) {
            // Milestone completed
            element.classList.add('completed');
        } else if (totalTrades >= (milestone.trades - 100) || today >= milestoneDate) {
            // Currently working on this milestone
            element.classList.add('current');
        }
    });
}

/**
 * Toggle training details visibility
 */
function toggleTrainingDetails() {
    const details = document.getElementById('training-details');
    const toggle = document.getElementById('training-details-toggle');

    if (details.style.display === 'none') {
        details.style.display = 'block';
        toggle.textContent = 'Hide Details ▲';
    } else {
        details.style.display = 'none';
        toggle.textContent = 'Show Details ▼';
    }
}

// CSS animation for refresh icon
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
