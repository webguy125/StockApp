// Trade Quality Analyzer - Frontend Logic
window.chartInstances = {};
const DEFAULT_START_MONTH = 2;
const DEFAULT_START_YEAR = 2026;
let fullTrades = [];

// ============================================================================
// Initialization
// ============================================================================
function loadTrades() {
    fetch('/api/performance/summary')
        .then(r => r.json())
        .then(data => {
            fullTrades = data.equity_curve;

            // Log enriched field availability
            const total = fullTrades.length;
            const withProbBuy = fullTrades.filter(t => t.prob_buy !== null && t.prob_buy !== undefined).length;
            const withProbSell = fullTrades.filter(t => t.prob_sell !== null && t.prob_sell !== undefined).length;
            const withProbHold = fullTrades.filter(t => t.prob_hold !== null && t.prob_hold !== undefined).length;
            const withATR = fullTrades.filter(t => t.entry_atr !== null && t.entry_atr !== undefined).length;
            const withRR = fullTrades.filter(t => t.rr !== null && t.rr !== undefined).length;
            const withDM = fullTrades.filter(t => t.directional_margin !== null && t.directional_margin !== undefined).length;

            console.log('[TRADE QUALITY] Loaded trades with enriched fields:');
            console.log(`  Total trades: ${total}`);
            console.log(`  prob_buy: ${withProbBuy}/${total} (${(100*withProbBuy/total).toFixed(1)}%)`);
            console.log(`  prob_sell: ${withProbSell}/${total} (${(100*withProbSell/total).toFixed(1)}%)`);
            console.log(`  prob_hold: ${withProbHold}/${total} (${(100*withProbHold/total).toFixed(1)}%)`);
            console.log(`  entry_atr: ${withATR}/${total} (${(100*withATR/total).toFixed(1)}%)`);
            console.log(`  rr: ${withRR}/${total} (${(100*withRR/total).toFixed(1)}%)`);
            console.log(`  directional_margin: ${withDM}/${total} (${(100*withDM/total).toFixed(1)}%)`);

            // Warn if critical fields are missing
            if (withProbBuy === 0) console.warn('[TRADE QUALITY] WARNING: No trades have prob_buy data');
            if (withProbSell === 0) console.warn('[TRADE QUALITY] WARNING: No trades have prob_sell data');
            if (withATR === 0) console.warn('[TRADE QUALITY] WARNING: No trades have entry_atr data');

            populateDateDropdowns(fullTrades);
            applyAllFilters();
        })
        .catch(error => {
            console.error('Error loading trades:', error);
            alert('Failed to load trade data. Please refresh the page.');
        });
}

// ============================================================================
// Date Dropdown Population
// ============================================================================
function populateDateDropdowns(trades) {
    const years = [...new Set(trades.map(t => new Date(t.timestamp).getFullYear()))];
    const yearFilter = document.getElementById('yearFilter');
    years.sort().forEach(y => {
        const opt = document.createElement('option');
        opt.value = y;
        opt.textContent = y;
        yearFilter.appendChild(opt);
    });

    const monthFilter = document.getElementById('monthFilter');
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December'];
    for (let m = 1; m <= 12; m++) {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = monthNames[m - 1];
        monthFilter.appendChild(opt);
    }
}

// ============================================================================
// Date Filtering Logic
// ============================================================================
function filterByDate(trades) {
    const month = parseInt(document.getElementById('monthFilter').value) || null;
    const year = parseInt(document.getElementById('yearFilter').value) || null;
    const isolate = document.getElementById('isolateModeCheckbox').checked;

    const startMonth = month || DEFAULT_START_MONTH;
    const startYear = year || DEFAULT_START_YEAR;
    const startDate = new Date(startYear, startMonth - 1, 1);

    return trades.filter(t => {
        const d = new Date(t.timestamp);
        const m = d.getMonth() + 1;
        const y = d.getFullYear();

        // Exclude January 2026 unless explicitly selected
        if (y === 2026 && m === 1 && !(month === 1 && year === 2026)) {
            return false;
        }

        if (isolate) {
            // Isolate mode: show ONLY selected month/year
            if (month && year) {
                return m === month && y === year;
            }
            if (year && !month) {
                if (y !== year) return false;
                if (y === 2026 && m === 1) return false;
                return true;
            }
            if (month && !year) {
                return m === month;
            }
            return !(y === 2026 && m === 1);
        }

        // Rolling-forward mode: show from start date forward
        return d >= startDate;
    });
}

// ============================================================================
// Trade Quality Filters
// ============================================================================
function filterByQuality(trades) {
    const minConf = parseFloat(document.getElementById('minConfidence').value) || 0;
    const minProbBuy = parseFloat(document.getElementById('minProbBuy').value) || 0;
    const minProbSell = parseFloat(document.getElementById('minProbSell').value) || 0;
    const minDM = parseFloat(document.getElementById('minDirectionalMargin').value) || 0;
    const minRR = parseFloat(document.getElementById('minRR').value) || 0;
    const minATR = parseFloat(document.getElementById('minATR').value) || 0;

    const whitelist = document.getElementById('symbolWhitelist').value.trim();
    const blacklist = document.getElementById('symbolBlacklist').value.trim();

    const whitelistSet = new Set(whitelist ? whitelist.split(',').map(s => s.trim().toUpperCase()) : []);
    const blacklistSet = new Set(blacklist ? blacklist.split(',').map(s => s.trim().toUpperCase()) : []);

    console.log('[TRADE QUALITY] Applying quality filters:');
    console.log(`  minConfidence: ${minConf}`);
    console.log(`  minProbBuy: ${minProbBuy}`);
    console.log(`  minProbSell: ${minProbSell}`);
    console.log(`  minDirectionalMargin: ${minDM}`);
    console.log(`  minRR: ${minRR}`);
    console.log(`  minATR: ${minATR}`);
    console.log(`  Whitelist: ${whitelist || 'none'}`);
    console.log(`  Blacklist: ${blacklist || 'none'}`);

    const beforeCount = trades.length;
    let filteredByWhitelist = 0;
    let filteredByBlacklist = 0;
    let filteredByConf = 0;
    let filteredByProbBuy = 0;
    let filteredByProbSell = 0;
    let filteredByDM = 0;
    let filteredByRR = 0;
    let filteredByATR = 0;

    const filtered = trades.filter(t => {
        // NULL-safe symbol filtering
        if (whitelistSet.size > 0 && !whitelistSet.has(t.symbol)) {
            filteredByWhitelist++;
            return false;
        }
        if (blacklistSet.has(t.symbol)) {
            filteredByBlacklist++;
            return false;
        }

        // NULL-safe quality metric filtering
        // Only apply filter if field is non-null AND below threshold
        if (t.confidence !== null && t.confidence !== undefined && t.confidence < minConf) {
            filteredByConf++;
            return false;
        }
        if (t.prob_buy !== null && t.prob_buy !== undefined && t.prob_buy < minProbBuy) {
            filteredByProbBuy++;
            return false;
        }
        if (t.prob_sell !== null && t.prob_sell !== undefined && t.prob_sell < minProbSell) {
            filteredByProbSell++;
            return false;
        }
        if (t.directional_margin !== null && t.directional_margin !== undefined && t.directional_margin < minDM) {
            filteredByDM++;
            return false;
        }
        if (t.rr !== null && t.rr !== undefined && t.rr < minRR) {
            filteredByRR++;
            return false;
        }
        if (t.entry_atr !== null && t.entry_atr !== undefined && t.entry_atr < minATR) {
            filteredByATR++;
            return false;
        }

        return true;
    });

    const afterCount = filtered.length;
    console.log(`[TRADE QUALITY] Filter results: ${beforeCount} → ${afterCount} trades (removed ${beforeCount - afterCount})`);
    if (filteredByWhitelist > 0) console.log(`  Whitelist: removed ${filteredByWhitelist} trades`);
    if (filteredByBlacklist > 0) console.log(`  Blacklist: removed ${filteredByBlacklist} trades`);
    if (filteredByConf > 0) console.log(`  Confidence: removed ${filteredByConf} trades`);
    if (filteredByProbBuy > 0) console.log(`  ProbBuy: removed ${filteredByProbBuy} trades`);
    if (filteredByProbSell > 0) console.log(`  ProbSell: removed ${filteredByProbSell} trades`);
    if (filteredByDM > 0) console.log(`  Directional Margin: removed ${filteredByDM} trades`);
    if (filteredByRR > 0) console.log(`  Risk/Reward: removed ${filteredByRR} trades`);
    if (filteredByATR > 0) console.log(`  ATR: removed ${filteredByATR} trades`);

    return filtered;
}

// ============================================================================
// Equity Recomputation
// ============================================================================
function recomputeEquity(trades) {
    let equity = 8000;
    return trades.map(t => {
        equity += t.dollar_pnl;
        return {
            timestamp: t.timestamp,
            equity: equity,
            dollar_pnl: t.dollar_pnl
        };
    });
}

// ============================================================================
// Statistics Calculation
// ============================================================================
function calculateStats(equityCurve, originalTradeCount) {
    if (equityCurve.length === 0) {
        return {
            totalTrades: 0,
            filteredTrades: 0,
            startingEquity: 8000,
            finalEquity: 8000,
            totalPnL: 0,
            maxEquity: 8000,
            maxDrawdown: 0,
            winRate: 0,
            avgPnL: 0
        };
    }

    const startingEquity = 8000;
    const finalEquity = equityCurve[equityCurve.length - 1].equity;
    const totalPnL = finalEquity - startingEquity;
    const maxEquity = Math.max(...equityCurve.map(e => e.equity));

    // Calculate max drawdown
    let maxDrawdown = 0;
    let peak = startingEquity;
    equityCurve.forEach(point => {
        if (point.equity > peak) peak = point.equity;
        const drawdown = peak - point.equity;
        if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    });

    // Calculate win rate
    const wins = equityCurve.filter(e => e.dollar_pnl > 0).length;
    const losses = equityCurve.filter(e => e.dollar_pnl < 0).length;
    const winRate = equityCurve.length > 0 ? (wins / equityCurve.length * 100) : 0;

    // Calculate average P&L
    const avgPnL = equityCurve.length > 0
        ? equityCurve.reduce((sum, e) => sum + e.dollar_pnl, 0) / equityCurve.length
        : 0;

    return {
        totalTrades: originalTradeCount,
        filteredTrades: equityCurve.length,
        startingEquity,
        finalEquity,
        totalPnL,
        maxEquity,
        maxDrawdown,
        winRate,
        avgPnL,
        wins,
        losses
    };
}

// ============================================================================
// Stats Rendering
// ============================================================================
function renderStats(stats) {
    const grid = document.getElementById('statsGrid');

    const cards = [
        { label: 'Trades Matched', value: `${stats.filteredTrades} / ${stats.totalTrades}`, class: '' },
        { label: 'Final Equity', value: `$${stats.finalEquity.toFixed(2)}`, class: stats.totalPnL > 0 ? 'positive' : 'negative' },
        { label: 'Total P&L', value: `$${stats.totalPnL.toFixed(2)}`, class: stats.totalPnL > 0 ? 'positive' : 'negative' },
        { label: 'Max Equity', value: `$${stats.maxEquity.toFixed(2)}`, class: 'positive' },
        { label: 'Max Drawdown', value: `$${stats.maxDrawdown.toFixed(2)}`, class: 'negative' },
        { label: 'Win Rate', value: `${stats.winRate.toFixed(1)}%`, class: '', sub: `${stats.wins}W / ${stats.losses}L` },
        { label: 'Avg P&L', value: `$${stats.avgPnL.toFixed(2)}`, class: stats.avgPnL > 0 ? 'positive' : 'negative' }
    ];

    grid.innerHTML = cards.map(card => `
        <div class="stat-card">
            <h3>${card.label}</h3>
            <div class="value ${card.class}">${card.value}</div>
            ${card.sub ? `<div style="font-size: 12px; color: #6b7280; margin-top: 5px;">${card.sub}</div>` : ''}
        </div>
    `).join('');
}

// ============================================================================
// Chart Rendering
// ============================================================================
function renderEquityChart(data) {
    const ctx = document.getElementById('equityChart').getContext('2d');

    if (window.chartInstances.equityChart) {
        window.chartInstances.equityChart.destroy();
    }

    const labels = data.map(point => {
        const date = new Date(point.timestamp);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    });

    window.chartInstances.equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Account Equity ($)',
                data: data.map(p => p.equity),
                borderColor: '#4A90E2',
                backgroundColor: 'rgba(74, 144, 226, 0.2)',
                fill: true,
                tension: 0.3,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// ============================================================================
// Symbol-Level Summary
// ============================================================================
function summarizeBySymbol(trades) {
    const map = {};

    trades.forEach(t => {
        if (!map[t.symbol]) {
            map[t.symbol] = {
                symbol: t.symbol,
                trades: 0,
                wins: 0,
                pnl: 0,
                sum_confidence: 0,
                sum_probability: 0,
                sum_rr: 0
            };
        }

        const s = map[t.symbol];
        s.trades++;
        if (t.dollar_pnl > 0) s.wins++;
        s.pnl += t.dollar_pnl;
        s.sum_confidence += (t.confidence || 0);
        s.sum_probability += (t.prob_buy !== undefined && t.prob_sell !== undefined)
            ? Math.max(t.prob_buy, t.prob_sell)
            : 0;
        s.sum_rr += (t.rr || 0);
    });

    return Object.values(map).map(s => ({
        symbol: s.symbol,
        trades: s.trades,
        win_rate: s.wins / s.trades,
        pnl: s.pnl,
        avg_confidence: s.sum_confidence / s.trades,
        avg_probability: s.sum_probability / s.trades,
        avg_rr: s.sum_rr / s.trades
    })).sort((a, b) => b.pnl - a.pnl); // Sort by P&L descending
}

function renderSymbolTable(summary) {
    const tbody = document.querySelector('#symbolTable tbody');
    tbody.innerHTML = '';

    if (summary.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #6b7280;">No symbols match the current filters</td></tr>';
        return;
    }

    summary.forEach(row => {
        const tr = document.createElement('tr');
        const pnlClass = row.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        tr.innerHTML = `
            <td><strong>${row.symbol}</strong></td>
            <td>${row.trades}</td>
            <td>${(row.win_rate * 100).toFixed(1)}%</td>
            <td class="${pnlClass}">$${row.pnl.toFixed(2)}</td>
            <td>${row.avg_confidence.toFixed(2)}</td>
            <td>${row.avg_probability.toFixed(2)}</td>
            <td>${row.avg_rr.toFixed(2)}</td>
        `;

        tbody.appendChild(tr);
    });

    makeTableSortable('symbolTable');
}

// ============================================================================
// Master Filter Pipeline
// ============================================================================
function applyAllFilters() {
    console.log('[TRADE QUALITY] ========== Starting Full Pipeline ==========');
    console.log(`[TRADE QUALITY] Step 1: Load fullTrades - ${fullTrades.length} total trades`);

    // Apply date filters
    let trades = filterByDate(fullTrades);
    console.log(`[TRADE QUALITY] Step 2: Apply date filters - ${trades.length} trades remain`);

    // Apply quality filters
    trades = filterByQuality(trades);
    console.log(`[TRADE QUALITY] Step 3: Apply quality filters - ${trades.length} trades remain`);

    // Recompute equity from $8,000
    const equity = recomputeEquity(trades);
    console.log(`[TRADE QUALITY] Step 4: Recompute equity from $8,000 - ${equity.length} equity points`);

    // Calculate statistics
    const stats = calculateStats(equity, fullTrades.length);
    console.log(`[TRADE QUALITY] Step 5: Calculate statistics - Final equity: $${stats.finalEquity.toFixed(2)}`);

    // Summarize by symbol
    const symbolSummary = summarizeBySymbol(trades);
    console.log(`[TRADE QUALITY] Step 6: Summarize by symbol - ${symbolSummary.length} unique symbols`);

    // Render results
    console.log(`[TRADE QUALITY] Step 7: Render UI components`);
    renderStats(stats);
    renderEquityChart(equity);
    renderSymbolTable(symbolSummary);

    console.log('[TRADE QUALITY] ========== Pipeline Complete ==========');
}

// ============================================================================
// Table Sorting
// ============================================================================
function makeTableSortable(tableId) {
    const table = document.getElementById(tableId);
    const headers = table.querySelectorAll('th');

    headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            const ascending = header.classList.toggle('ascending');

            rows.sort((a, b) => {
                const aText = a.children[index].innerText;
                const bText = b.children[index].innerText;

                const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return ascending ? aNum - bNum : bNum - aNum;
                }

                return ascending
                    ? aText.localeCompare(bText)
                    : bText.localeCompare(aText);
            });

            rows.forEach(row => tbody.appendChild(row));
        });
    });
}

// ============================================================================
// Event Listeners
// ============================================================================
document.getElementById('applyFiltersBtn').addEventListener('click', applyAllFilters);

document.getElementById('resetFiltersBtn').addEventListener('click', () => {
    document.querySelectorAll('input, select').forEach(el => {
        if (el.type === 'checkbox') {
            el.checked = false;
        } else {
            el.value = '';
        }
    });
    applyAllFilters();
});

// Auto-apply on Enter key in input fields
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            applyAllFilters();
        }
    });
});

// ============================================================================
// Initialize on page load
// ============================================================================
loadTrades();
