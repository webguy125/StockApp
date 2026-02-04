# Session Notes - 2026-01-25

## Problem Statement
chart.png shows all stocks with 130.0% scores in the adaptive ranking system.

## Root Cause Analysis

### Initial Hypothesis
Empty signal_history table causing win rates to default to 100%, leading to:
- Score = (100% × 0.5) + (100% × 0.3) + (100% × 0.2) + freq_bonus + persistence = 130%

### Investigation Steps

1. **Database Inspection**
   - signal_history: 0 rows initially
   - trades (backtest): 553,952 rows of training data
   - active_signals: 154 open positions (no closed trades)

2. **Bootstrap Attempt**
   - Created bootstrap_signal_history.py
   - Populated signal_history with 236,994 BUY/SELL signals from backtest data
   - Result: Still 130% scores

3. **Win Rate Calculation Bug**
   - Found case sensitivity issue: signal_type='BUY' but checking `if outcome == 'buy'`
   - Fixed by normalizing to uppercase
   - Result: Still 130% scores

4. **Data Quality Analysis**
   ```
   BUY signals:  137,170 total, 137,170 positive P/L (100% win rate)
   SELL signals: 99,824 total,  99,824 negative P/L (100% win rate)
   ```

### THE ACTUAL PROBLEM

The backtest data in `trades` table is **synthetic training data**, NOT real closed trades:
- Created by scanning historical prices and labeling outcomes
- Outcome = 'buy' only if price went up by target (e.g., +5%)
- Outcome = 'sell' only if price went down by target (e.g., -5%)
- By definition, 100% of these samples are "wins"
- This data is for **training ML models**, not for calculating real-world win rates

### Real Trade Status

**Active Signals**: 154 open positions (2026-01-11 to 2026-01-25)
- All still open (no exit_price, no profit_loss_pct)
- Cannot calculate win rates from open positions

**Closed Signals**: 0 real closed trades exist
- No historical scanner output has been closed yet
- signal_history contains only training data, not real trades

## Conclusion

The 130% scores are **mathematically correct** given the data source:
- Bootstrap populated signal_history with training data
- Training data has 100% win rate by design
- Ranking engine correctly calculated: 100% WR → 130% composite score

The fundamental issue is **no real closed trades exist yet**.

## Solution Path

As specified in the user's original JSON request:

1. **Keep Bootstrap** (for testing/development)
   - Useful for validating calculation logic
   - Shows what scores WOULD look like with perfect trades

2. **Implement Real-Time Closing Logic**
   - File: `backend/turbomode/core_engine/overnight_scanner.py`
   - When scanner runs, check for existing signals in active_signals
   - If holding period complete (14 days) or stop/target hit:
     - Calculate exit_price and profit_loss_pct
     - Insert into signal_history
     - Remove from active_signals

3. **Wait for Real Data**
   - Let scanner run daily
   - Accumulate real closed trades over weeks/months
   - Win rates will reflect actual model performance

4. **Add Diagnostics**
   - Create tools/diagnose_signals.py
   - Separate display for:
     - Training data win rates (synthetic, 100%)
     - Real signal win rates (actual performance)

## Files Modified

1. `backend/turbomode/tools/bootstrap_signal_history.py` (created)
   - Populates signal_history from backtest data
   - Safety check to prevent overwriting

2. `backend/turbomode/adaptive_stock_ranker.py` (modified)
   - Line 194-205: Query signal_history instead of trades
   - Line 86: Normalize outcome to uppercase
   - Line 94-106: Case sensitivity fix for BUY/SELL

## Implementation Complete

CONTAMINATION-PROOF SIGNAL LIFECYCLE NOW ACTIVE

Files Created:
1. backend/turbomode/tools/migrate_training_data.py
   - Moves 553,952 synthetic training samples to training_samples table
   - Clears signal_history for real closed trades only

2. backend/turbomode/core_engine/signal_closer.py
   - Real-time exit logic (14-day holding period, target/stop hits)
   - Computes realized P/L and moves closed trades to signal_history
   - Integrated into overnight_scanner.py (runs before each scan)

Files Modified:
1. backend/turbomode/core_engine/overnight_scanner.py
   - Added signal_closer integration (STEP -1)
   - Closes ready signals before generating new ones

2. backend/turbomode/adaptive_stock_ranker.py
   - Updated to use ONLY signal_history (real closed trades)
   - Clear messaging when no real data exists yet
   - No fallback to synthetic data

Database State After Migration:
- signal_history: 0 rows (clean, awaiting real closed trades)
- training_samples: 553,952 rows (synthetic backtest data for ML)
- active_signals: 154 rows (open positions, 2 ready to close)

Contamination Status: CLEAN
- Complete separation between training data and performance tracking
- No synthetic data can influence performance metrics
- Real-time closing logic will populate signal_history going forward

## Next Steps

The system is now production-ready with contamination-proof lifecycle.

Real performance data will accumulate as:
1. Scanner runs daily and generates signals
2. Signals reach exit conditions (14 days or target/stop)
3. Closed trades populate signal_history
4. Win rates and rankings compute from real outcomes only

Expected timeline for meaningful statistics:
- 30 days: Initial win rate data
- 60 days: Reliable rolling metrics
- 90 days: Full 30/60/90 day win rate windows

## Technical Notes

### Database Tables
- `trades` (trade_type='backtest'): 553,952 training samples (2016-2026)
- `signal_history`: Closed trades for win-rate calculation (currently has synthetic data)
- `active_signals`: 154 open positions (real scanner output)

### Win Rate Formula (Current)
```python
# BUY signal is a win if profit_loss_pct > 0
# SELL signal is a win if profit_loss_pct < 0
```

### Composite Score Formula
```python
score = (wr_30d × 0.5) + (wr_60d × 0.3) + (wr_90d × 0.2) +
        (freq_score × 0.1) + persistence_bonus
```

With 100% win rates:
```python
score = (1.0 × 0.5) + (1.0 × 0.3) + (1.0 × 0.2) + 0.1 + 0.0 = 1.1 to 1.3
```

Displayed as 130.0%
