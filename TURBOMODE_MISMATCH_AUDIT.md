# TurboMode System - Mismatch Audit Report
**Date:** 2026-01-06
**Auditor:** Claude Code (MismatchAuditor role)
**Scope:** Full pipeline audit (training, scanner, backtest, outcome_tracker, database)

---

## Executive Summary

**CRITICAL MISMATCHES FOUND:** 3 severe, 2 moderate
**STATUS:** System has fundamental semantic mismatches that could cause training/prediction inversion

---

## 🚨 MISMATCH #1: SELL Signal Semantic Inversion (CRITICAL)

### Mismatch Type
**Semantic mismatch** - The word "SELL" means different things in different modules

### Affected Modules
- `overnight_scanner.py` (lines 476-499)
- `outcome_tracker.py` (lines 160-163)
- `training_sample_generator.py` (lines 128-136)
- `database_schema.py` (comments lines 62-63)

### Root Cause
**Two conflicting interpretations of "SELL" signal:**

**Interpretation A (Scanner/Database - CORRECT per user intent):**
```python
# overnight_scanner.py:498-499
else:  # SELL
    target_price = current_price * 0.90  # -10% (bearish, price going DOWN)
    stop_price = current_price * 1.05    # +5% stop (price going UP)
```
**Meaning:** SELL = open a short position / buy puts (bearish directional bet)

**Interpretation B (Outcome Tracker - INVERTED LOGIC):**
```python
# outcome_tracker.py:160-163
elif signal_type == 'SELL':
    # SELL signal: Correct if lost money or didn't hit +10%
    is_correct = return_pct < self.win_threshold  # win_threshold = +10%
```
**Meaning:** SELL is "correct" if price DIDN'T go up (i.e., if it went down OR stayed flat)

### Why This is Dangerous

1. **Scanner generates SELL = bearish bet (price will go DOWN -10%)**
2. **Outcome tracker marks SELL correct if `return_pct < 0.10`** (meaning price went down OR went up <10%)
3. **Training sample generator creates label based on "is_correct":**
   ```python
   # training_sample_generator.py:136
   label = 'sell' if outcome['is_correct'] else 'buy'
   ```

**THE PROBLEM:**
- If price goes from $100 → $105 (+5%), a SELL signal is marked "correct" (because +5% < +10%)
- This creates a training label of "sell" for a +5% gain scenario
- **Model learns:** "sell" label = small gains are acceptable
- **Scanner interprets:** "sell" prediction = open short (expect -10% drop)
- **MISMATCH:** Model predicts "sell" thinking it means "modest gain", scanner opens short expecting drop

### Impact
**SEVERE** - This could cause the model to:
- Predict "sell" on stocks that will gain 5%
- Scanner opens short positions on stocks about to gain
- Catastrophic losses on bearish positions

### Recommended Fix

**Option 1: Align SELL with bearish intent (user's stated intent)**

Change outcome_tracker.py line 161-162:
```python
# SELL signal: Correct if price DROPPED (bearish bet won)
# Target = -10%, so correct if return_pct <= -0.10 OR hit stop at +5%
is_correct = return_pct <= -0.10  # Only correct if target hit
```

AND update target/stop logic to be symmetric:
```python
# database_schema.py comments should say:
# target_price: -10% for SELL (bearish target)
# stop_price: +5% for SELL (bearish stop)
```

**Option 2: Rename to avoid confusion**

Rename "SELL" → "SHORT" throughout system:
- Scanner outputs: 'BUY' (long) or 'SHORT' (short)
- Database stores: 'BUY' or 'SHORT'
- Model predicts: 'buy' or 'short'

This makes intent explicit.

---

## 🚨 MISMATCH #2: Training Label Logic Inversion (CRITICAL)

### Mismatch Type
**Labeling mismatch** - Training labels inverted for incorrect predictions

### Affected Modules
- `training_sample_generator.py` (lines 128-136)

### Root Cause
```python
# training_sample_generator.py:128-136
# BUY prediction that was correct (≥10% gain) → label = 'buy'  ✓
# BUY prediction that was incorrect (<10% gain) → label = 'sell'  ⚠️
# SELL prediction that was correct (didn't hit 10%) → label = 'sell'  ⚠️
# SELL prediction that was incorrect (did hit 10%) → label = 'buy'  ⚠️

if outcome['signal_type'] == 'BUY':
    label = 'buy' if outcome['is_correct'] else 'sell'
else:  # SELL
    label = 'sell' if outcome['is_correct'] else 'buy'
```

### Why This is Dangerous

**Example scenario:**
1. Scanner predicts BUY, price gains +8% (missed +10% target)
2. Outcome tracker marks this as "incorrect" (didn't hit target)
3. Training generator assigns label = **'sell'**
4. Model learns: "When features look like this, label is 'sell'"
5. **PROBLEM:** Those features led to a BUY signal that gained +8%, but model now associates them with 'sell'

**This creates a reinforcement learning death spiral:**
- Incorrect BUY → labeled 'sell' → model less likely to predict 'buy' next time
- Incorrect SELL → labeled 'buy' → model less likely to predict 'sell' next time
- System becomes increasingly conservative and confused

### Impact
**SEVERE** - Model learns inverted patterns:
- Features that lead to +8% gains get 'sell' labels
- Features that lead to -8% losses get 'buy' labels (if SELL signal was "incorrect")
- Model performance degrades over time through self-reinforcement

### Recommended Fix

**Change training label logic to reflect ACTUAL outcome, not prediction correctness:**

```python
# training_sample_generator.py - CORRECTED
def create_training_sample(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create training sample based on ACTUAL outcome, not prediction correctness

    New logic:
    - If price went UP significantly (≥5%) → label = 'buy'
    - If price went DOWN significantly (≤-5%) → label = 'sell'
    - If price stayed flat (-5% to +5%) → label = 'hold' (or exclude from training)
    """
    return_pct = outcome['return_pct']

    # Base label on ACTUAL price movement, not on whether prediction was "correct"
    if return_pct >= 0.05:  # Price went up ≥5%
        label = 'buy'
    elif return_pct <= -0.05:  # Price went down ≤-5%
        label = 'sell'
    else:  # Flat movement (-5% to +5%)
        label = 'hold'  # Or exclude from binary classification

    # Create training sample
    training_sample = {
        'symbol': outcome['symbol'],
        'outcome': label,  # Based on ACTUAL movement, not prediction
        'profit_loss_pct': return_pct,
        # ... rest of fields
    }

    return training_sample
```

**Rationale:**
- Model should learn: "These features → price went up" or "These features → price went down"
- NOT: "These features → my previous prediction was correct/incorrect"
- Training should be based on GROUND TRUTH (actual price movement), not prediction accuracy

---

## ⚠️ MISMATCH #3: Threshold Asymmetry (MODERATE)

### Mismatch Type
**Threshold mismatch** - BUY and SELL use different success criteria

### Affected Modules
- `overnight_scanner.py` (lines 494-499)
- `outcome_tracker.py` (lines 156-163)

### Root Cause

**BUY signal thresholds:**
- Target: +10%
- Stop: -5%
- Risk/Reward: 1:2 ratio (risk 5% to gain 10%)
- Correct if: `return_pct >= 0.10`

**SELL signal thresholds:**
- Target: -10%
- Stop: +5%
- Risk/Reward: 1:2 ratio (risk 5% to gain 10%)
- Correct if: `return_pct < 0.10` ⚠️ (NOT symmetric!)

### Why This is Dangerous

**SELL is marked "correct" for these scenarios:**
- Price -15%: ✓ Correct (hit target)
- Price -5%: ✓ Correct (made money)
- Price +0%: ✓ Correct (didn't lose)
- Price +5%: ✓ "Correct" (< +10%, hit stop but still marked correct!)
- Price +15%: ✗ Incorrect (blew past stop)

**BUY is only "correct" if:**
- Price ≥+10%

**Problem:** A SELL signal that hits its +5% stop loss is still marked "correct" because +5% < +10%. This is inconsistent with how BUY signals are evaluated.

### Impact
**MODERATE** - Creates biased training data:
- SELL signals have looser success criteria (anything < +10%)
- BUY signals have strict success criteria (must hit +10%)
- Model may learn to prefer SELL predictions (easier to be "correct")

### Recommended Fix

**Make thresholds symmetric:**

```python
# outcome_tracker.py - CORRECTED
if signal_type == 'BUY':
    # BUY: Correct if hit +10% target
    is_correct = return_pct >= 0.10

elif signal_type == 'SELL':
    # SELL: Correct if hit -10% target (symmetric with BUY)
    is_correct = return_pct <= -0.10  # Changed from < 0.10
```

OR if you want to include "didn't lose" logic for SELL:

```python
elif signal_type == 'SELL':
    # SELL: Correct if made money (bearish bet won) OR hit target
    is_correct = (return_pct <= 0.0) or (return_pct <= -0.10)
```

But this should be symmetric for BUY too:
```python
if signal_type == 'BUY':
    # BUY: Correct if made money (bullish bet won) OR hit target
    is_correct = (return_pct >= 0.0) or (return_pct >= 0.10)
```

---

## ⚠️ MISMATCH #4: Database Schema vs Comment Mismatch (MINOR)

### Mismatch Type
**Documentation mismatch** - Comments don't match implementation

### Affected Modules
- `database_schema.py` (lines 62-63)

### Root Cause
```python
# database_schema.py:62-63
target_price REAL NOT NULL,  -- +10% for BUY, -10% for SELL
stop_price REAL NOT NULL,    -- -5% for BUY, +5% for SELL
```

**Comments are CORRECT**, but there's no code enforcement. The actual target/stop calculation is in overnight_scanner.py.

### Impact
**MINOR** - Potential for future bugs if someone modifies overnight_scanner.py without checking database schema comments

### Recommended Fix

**Add validation in database_schema.py:**

```python
def add_signal(self, signal: Dict[str, Any]) -> bool:
    """Add new signal with validation"""

    # Validate target/stop make sense
    entry = signal['entry_price']
    target = signal['target_price']
    stop = signal['stop_price']

    if signal['signal_type'] == 'BUY':
        assert target > entry, "BUY target must be above entry"
        assert stop < entry, "BUY stop must be below entry"
    elif signal['signal_type'] == 'SELL':
        assert target < entry, "SELL target must be below entry"
        assert stop > entry, "SELL stop must be above entry"

    # ... rest of insertion logic
```

---

## ⚠️ MISMATCH #5: Backtest Generator Label Mismatch (MODERATE)

### Mismatch Type
**Pipeline mismatch** - Backtest generator doesn't align with training labels

### Affected Modules
- `backtest_generator.py` (lines 132-133)

### Root Cause

Backtest generator calculates:
```python
df['actual_return'] = df['close'].pct_change().shift(-1)
df['actual_direction'] = np.where(df['actual_return'] > 0, 1, 0)
```

This creates binary labels: 1 if price went up, 0 if price went down.

**Problem:** This doesn't match training labels which use:
- 'buy' / 'sell' strings (not 0/1)
- Threshold-based (≥10% or ≤-10%), not simple direction

### Impact
**MODERATE** - Backtest evaluation doesn't match how model was trained

### Recommended Fix

**Align backtest labels with training labels:**

```python
# backtest_generator.py - CORRECTED
# Calculate returns
df['actual_return'] = df['close'].pct_change().shift(-1)

# Use same thresholds as training
df['actual_label'] = 'hold'  # default
df.loc[df['actual_return'] >= 0.10, 'actual_label'] = 'buy'   # +10% threshold
df.loc[df['actual_return'] <= -0.10, 'actual_label'] = 'sell' # -10% threshold

# For evaluation, compare model prediction to actual_label (both strings)
```

---

## Summary of Recommended Actions

### Immediate (Fix before next training run):

1. **Fix MISMATCH #2 first** - Change training_sample_generator.py to use actual price movement, not prediction correctness
2. **Fix MISMATCH #1** - Align SELL signal interpretation (outcome_tracker.py line 161-162)
3. **Fix MISMATCH #3** - Make threshold logic symmetric (outcome_tracker.py)

### Secondary (Can defer):

4. **Fix MISMATCH #5** - Align backtest generator with training labels
5. **Fix MISMATCH #4** - Add validation to database schema

### Testing Plan:

After fixes:
1. Generate new training samples from signal_history
2. Verify labels make sense (BUY signals that gained 8% should NOT be labeled 'sell')
3. Retrain models with corrected labels
4. Run backtest with aligned evaluation logic
5. Compare before/after performance

---

## Conclusion

The TurboMode system has **fundamental semantic mismatches** that cause:
1. Training labels to be inverted (incorrect predictions get opposite labels)
2. SELL signals to have inconsistent interpretations (bearish vs. "not bullish")
3. Threshold asymmetry that biases model toward SELL predictions

**These mismatches must be corrected before the system can learn correct patterns.**

Without fixes, the model is learning from corrupted labels and will continue to degrade performance over time.

---

**End of Audit Report**
