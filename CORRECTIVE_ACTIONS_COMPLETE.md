# TurboMode Corrective Actions - COMPLETE
**Date:** 2026-01-06
**Status:** ✅ All critical mismatches fixed
**Action:** DO NOT TRAIN until explicit approval

---

## A. CORRECTED CODE BLOCKS

### 1. training_sample_generator.py (Lines 117-167)

**WHAT WAS WRONG:**
```python
# OLD - INVERTED LOGIC
if outcome['signal_type'] == 'BUY':
    label = 'buy' if outcome['is_correct'] else 'sell'  # ❌ WRONG!
else:  # SELL
    label = 'sell' if outcome['is_correct'] else 'buy'  # ❌ WRONG!
```

**WHAT IS NOW CORRECT:**
```python
# NEW - ACTUAL PRICE MOVEMENT
return_pct = outcome['return_pct']

if return_pct >= 0.05:  # Price went UP ≥5%
    label = 'buy'
elif return_pct <= -0.05:  # Price went DOWN ≤-5%
    label = 'sell'
else:  # Price moved between -5% and +5% (flat/sideways)
    label = 'hold'
```

**WHY THIS FIXES IT:**
- Labels now based on ACTUAL outcome (did price go up or down?)
- NOT based on whether prediction was "correct"
- If BUY signal resulted in +8% gain (missed 10% target), label is still 'buy' (price went up!)
- If SELL signal resulted in -8% drop (missed -10% target), label is still 'sell' (price went down!)
- Model learns: "These features → price goes UP" (not "prediction was correct/incorrect")

---

### 2. outcome_tracker.py (Lines 151-181)

**WHAT WAS WRONG:**
```python
# OLD - ASYMMETRIC LOGIC
if signal_type == 'BUY':
    is_correct = return_pct >= self.win_threshold  # ✓ Correct
elif signal_type == 'SELL':
    is_correct = return_pct < self.win_threshold   # ❌ WRONG! Marks +5% as "correct"
```

**WHAT IS NOW CORRECT:**
```python
# NEW - SYMMETRIC LOGIC
if signal_type == 'BUY':
    # BUY signal: Correct ONLY if gained ≥10% (hit target)
    is_correct = return_pct >= self.win_threshold  # +0.10

elif signal_type == 'SELL':
    # SELL signal: Correct ONLY if lost ≥10% (hit bearish target)
    # IMPORTANT: SELL is a SHORT position - profit when price goes DOWN
    is_correct = return_pct <= -self.win_threshold  # -0.10
```

**WHY THIS FIXES IT:**
- BUY correct if: return ≥ +10%
- SELL correct if: return ≤ -10%
- **SYMMETRIC** - both require hitting 10% target
- SELL at +5% is now correctly marked "incorrect" (hit stop loss, not target)
- SELL is explicitly a SHORT/bearish position (price must go DOWN to win)

---

## B. THE HOLD LOGIC (EXACT IMPLEMENTATION)

**Location:** `training_sample_generator.py:142-147`

```python
if return_pct >= 0.05:  # Price went UP ≥5%
    label = 'buy'
elif return_pct <= -0.05:  # Price went DOWN ≤-5%
    label = 'sell'
else:  # Price moved between -5% and +5% (flat/sideways)
    label = 'hold'
```

**HOLD Definition:**
- HOLD = price movement strictly between -5% and +5%
- If return = +4.9%: label = 'hold'
- If return = +5.0%: label = 'buy'
- If return = -4.9%: label = 'hold'
- If return = -5.0%: label = 'sell'
- If return = 0.0%: label = 'hold'

**Purpose of HOLD:**
- Captures flat/sideways/choppy market conditions
- Model learns: "These features → price goes nowhere"
- Does NOT absorb mislabeled BUY/SELL cases
- Clean separation: buy (up), hold (flat), sell (down)

---

## C. RECOMMENDED DISPLAY WORDING FOR SIGNALS

### For User-Facing UI (Frontend/Dashboard):

**BUY Signal:**
```
🟢 BUY (Go Long)
Entry: $100.00
Target: $110.00 (+10%)
Stop: $95.00 (-5%)
Confidence: 72%

Action: Open long position or buy call options
```

**SELL Signal:**
```
🔴 SELL (Go Short)
Entry: $100.00
Target: $90.00 (-10%)
Stop: $105.00 (+5%)
Confidence: 68%

Action: Open short position or buy put options
```

**HOLD Signal (if displayed):**
```
⚪ HOLD (No Action)
Current: $100.00
Confidence: 54%

Action: No trade recommended - wait for clearer setup
```

### For Database/Logs (Internal):

- `signal_type = 'BUY'` → Bullish / Long position
- `signal_type = 'SELL'` → Bearish / Short position
- Model prediction: `'buy'`, `'sell'`, `'hold'` (lowercase, training labels)
- Scanner output: `'BUY'`, `'SELL'` (uppercase, actionable signals)

### For Comments in Code:

```python
# BUY = bullish signal, expect price to rise +10%
# SELL = bearish signal, expect price to fall -10%
# HOLD = neutral, no strong directional bias
```

---

## D. CONFIRMATION - SAMPLES CAN NOW BE REGENERATED SAFELY

✅ **ALL FIXES COMPLETE**

The system is now ready to regenerate training samples from `signal_history` table with corrected logic:

### What Will Happen During Regeneration:

1. **Read outcomes from signal_history:**
   - Each outcome has: `symbol`, `signal_type` (BUY/SELL), `return_pct`, `is_correct`

2. **Apply NEW label logic (training_sample_generator.py):**
   - Ignore `is_correct` field (was using inverted logic)
   - Use ONLY `return_pct` to determine label:
     - `return_pct >= +5%` → label = 'buy'
     - `return_pct <= -5%` → label = 'sell'
     - `-5% < return_pct < +5%` → label = 'hold'

3. **Save to trades table:**
   - Clean training samples with correct labels
   - Model will learn from ACTUAL price movements

### Example Scenarios (Before vs After):

| Signal Type | Return | Old Label | New Label | Explanation |
|-------------|--------|-----------|-----------|-------------|
| BUY | +8% | 'sell' ❌ | 'buy' ✅ | Price went UP, correct label is 'buy' |
| BUY | -3% | 'sell' ❌ | 'hold' ✅ | Small loss, flat movement |
| SELL | -12% | 'sell' ✅ | 'sell' ✅ | Price went DOWN, correct |
| SELL | +5% | 'sell' ❌ | 'buy' ✅ | Price went UP (stop hit), label is 'buy' |
| BUY | +15% | 'buy' ✅ | 'buy' ✅ | Price went UP, correct |
| SELL | -8% | 'sell' ✅ | 'sell' ✅ | Price went DOWN, correct |

---

## E. WHAT TO DO NEXT (AWAITING APPROVAL)

### ⚠️ STOP HERE - DO NOT PROCEED WITHOUT EXPLICIT APPROVAL

**Next steps (ONLY after you approve):**

1. **Regenerate ALL training samples:**
   ```python
   # Run training_sample_generator.py to regenerate samples
   python backend/turbomode/training_sample_generator.py
   ```

2. **Verify label distribution:**
   ```sql
   SELECT outcome, COUNT(*) as count
   FROM trades
   GROUP BY outcome;
   ```
   - Should see reasonable distribution of buy/sell/hold
   - No more inverted labels

3. **Only THEN proceed to retraining:**
   - After you confirm labels look correct
   - Use corrected training data
   - Models will now learn correct patterns

---

## F. FILES MODIFIED

1. `backend/turbomode/training_sample_generator.py`
   - Lines 117-167: Replaced inverted label logic with actual price movement logic

2. `backend/turbomode/outcome_tracker.py`
   - Lines 151-181: Fixed SELL outcome evaluation to be symmetric with BUY

**NO OTHER FILES MODIFIED** - Scope limited per instructions

---

## G. SUMMARY OF WHAT WAS FIXED

### MISMATCH #1: SELL Signal Semantic Inversion
- **Status:** ✅ FIXED
- **Change:** outcome_tracker.py now marks SELL correct only if return ≤ -10%
- **Impact:** SELL signals now properly evaluated as bearish bets (short positions)

### MISMATCH #2: Training Label Logic Inversion (CRITICAL)
- **Status:** ✅ FIXED
- **Change:** training_sample_generator.py now bases labels on actual price movement
- **Impact:** Model will learn "features → price direction" not "features → was I correct?"

### MISMATCH #3: Threshold Asymmetry
- **Status:** ✅ FIXED
- **Change:** Both BUY and SELL now require 10% target to be "correct"
- **Impact:** Symmetric evaluation, no bias toward SELL predictions

---

## H. WHAT MODELS WILL NOW LEARN

### BEFORE (Corrupted Labels):
- "When BUY prediction gains +8% (missed target), label = 'sell'" ❌
- "When SELL prediction hits +5% stop, label = 'sell'" ❌
- Model confused about what 'sell' means

### AFTER (Corrected Labels):
- "When price gains +8%, label = 'buy' (price went UP)" ✅
- "When price hits +5%, label = 'buy' (price went UP)" ✅
- "When price drops -8%, label = 'sell' (price went DOWN)" ✅
- Model learns clear directional patterns

---

## FINAL CONFIRMATION

✅ **All critical mismatches fixed**
✅ **HOLD logic confirmed: -5% to +5%**
✅ **Display wording recommended**
✅ **Samples ready for regeneration**

🛑 **WAITING FOR APPROVAL TO REGENERATE SAMPLES**

Do NOT run training until:
1. You approve the logic above
2. Samples are regenerated
3. You verify label distribution looks correct

---

**End of Corrective Actions Report**
