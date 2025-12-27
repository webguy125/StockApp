# TurboMode - Time Decay & Smart Signal Replacement

## Overview

TurboMode now includes intelligent time-decay logic that automatically replaces aged signals with fresh, high-confidence ones. This keeps your signal list fresh and relevant.

## Time Decay Formula

**Effective Confidence = Original Confidence × (1 - (age_days / 14) × 0.3)**

### Examples:
- **Day 0** (Fresh): 95% confidence → 95.0% effective (100% of original)
- **Day 7** (Mid-age): 95% confidence → 80.8% effective (85% of original)
- **Day 14** (Max age): 95% confidence → 66.5% effective (70% of original)

### Why Time Decay?

1. **Market conditions change** - A 10-day-old signal may no longer be valid
2. **Fresh signals are better** - Recent ML predictions reflect current market
3. **Automatic rotation** - Keeps your top 20 list fresh without manual intervention

## Smart Replacement Logic

### Per Market Cap + Signal Type

The system maintains **20 signals maximum** for each combination:
- Large Cap BUY (max 20)
- Large Cap SELL (max 20)
- Mid Cap BUY (max 20)
- Mid Cap SELL (max 20)
- Small Cap BUY (max 20)
- Small Cap SELL (max 20)

**Total capacity: 120 active signals** (20 × 6 categories)

### Replacement Process

When the nightly scan runs at 11 PM:

1. **Scanner generates new signals** (sorted by confidence)
2. **For each new signal:**
   - If category has < 20 signals → Add it
   - If category has 20 signals → Check replacement logic

3. **Replacement Check:**
   - Find the **weakest signal** in that category (lowest effective confidence with time decay)
   - Compare new signal's confidence to weakest's effective confidence
   - If new > weakest → Replace old with new
   - If new ≤ weakest → Skip new signal

4. **Old signal moved to history:**
   - Marked as "REPLACED" in signal_history table
   - No P&L calculated (not actually traded)
   - Preserved for record-keeping

### Example Replacement Scenario

**Current Large Cap BUY signals (at limit: 20)**
- AAPL: 95% confidence, Day 0 → Effective: 95.0%
- MSFT: 92% confidence, Day 3 → Effective: 91.1%
- ...
- XOM: 88% confidence, Day 10 → Effective: 73.9% ← WEAKEST

**New scan finds:**
- JPM: 90% confidence, Day 0 → Effective: 90.0%

**Result:**
✅ JPM (90.0%) **REPLACES** XOM (73.9% effective)
- XOM moved to history with "REPLACED" status
- JPM added as fresh signal

## Viewing Effective Confidence

### API Response
All signals now include both:
- `confidence`: Original ML prediction confidence
- `effective_confidence`: Time-decayed confidence for ranking

### Frontend Display
Signals are sorted by effective confidence (not original confidence), ensuring the freshest, highest-confidence signals appear first.

## Benefits

1. **Always Fresh**: Signals naturally rotate as they age
2. **Quality Control**: Low-confidence aged signals automatically replaced
3. **Transparency**: Old signals preserved in history (not deleted)
4. **Hands-Off**: Runs automatically at 11 PM nightly
5. **Smart Ranking**: Combines confidence + freshness in one metric

## Manual Override

If you don't want a signal replaced:
- Currently no manual "pin" feature
- Signal will be replaced if it becomes the weakest (effective confidence)
- Future enhancement: Add "pinned" flag to prevent replacement

## Configuration

Time decay parameters (in `database_schema.py`):
```python
decay_factor = 1.0 - (age_days / 14.0) * 0.3
# 30% maximum penalty at 14 days
# Minimum effective confidence: 70% of original
```

To adjust decay rate:
- Increase 0.3 → More aggressive aging
- Decrease 0.3 → Less aggressive aging

## Monitoring Replacements

Check Flask logs during nightly scan for replacement activity:
```
[REPLACE] JPM (90.0%) replaces XOM (eff: 73.9%, age: 10d)
```

View history table for all replaced signals:
```sql
SELECT * FROM signal_history WHERE exit_reason = 'REPLACED';
```

---

**Status:** ✅ Implemented and Active
**Auto-runs:** 11 PM nightly with overnight scan
