# Rare Event Archive

## Purpose

This archive contains curated samples from 7 major market stress events (2008-2023) to ensure the ML trading system never forgets critical crash patterns, even as the rolling 5-year training window moves forward.

## Problem Solved

**Without Archive:**
- Rolling 5-year window forgets old crashes
- Model struggles when rare events reoccur
- Performance degrades in stress scenarios

**With Archive:**
- System remembers 2008-style crashes
- Better performance in bear markets
- More robust across all market regimes

## Events Included

| Event | Date Range | Duration | Regime | Description |
|-------|-----------|----------|--------|-------------|
| 2008 Financial Crisis | Sept 2008 - Mar 2009 | 7 months | crash | Lehman collapse, financial system freeze |
| 2011 Debt Ceiling | July-Aug 2011 | 2 months | high_volatility | US debt crisis, S&P downgrade |
| 2015 China Devaluation | Aug-Sep 2015 | 2 months | high_volatility | Yuan devaluation, commodity collapse |
| 2018 Volmageddon | Feb 2018 | 1 month | crash | VIX spike, vol ETF implosion |
| 2020 COVID Crash | Feb 20 - Mar 23, 2020 | 1 month | crash | Pandemic crash period |
| 2020 COVID Recovery | Mar 24 - Jun 30, 2020 | 3 months | recovery | V-shaped recovery period |
| 2022 Inflation Bear | Jan-Dec 2022 | 12 months | high_volatility | Rate hikes, inflation, full year bear |
| 2023 Banking Mini-Crisis | Mar-Apr 2023 | 2 months | high_volatility | SVB collapse, regional bank stress |

## Archive Structure

```
rare_event_archive/
├── archive.db                     # SQLite database with all samples
├── metadata/
│   ├── archive_config.json       # Configuration & event weights
│   ├── event_metadata.json       # Per-event statistics (generated)
│   └── generation_log.json       # Generation history (generated)
├── scripts/
│   └── generate_rare_event_archive.py  # Generation script
└── README.md                     # This file
```

## Usage

### Generate Archive (One-Time)

```bash
cd backend/data/rare_event_archive/scripts
python generate_rare_event_archive.py
```

**Runtime:** 2-4 hours (processes 8 events across multiple years)
**Output:** `archive.db` with ~18,000-24,000 samples

### Use in Training

Archive is automatically integrated when training:

```python
from advanced_ml.training.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline(
    symbols=all_symbols,
    years=5,
    use_rare_event_archive=True  # Default: enabled
)
```

**Archive Mix:** 7% of training samples come from archive
**Event Weights:** Emphasize recent major events (2008, 2020, 2022)

### Disable Archive

```python
# To train without archive
results = pipeline.run_full_pipeline(
    use_rare_event_archive=False
)
```

## Archive Composition

**Example for 85,000 total training samples:**

- Normal (5-year window): 79,050 samples (93%)
- Archive: 5,950 samples (7%)
  - 2008 Crisis: 1,488 samples (25% of archive)
  - 2020 COVID: 1,488 samples (25% of archive)
  - 2022 Bear: 1,190 samples (20% of archive)
  - 2018 Volma: 595 samples (10% of archive)
  - 2023 Banking: 595 samples (10% of archive)
  - 2015 China: 298 samples (5% of archive)
  - 2011 Debt: 298 samples (5% of archive)

## Expected Impact

**Crash Scenario Accuracy:**
- Before: ~70-75%
- After: ~80-85% (+10-15% improvement)

**Overall Accuracy:**
- Maintains or slightly improves (+1-2%)

**Risk-Adjusted Returns:**
- +15-25% improvement in stress periods

## Regeneration

Re-generate archive when:
- Feature engineering changes (new indicators added)
- Symbol list changes significantly
- Event definitions need adjustment

```bash
python generate_rare_event_archive.py --regenerate
```

## Database Schema

**Table:** `archive_samples`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| event_name | TEXT | Event identifier |
| symbol | TEXT | Stock symbol |
| date | TEXT | Entry date (ISO format) |
| entry_price | REAL | Entry price |
| return_pct | REAL | 14-day return % |
| label | INTEGER | 0=buy, 1=hold, 2=sell |
| exit_reason | TEXT | How trade ended |
| features | TEXT | JSON of 204 features |
| created_at | TEXT | Generation timestamp |

## Configuration

Edit `metadata/archive_config.json` to adjust:
- `archive_mix_ratio`: Percentage of archive samples (default: 0.07)
- `event_weights`: Relative importance of each event
- `active_events`: Which events to include

Changes take effect on next training run.

## Validation

After generation, validate archive:

```python
from advanced_ml.archive.rare_event_archive import RareEventArchive

archive = RareEventArchive()
stats = archive.get_archive_stats()
print(stats)
```

Expected output:
- Total samples: ~18,000-24,000
- All 8 events present (3 crash, 1 recovery, 4 high_vol)
- Feature count: 204 per sample
- Label distribution: Skewed toward "sell" (crashes)
- Regime distribution: ~40% crash, ~15% recovery, ~45% high_volatility

## Maintenance

**Monthly:** Review generation log for any fetch errors
**Quarterly:** Validate archive integrity
**Yearly:** Consider adding new major events if they occur

## Notes

- Archive uses same feature engineering as main pipeline (204 features)
- Regime/macro features work for historical dates (VIX, yields available back to 1990s)
- Symbol availability varies by event (30-80 symbols depending on IPO dates)
- Archive samples never overwrite main database (separate .db file)
