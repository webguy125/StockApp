# Phase 2 Implementation Plan: Monitoring & Quality Control

**Status**: Planning Complete - Ready for Implementation
**Date**: 2025-12-23
**Phase**: 2 of 3
**Modules**: 6, 8, 9, 10

---

## Overview

Phase 2 adds production monitoring, error analysis, and model validation to ensure the ML trading system maintains high quality over time. This phase builds on Phase 1's regime-aware training foundation by adding continuous monitoring and quality gates.

### Phase 2 Modules

- **Module 6**: Drift Detection - Monitor feature/regime/prediction changes
- **Module 8**: Error Replay Buffer - Store and replay worst predictions
- **Module 9**: Sector/Symbol Tracking - Performance by GICS sector and symbol
- **Module 10**: Model Promotion Gate - Multi-criteria validation before deployment

### Dependencies

- **Phase 1 Complete**: ✅ All 5 modules validated (7/7 integration tests passed)
- **Archive Generation**: 🔄 Running (Process 2a985f, ETA 12-20 hours)

---

## Module 6: Drift Detection

**Purpose**: Monitor data distribution changes over time to detect when the market regime shifts or features become stale.

### Components to Build

#### 6.1 DriftDetector Class
**File**: `backend/advanced_ml/monitoring/drift_detector.py`

```python
class DriftDetector:
    """
    Monitors three types of drift:
    1. Feature Drift - Are technical indicators changing distribution?
    2. Regime Drift - Are we staying in same regime or shifting?
    3. Prediction Drift - Are model predictions changing pattern?
    """

    def __init__(self, window_size=100, alert_threshold=0.15):
        """
        Args:
            window_size: Number of samples for baseline comparison
            alert_threshold: KS statistic threshold (0.15 = 15% drift)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.baseline_features = None
        self.baseline_regime_dist = None
        self.baseline_predictions = None

    def set_baseline(self, features: np.ndarray, regimes: List[str], predictions: np.ndarray):
        """Store baseline distributions for comparison"""

    def detect_feature_drift(self, new_features: np.ndarray) -> Dict[str, Any]:
        """
        Use Kolmogorov-Smirnov test to compare feature distributions
        Returns: {
            'drift_detected': bool,
            'ks_statistic': float,
            'drifted_features': List[int],  # Feature indices with drift
            'max_drift': float
        }
        """

    def detect_regime_drift(self, new_regimes: List[str]) -> Dict[str, Any]:
        """
        Compare regime distribution (crash vs normal vs recovery)
        Returns: {
            'drift_detected': bool,
            'regime_shift': str,  # "normal -> crash" or "stable"
            'distribution_change': float
        }
        """

    def detect_prediction_drift(self, new_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Check if model prediction distribution has changed
        Returns: {
            'drift_detected': bool,
            'prediction_shift': float,
            'avg_confidence_change': float
        }
        """
```

#### 6.2 Database Schema
**File**: `backend/advanced_ml/database/schema.py` (add to existing)

```sql
-- Drift monitoring table
CREATE TABLE drift_monitoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    drift_type TEXT NOT NULL,  -- 'feature', 'regime', 'prediction'
    ks_statistic REAL,
    drift_detected INTEGER,  -- 0 or 1
    details_json TEXT,  -- JSON with specific drift info
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### 6.3 Integration Points

1. **During Training** (automated_learner.py):
   - After loading new data, check for feature drift
   - Log drift events to database
   - If high drift detected, trigger full retraining

2. **During Prediction** (ml_trading.py):
   - Check regime drift before making predictions
   - Alert if model operating outside trained distribution

### Implementation Steps

1. ✅ Create `backend/advanced_ml/monitoring/` directory
2. ✅ Implement `DriftDetector` class with KS tests
3. ✅ Add drift_monitoring table to database schema
4. ✅ Integrate with automated_learner.py
5. ✅ Add drift alerts to training logs
6. ✅ Test with historical data (simulate regime shifts)

### Success Criteria

- [ ] Detects crash regime shift within 5 trading days
- [ ] Feature drift alerts trigger when VIX spikes 50%+
- [ ] False positive rate < 10% (doesn't alert during normal volatility)
- [ ] Drift events logged to database with timestamps

---

## Module 8: Error Replay Buffer

**Purpose**: Store the worst predictions for targeted retraining, similar to experience replay in reinforcement learning.

### Components to Build

#### 8.1 ErrorReplayBuffer Class
**File**: `backend/advanced_ml/training/error_replay_buffer.py`

```python
class ErrorReplayBuffer:
    """
    Stores worst predictions for replay during training

    Strategy:
    1. During validation, track predictions with highest error
    2. Store top N worst predictions (with features + labels)
    3. During training, replay these samples 3x more frequently
    4. Helps model learn from mistakes
    """

    def __init__(self, max_size=1000, db_path="backend/data/advanced_ml_system.db"):
        """
        Args:
            max_size: Maximum errors to store
        """
        self.max_size = max_size
        self.db_path = db_path

    def add_error(self, features: np.ndarray, label: int, prediction: int,
                  confidence: float, symbol: str, date: str, regime: str):
        """
        Add a prediction error to the buffer

        Only stores if error is in worst N errors (sorted by confidence)
        High-confidence wrong predictions are worst errors
        """

    def get_replay_samples(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve error samples for replay training

        Returns:
            (features, labels) for retraining
        """

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored errors

        Returns: {
            'total_errors': int,
            'by_regime': Dict[str, int],
            'by_label': Dict[str, int],
            'avg_confidence': float
        }
        """

    def clear_buffer(self):
        """Clear all stored errors (after successful retraining)"""
```

#### 8.2 Database Schema
**File**: `backend/advanced_ml/database/schema.py` (add to existing)

```sql
-- Error replay buffer table
CREATE TABLE error_replay_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    regime TEXT,
    features_json TEXT NOT NULL,
    true_label INTEGER NOT NULL,
    predicted_label INTEGER NOT NULL,
    confidence REAL NOT NULL,  -- Higher = worse error (confident but wrong)
    error_score REAL,  -- confidence * (1 if wrong else 0)
    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
    replayed_count INTEGER DEFAULT 0
);

CREATE INDEX idx_error_score ON error_replay_buffer(error_score DESC);
CREATE INDEX idx_regime ON error_replay_buffer(regime);
```

#### 8.3 Integration Points

1. **During Validation** (automated_learner.py):
   - After each validation fold, add worst errors to buffer
   - Store top 100 errors per validation run

2. **During Training** (hybrid_memory_trainer.py):
   - Mix replay samples into training data (20% of batch)
   - Weight replay samples 3x higher than normal samples
   - Clear buffer after successful training

### Implementation Steps

1. ✅ Create `ErrorReplayBuffer` class
2. ✅ Add error_replay_buffer table to database
3. ✅ Integrate with validation in automated_learner.py
4. ✅ Modify training loop to include replay samples
5. ✅ Add error statistics to training reports
6. ✅ Test with intentionally bad predictions

### Success Criteria

- [ ] Buffer stores top 1000 worst predictions
- [ ] Replay samples reduce repeat errors by 30%+
- [ ] High-confidence errors (>0.8) prioritized for replay
- [ ] Error stats show improvement after replay training

---

## Module 9: Sector/Symbol Tracking

**Purpose**: Track performance by GICS sector and individual symbols to identify model strengths/weaknesses.

### Components to Build

#### 9.1 SectorTracker Class
**File**: `backend/advanced_ml/monitoring/sector_tracker.py`

```python
class SectorTracker:
    """
    Tracks model performance by:
    1. GICS Sector (Energy, Technology, Healthcare, etc.)
    2. Individual Symbols
    3. Sector + Regime combinations
    """

    # GICS Sector mappings (from advanced_ml/config.py)
    SECTOR_MAP = {
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'FANG'],
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
        'Healthcare': ['JNJ', 'UNH', 'ABBV', 'LLY', 'DXCM'],
        'Financials': ['JPM', 'BAC', 'GS', 'WFC', 'SCHW'],
        # ... 11 sectors total
    }

    def __init__(self, db_path="backend/data/advanced_ml_system.db"):
        self.db_path = db_path

    def track_prediction(self, symbol: str, regime: str, prediction: int,
                        actual: int, confidence: float):
        """Record a prediction for sector tracking"""

    def get_sector_performance(self, sector: str = None) -> Dict[str, Any]:
        """
        Get accuracy/metrics by sector

        Returns: {
            'Energy': {'accuracy': 0.72, 'total': 1500, 'by_regime': {...}},
            'Technology': {'accuracy': 0.68, 'total': 2000, ...},
            ...
        }
        """

    def get_symbol_performance(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get accuracy for specific symbols

        Returns: {
            'AAPL': {'accuracy': 0.70, 'total': 500, 'best_regime': 'normal'},
            'TSLA': {'accuracy': 0.65, 'total': 450, 'best_regime': 'high_volatility'},
            ...
        }
        """

    def get_weakest_sectors(self, n: int = 3) -> List[Tuple[str, float]]:
        """
        Identify sectors with lowest accuracy

        Returns: [('Energy', 0.58), ('Real Estate', 0.60), ...]
        """
```

#### 9.2 Database Schema
**File**: `backend/advanced_ml/database/schema.py` (add to existing)

```sql
-- Sector/symbol performance tracking
CREATE TABLE sector_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    sector TEXT NOT NULL,
    regime TEXT NOT NULL,
    predicted_label INTEGER NOT NULL,
    actual_label INTEGER,
    confidence REAL NOT NULL,
    correct INTEGER,  -- 0 or 1
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sector_performance ON sector_performance(sector, regime);
CREATE INDEX idx_symbol_performance ON sector_performance(symbol, regime);
```

#### 9.3 Integration Points

1. **During Prediction** (ml_trading.py):
   - After each prediction, log to sector_performance table
   - Include symbol, sector, regime, prediction, confidence

2. **During Training** (automated_learner.py):
   - Generate sector performance report
   - Use weak sector data for targeted data collection

3. **Frontend Display** (ml_trading.html):
   - Add sector performance dashboard
   - Show accuracy by sector in table format

### Implementation Steps

1. ✅ Create `SectorTracker` class
2. ✅ Add sector_performance table to database
3. ✅ Define GICS sector mappings in config.py
4. ✅ Integrate tracking with ml_trading.py
5. ✅ Add sector stats to training reports
6. ✅ Create frontend dashboard for sector metrics

### Success Criteria

- [ ] Tracks all 11 GICS sectors correctly
- [ ] Symbol → Sector mapping 100% accurate
- [ ] Identifies weakest sectors (e.g., Energy at 58% vs Tech at 72%)
- [ ] Frontend displays sector breakdown table

---

## Module 10: Model Promotion Gate

**Purpose**: Multi-criteria validation before promoting a new model to production. Prevents bad models from reaching live trading.

### Components to Build

#### 10.1 ModelPromotionGate Class
**File**: `backend/advanced_ml/training/model_promotion_gate.py`

```python
class ModelPromotionGate:
    """
    Validates model before production deployment

    Criteria (ALL must pass):
    1. Overall Accuracy > 65%
    2. Crash Regime Accuracy > 65% (most important)
    3. No sector accuracy < 55% (prevent weak sectors)
    4. Drift score < 0.20 (model stable)
    5. Error replay improvement > 0% (learning from mistakes)

    Only if ALL pass → promote model
    """

    def __init__(self, db_path="backend/data/advanced_ml_system.db"):
        self.db_path = db_path

        # Validation thresholds
        self.min_overall_accuracy = 0.65
        self.min_crash_accuracy = 0.65
        self.min_sector_accuracy = 0.55
        self.max_drift_score = 0.20
        self.min_replay_improvement = 0.0

    def validate_model(self, model_path: str, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all validation checks

        Args:
            model_path: Path to trained model file
            validation_results: Dict from training (accuracy, regime_accuracy, etc.)

        Returns: {
            'approved': bool,
            'checks_passed': List[str],
            'checks_failed': List[str],
            'details': Dict[str, Any],
            'recommendation': str  # "PROMOTE" or "REJECT"
        }
        """

    def _check_overall_accuracy(self, results: Dict) -> bool:
        """Check overall accuracy > 65%"""

    def _check_crash_accuracy(self, results: Dict) -> bool:
        """Check crash regime accuracy > 65%"""

    def _check_sector_accuracy(self, results: Dict) -> bool:
        """Check no sector < 55% accuracy"""

    def _check_drift_score(self, model_path: str) -> bool:
        """Check drift detector score < 0.20"""

    def _check_replay_improvement(self, results: Dict) -> bool:
        """Check error replay improved performance"""

    def promote_model(self, model_path: str, backup_old: bool = True):
        """
        Copy validated model to production path

        Production paths:
        - backend/data/ml_models/meta_learner_production.pkl
        - backend/data/ml_models/base_models_production/
        """
```

#### 10.2 Database Schema
**File**: `backend/advanced_ml/database/schema.py` (add to existing)

```sql
-- Model promotion history
CREATE TABLE model_promotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path TEXT NOT NULL,
    validation_timestamp TEXT NOT NULL,
    approved INTEGER NOT NULL,  -- 0 or 1
    overall_accuracy REAL,
    crash_accuracy REAL,
    min_sector_accuracy REAL,
    drift_score REAL,
    replay_improvement REAL,
    checks_passed TEXT,  -- JSON array
    checks_failed TEXT,  -- JSON array
    promoted_to_production INTEGER DEFAULT 0,
    notes TEXT
);
```

#### 10.3 Integration Points

1. **After Training** (automated_learner.py):
   - Run ModelPromotionGate validation
   - Only promote if all checks pass
   - Log promotion decision to database

2. **Frontend Display** (ml_settings.html):
   - Show promotion gate status
   - Display which checks passed/failed
   - Manual override option (for emergencies)

### Implementation Steps

1. ✅ Create `ModelPromotionGate` class
2. ✅ Define validation thresholds
3. ✅ Implement all 5 validation checks
4. ✅ Add model_promotion_history table
5. ✅ Integrate with automated_learner.py
6. ✅ Add promotion status to frontend
7. ✅ Test with intentionally bad models

### Success Criteria

- [ ] Rejects models with <65% overall accuracy
- [ ] Rejects models with <65% crash accuracy (most critical)
- [ ] Rejects models with any sector <55% accuracy
- [ ] Approves models meeting all 5 criteria
- [ ] Logs all promotion attempts to database
- [ ] Frontend shows validation report

---

## Implementation Order

### Week 1: Module 6 + Module 8
**Day 1-2**: Drift Detection
- Implement DriftDetector class
- Add database schema
- Integrate with training loop
- Test with simulated drift

**Day 3-4**: Error Replay Buffer
- Implement ErrorReplayBuffer class
- Add database schema
- Integrate with validation
- Test replay training

### Week 2: Module 9 + Module 10
**Day 5-6**: Sector/Symbol Tracking
- Implement SectorTracker class
- Define GICS sector mappings
- Add database schema
- Create frontend dashboard

**Day 7-8**: Model Promotion Gate
- Implement ModelPromotionGate class
- Define validation criteria
- Add database schema
- Test with various model quality levels

### Week 3: Integration & Testing
**Day 9-10**: End-to-End Testing
- Test all 4 modules together
- Validate promotion gate rejects bad models
- Confirm drift detection triggers retraining
- Verify sector tracking accuracy

---

## Testing Strategy

### Unit Tests
- `test_drift_detector.py` - KS test accuracy
- `test_error_replay.py` - Buffer operations
- `test_sector_tracker.py` - Sector mapping
- `test_promotion_gate.py` - Validation logic

### Integration Tests
- `test_phase2_integration.py` - All 4 modules working together
- Simulate:
  - Market crash → drift detected → retraining triggered
  - Bad model → promotion gate rejects
  - Weak sector → targeted error replay

### Success Criteria (Overall)
- [ ] Drift detection alerts within 5 days of regime shift
- [ ] Error replay reduces repeat errors by 30%
- [ ] Sector tracking identifies all 11 GICS sectors
- [ ] Promotion gate rejects models with <65% crash accuracy
- [ ] All modules log to database correctly
- [ ] Frontend displays all monitoring metrics

---

## Database Migration

Add to `backend/advanced_ml/database/schema.py`:

```python
def upgrade_to_phase2_schema(db_path: str):
    """Add Phase 2 tables to existing database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Module 6: Drift Detection
    cursor.execute('''CREATE TABLE IF NOT EXISTS drift_monitoring (...)''')

    # Module 8: Error Replay Buffer
    cursor.execute('''CREATE TABLE IF NOT EXISTS error_replay_buffer (...)''')

    # Module 9: Sector Tracking
    cursor.execute('''CREATE TABLE IF NOT EXISTS sector_performance (...)''')

    # Module 10: Model Promotion
    cursor.execute('''CREATE TABLE IF NOT EXISTS model_promotion_history (...)''')

    conn.commit()
    conn.close()
```

---

## Files to Create

### New Directories
- `backend/advanced_ml/monitoring/`
- `backend/advanced_ml/training/` (may already exist)

### New Python Files
1. `backend/advanced_ml/monitoring/drift_detector.py` (~200 lines)
2. `backend/advanced_ml/monitoring/sector_tracker.py` (~150 lines)
3. `backend/advanced_ml/training/error_replay_buffer.py` (~150 lines)
4. `backend/advanced_ml/training/model_promotion_gate.py` (~200 lines)

### Modified Files
1. `backend/advanced_ml/database/schema.py` - Add 4 new tables
2. `backend/advanced_ml/automated_learner.py` - Integrate all modules
3. `frontend/ml_settings.html` - Add promotion gate status
4. `frontend/ml_trading.html` - Add sector performance dashboard

### Test Files
1. `test_drift_detector.py`
2. `test_error_replay.py`
3. `test_sector_tracker.py`
4. `test_promotion_gate.py`
5. `test_phase2_integration.py`

---

## Risk Mitigation

### Risk 1: False Drift Alerts
**Mitigation**: Use 100-sample window and 0.15 threshold (15% drift) to reduce noise

### Risk 2: Error Replay Overfitting
**Mitigation**: Limit replay samples to 20% of training batch, weight 3x (not 10x)

### Risk 3: Promotion Gate Too Strict
**Mitigation**: Manual override option in frontend for emergencies

### Risk 4: Sector Mapping Errors
**Mitigation**: Use official GICS sector classifications, test all symbol mappings

---

## Completion Checklist

### Module 6: Drift Detection
- [ ] DriftDetector class implemented
- [ ] KS test for feature drift working
- [ ] Regime shift detection working
- [ ] Database logging functional
- [ ] Integration with training complete
- [ ] Unit tests passing

### Module 8: Error Replay Buffer
- [ ] ErrorReplayBuffer class implemented
- [ ] Top-N error storage working
- [ ] Replay sample retrieval working
- [ ] Database storage functional
- [ ] Integration with training complete
- [ ] Unit tests passing

### Module 9: Sector/Symbol Tracking
- [ ] SectorTracker class implemented
- [ ] GICS sector mappings complete
- [ ] Performance tracking working
- [ ] Database storage functional
- [ ] Frontend dashboard created
- [ ] Unit tests passing

### Module 10: Model Promotion Gate
- [ ] ModelPromotionGate class implemented
- [ ] All 5 validation checks working
- [ ] Promotion/rejection logic correct
- [ ] Database logging functional
- [ ] Integration with training complete
- [ ] Unit tests passing

### Integration Testing
- [ ] All 4 modules work together
- [ ] End-to-end test passing
- [ ] Frontend displays all metrics
- [ ] Production deployment tested

---

## Next Steps After Phase 2

Once Phase 2 is complete, proceed to **Phase 3: Advanced Analytics & Automation**:
- Module 7: Dynamic Archive Updates
- Module 11: SHAP Feature Analysis
- Module 12: Training Orchestrator

Phase 3 builds on Phase 2's monitoring to enable autonomous operation and deep model interpretability.
