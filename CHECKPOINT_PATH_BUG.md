# CHECKPOINT PATH BUG - NEEDS FIX

## Issue
The checkpoint manager is creating checkpoints in the WRONG directory due to incorrect path resolution.

## Current Behavior (WRONG)
```
backend/advanced_ml/backend/turbomode/data/checkpoints/training_checkpoint.json
```

This creates a duplicate nested directory structure: `backend/advanced_ml/backend/turbomode/`

## Expected Behavior (CORRECT)
```
backend/turbomode/data/checkpoints/training_checkpoint.json
```

## Root Cause
In `backend/turbomode/checkpoint_manager.py` lines 16-21:

```python
def __init__(self, checkpoint_dir: str = "backend/turbomode/data/checkpoints"):
    # Convert to absolute path to avoid duplicate directories
    if not os.path.isabs(checkpoint_dir):
        # Get the project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        checkpoint_dir = project_root / checkpoint_dir
```

The problem:
- `Path(__file__)` = `C:\StockApp\backend\turbomode\checkpoint_manager.py`
- `.parent.parent.parent` = `C:\StockApp\backend\turbomode` → `C:\StockApp\backend` → `C:\StockApp`
- BUT then it appends `backend/turbomode/data/checkpoints` AGAIN
- Result: `C:\StockApp\backend\turbomode\data\checkpoints`... NO WAIT

Actually the issue is:
- The file is imported from `backend/advanced_ml/backtesting/historical_backtest.py`
- So `Path(__file__)` resolves relative to WHERE IT'S IMPORTED FROM
- This creates the wrong path resolution

## Fix Required
Change the checkpoint manager to use an ABSOLUTE path based on a fixed anchor point, not relative to `__file__`.

### Option 1: Use environment variable
```python
PROJECT_ROOT = os.environ.get('STOCKAPP_ROOT', 'C:/StockApp')
checkpoint_dir = os.path.join(PROJECT_ROOT, 'backend/turbomode/data/checkpoints')
```

### Option 2: Search for project root marker
```python
def find_project_root():
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / 'backend').exists() and (current / 'frontend').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")

project_root = find_project_root()
checkpoint_dir = project_root / 'backend/turbomode/data/checkpoints'
```

### Option 3: Just use absolute path directly
```python
def __init__(self, checkpoint_dir: str = None):
    if checkpoint_dir is None:
        # Find the actual location of this file and go up to turbomode/
        this_file = Path(__file__).resolve()
        turbomode_dir = this_file.parent  # Should be backend/turbomode/
        checkpoint_dir = turbomode_dir / 'data' / 'checkpoints'

    self.checkpoint_dir = Path(checkpoint_dir)
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

## Impact
- Currently causing duplicate directory creation
- Checkpoint file IS working (backtest is running successfully)
- Just creates messy directory structure
- Should fix before next run

## Status
- **Current run**: Let it complete, checkpoint is working despite wrong location
- **Next run**: Fix the path resolution before starting
- **Priority**: Medium (not blocking, but needs cleanup)

## Session Info
- Discovered: 2025-12-30, during backtest run
- Backtest status when discovered: 261/510 symbols complete (51%)
