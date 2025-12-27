"""
View Current Checkpoint Status
Quick script to check training progress
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.checkpoint_manager import CheckpointManager


if __name__ == "__main__":
    manager = CheckpointManager()
    print(manager.get_summary())

    # Show completed symbols if any
    completed = manager.state['backtest']['completed_symbols']
    if completed:
        print(f"\nCompleted symbols ({len(completed)}):")
        for i, symbol in enumerate(completed, 1):
            print(f"  {i}. {symbol}")

    # Show failed symbols if any
    failed = manager.state['backtest']['failed_symbols']
    if failed:
        print(f"\nFailed symbols ({len(failed)}):")
        for i, symbol in enumerate(failed, 1):
            print(f"  {i}. {symbol}")

    # Show trained models
    trained = manager.state['training']['base_models_trained']
    if trained:
        print(f"\nTrained models ({len(trained)}):")
        for i, model in enumerate(trained, 1):
            print(f"  {i}. {model}")
