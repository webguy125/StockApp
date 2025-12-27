"""
Reset Checkpoint - Start Training from Scratch
WARNING: This will backup current checkpoint and start fresh
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.checkpoint_manager import CheckpointManager


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESET CHECKPOINT")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Backup current checkpoint")
    print("  2. Reset all progress")
    print("  3. Start training from scratch")
    print("\nNOTE: This does NOT delete database data!")
    print("=" * 60)

    response = input("\nAre you sure? (yes/no): ").strip().lower()

    if response == "yes":
        manager = CheckpointManager()
        manager.reset()
        print("\n[OK] Checkpoint reset complete!")
        print("Run 'python run_training_with_checkpoints.py' to start fresh.")
    else:
        print("\n[CANCELLED] No changes made.")
