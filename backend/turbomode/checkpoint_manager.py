"""
Checkpoint Manager for Long-Running Training
Saves progress after each symbol/phase to allow resume after restart
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class CheckpointManager:
    """Manages training checkpoints for resume capability"""

    def __init__(self, checkpoint_dir: str = None):
        # Use absolute path based on THIS file's actual location
        if checkpoint_dir is None:
            # This file is in backend/turbomode/checkpoint_manager.py
            # So parent is backend/turbomode/, and we want backend/turbomode/data/checkpoints/
            this_file = Path(__file__).resolve()
            turbomode_dir = this_file.parent  # backend/turbomode/
            checkpoint_dir = turbomode_dir / 'data' / 'checkpoints'
        else:
            checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "training_checkpoint.json"
        self.state = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new one"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                print(f"\n[CHECKPOINT] Loaded existing checkpoint from {self.checkpoint_file}")
                print(f"             Last update: {state.get('last_update', 'Unknown')}")
                return state
        else:
            print(f"\n[CHECKPOINT] No existing checkpoint found, starting fresh")
            return {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'current_phase': 'initialization',
                'backtest': {
                    'completed_symbols': [],
                    'failed_symbols': [],
                    'total_samples': 0,
                    'in_progress': False
                },
                'training': {
                    'base_models_trained': [],
                    'meta_learner_trained': False
                },
                'evaluation': {
                    'completed': False,
                    'results': {}
                }
            }

    def save_checkpoint(self):
        """Save current state to disk"""
        self.state['last_update'] = datetime.now().isoformat()

        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)

        print(f"[CHECKPOINT] Saved at {datetime.now().strftime('%H:%M:%S')}")

    def mark_symbol_complete(self, symbol: str, samples_added: int):
        """Mark a symbol as processed in backtest"""
        if symbol not in self.state['backtest']['completed_symbols']:
            self.state['backtest']['completed_symbols'].append(symbol)
            self.state['backtest']['total_samples'] += samples_added
            self.save_checkpoint()

            completed = len(self.state['backtest']['completed_symbols'])
            print(f"[CHECKPOINT] Symbol {symbol} complete ({completed} total)")

    def mark_symbol_failed(self, symbol: str, error: str):
        """Mark a symbol as failed"""
        if symbol not in self.state['backtest']['failed_symbols']:
            self.state['backtest']['failed_symbols'].append(symbol)
            self.save_checkpoint()
            print(f"[CHECKPOINT] Symbol {symbol} failed: {error}")

    def get_remaining_symbols(self, all_symbols: List[str]) -> List[str]:
        """Get list of symbols that still need processing"""
        completed = set(self.state['backtest']['completed_symbols'])
        failed = set(self.state['backtest']['failed_symbols'])
        processed = completed | failed

        remaining = [s for s in all_symbols if s not in processed]

        print(f"\n[CHECKPOINT] Symbol Status:")
        print(f"             Completed: {len(completed)}")
        print(f"             Failed: {len(failed)}")
        print(f"             Remaining: {len(remaining)}")
        print(f"             Total samples collected: {self.state['backtest']['total_samples']}")

        return remaining

    def is_backtest_complete(self) -> bool:
        """Check if backtest phase is done"""
        return len(self.state['backtest']['completed_symbols']) > 0 and \
               not self.state['backtest']['in_progress']

    def set_phase(self, phase: str):
        """Update current phase"""
        self.state['current_phase'] = phase
        self.save_checkpoint()

    def mark_backtest_start(self):
        """Mark backtest as in progress"""
        self.state['backtest']['in_progress'] = True
        self.state['current_phase'] = 'backtest'
        self.save_checkpoint()

    def mark_backtest_complete(self):
        """Mark backtest as finished and reset checkpoint for next run"""
        print(f"\n[CHECKPOINT] Backtest Complete!")
        print(f"             Symbols processed: {len(self.state['backtest']['completed_symbols'])}")
        print(f"             Total samples: {self.state['backtest']['total_samples']}")
        print(f"             Failed symbols: {len(self.state['backtest']['failed_symbols'])}")

        # Clear checkpoint so next backtest starts fresh
        if self.checkpoint_file.exists():
            backup_file = self.checkpoint_dir / f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(self.checkpoint_file, backup_file)
            print(f"[CHECKPOINT] Backed up to {backup_file.name}")
            print(f"[CHECKPOINT] Checkpoint cleared - next backtest will start fresh")

    def mark_model_trained(self, model_name: str):
        """Mark a base model as trained"""
        if model_name not in self.state['training']['base_models_trained']:
            self.state['training']['base_models_trained'].append(model_name)
            self.state['current_phase'] = 'training_base_models'
            self.save_checkpoint()

            print(f"[CHECKPOINT] Model {model_name} trained ({len(self.state['training']['base_models_trained'])}/8)")

    def mark_meta_learner_trained(self):
        """Mark meta-learner as trained"""
        self.state['training']['meta_learner_trained'] = True
        self.state['current_phase'] = 'training_complete'
        self.save_checkpoint()
        print(f"[CHECKPOINT] Meta-learner trained")

    def mark_evaluation_complete(self, results: Dict[str, Any]):
        """Mark evaluation as complete with results"""
        self.state['evaluation']['completed'] = True
        self.state['evaluation']['results'] = results
        self.state['current_phase'] = 'evaluation_complete'
        self.save_checkpoint()
        print(f"[CHECKPOINT] Evaluation complete")

    def get_summary(self) -> str:
        """Get human-readable summary of checkpoint state"""
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append("CHECKPOINT STATUS")
        summary.append(f"{'='*60}")
        summary.append(f"Current Phase: {self.state['current_phase']}")
        summary.append(f"Last Update: {self.state['last_update']}")
        summary.append(f"\nBacktest Progress:")
        summary.append(f"  Completed symbols: {len(self.state['backtest']['completed_symbols'])}")
        summary.append(f"  Failed symbols: {len(self.state['backtest']['failed_symbols'])}")
        summary.append(f"  Total samples: {self.state['backtest']['total_samples']}")
        summary.append(f"\nTraining Progress:")
        summary.append(f"  Base models: {len(self.state['training']['base_models_trained'])}/8")
        summary.append(f"  Meta-learner: {'YES' if self.state['training']['meta_learner_trained'] else 'NO'}")
        summary.append(f"\nEvaluation:")
        summary.append(f"  Complete: {'YES' if self.state['evaluation']['completed'] else 'NO'}")
        summary.append(f"{'='*60}")

        return "\n".join(summary)

    def reset(self):
        """Reset checkpoint (start fresh)"""
        if self.checkpoint_file.exists():
            backup_file = self.checkpoint_dir / f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(self.checkpoint_file, backup_file)
            print(f"[CHECKPOINT] Backed up to {backup_file}")

        self.state = self._load_checkpoint()
        print(f"[CHECKPOINT] Reset complete")


if __name__ == "__main__":
    # Test checkpoint manager
    manager = CheckpointManager()
    print(manager.get_summary())
