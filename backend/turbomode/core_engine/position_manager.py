
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Position Manager - Persistent State Management for TurboMode

Handles position state persistence to survive system restarts.
Supports atomic writes and deterministic state recovery.
"""

import json
import os
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import threading


class PositionManager:
    """
    Manages persistent position state for all symbols.

    Position State Schema:
    {
        "symbol": str,
        "position": "flat" | "long" | "short",
        "entry_price": float,
        "entry_time": str (ISO timestamp),
        "current_price": float,
        "stop_price": float,
        "target_price": float,
        "initial_stop_distance": float,
        "reward_ratio": float,
        "position_size": float,
        "partial_1R_done": bool,
        "partial_2R_done": bool,
        "partial_3R_done": bool,
        "entry_confidence": float,
        "horizon": str ("1d" | "2d" | "5d"),
        "sector": str,
        "atr_at_entry": float,
        "recent_signals": list[str],  # Last N signals
        "persistence_counter": int,   # Counter for exit signal persistence
        "last_update": str (ISO timestamp)
    }
    """

    def __init__(self, state_file: str = None):
        """
        Initialize position manager.

        Args:
            state_file: Path to JSON state file (default: backend/data/position_state.json)
        """
        if state_file is None:
            state_file = r"C:\StockApp\backend\data\position_state.json"

        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory state cache
        self._positions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        # Load existing state
        self._load_state()

        print(f"[POSITION_MANAGER] Initialized with state file: {self.state_file}")
        print(f"[POSITION_MANAGER] Loaded {len(self._positions)} existing positions")

    def _load_state(self):
        """Load position state from disk."""
        if not self.state_file.exists():
            self._positions = {}
            return

        try:
            with open(self.state_file, 'r') as f:
                self._positions = json.load(f)
            print(f"[OK] Loaded {len(self._positions)} positions from {self.state_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load position state: {e}")
            self._positions = {}

    def _save_state(self):
        """Save position state to disk atomically."""
        try:
            # Write to temporary file
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self._positions, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

        except Exception as e:
            print(f"[ERROR] Failed to save position state: {e}")

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position state for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position dict or None if flat
        """
        with self._lock:
            pos = self._positions.get(symbol)
            if pos and pos['position'] == 'flat':
                return None
            return pos.copy() if pos else None

    def is_flat(self, symbol: str) -> bool:
        """Check if position is flat."""
        pos = self.get_position(symbol)
        return pos is None or pos['position'] == 'flat'

    def is_long(self, symbol: str) -> bool:
        """Check if position is long."""
        pos = self.get_position(symbol)
        return pos is not None and pos['position'] == 'long'

    def is_short(self, symbol: str) -> bool:
        """Check if position is short."""
        pos = self.get_position(symbol)
        return pos is not None and pos['position'] == 'short'

    def open_position(
        self,
        symbol: str,
        position_type: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        stop_distance: float,
        reward_ratio: float,
        position_size: float,
        confidence: float,
        horizon: str,
        sector: str,
        atr: float
    ):
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            position_type: "long" or "short"
            entry_price: Entry price
            stop_price: Initial stop price
            target_price: Initial target price
            stop_distance: Initial stop distance (1R)
            reward_ratio: Target distance / stop distance
            position_size: Number of shares
            confidence: Entry confidence (prob_buy or prob_sell)
            horizon: "1d", "2d", or "5d"
            sector: Sector name
            atr: ATR at entry
        """
        with self._lock:
            self._positions[symbol] = {
                'symbol': symbol,
                'position': position_type,
                'entry_price': entry_price,
                'entry_time': datetime.now().isoformat(),
                'current_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'initial_stop_distance': stop_distance,
                'reward_ratio': reward_ratio,
                'position_size': position_size,
                'partial_1R_done': False,
                'partial_2R_done': False,
                'partial_3R_done': False,
                'entry_confidence': confidence,
                'horizon': horizon,
                'sector': sector,
                'atr_at_entry': atr,
                'recent_signals': [],
                'persistence_counter': 0,
                'last_update': datetime.now().isoformat()
            }
            self._save_state()

        print(f"[POSITION] Opened {position_type.upper()} {symbol} @ ${entry_price:.2f}")
        print(f"           Stop: ${stop_price:.2f} | Target: ${target_price:.2f}")
        print(f"           Size: {position_size:.0f} shares | Confidence: {confidence:.2%}")

    def update_position(self, symbol: str, updates: Dict):
        """
        Update position state.

        Args:
            symbol: Stock symbol
            updates: Dict of fields to update
        """
        with self._lock:
            if symbol not in self._positions:
                return

            self._positions[symbol].update(updates)
            self._positions[symbol]['last_update'] = datetime.now().isoformat()
            self._save_state()

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """
        Close a position.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Exit reason
        """
        with self._lock:
            if symbol not in self._positions:
                return

            pos = self._positions[symbol]
            position_type = pos['position']
            entry_price = pos['entry_price']
            position_size = pos['position_size']

            # Calculate P&L
            if position_type == 'long':
                pnl = (exit_price - entry_price) * position_size
                pnl_pct = (exit_price / entry_price - 1) * 100
            else:  # short
                pnl = (entry_price - exit_price) * position_size
                pnl_pct = (1 - exit_price / entry_price) * 100

            print(f"[POSITION] Closed {position_type.upper()} {symbol} @ ${exit_price:.2f}")
            print(f"           Entry: ${entry_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"           Reason: {reason}")

            # Mark as flat
            self._positions[symbol]['position'] = 'flat'
            self._positions[symbol]['current_price'] = exit_price
            self._positions[symbol]['last_update'] = datetime.now().isoformat()
            self._save_state()

    def get_all_active_positions(self) -> Dict[str, Dict]:
        """Get all active (non-flat) positions."""
        with self._lock:
            return {
                symbol: pos.copy()
                for symbol, pos in self._positions.items()
                if pos['position'] != 'flat'
            }

    def add_signal_to_history(self, symbol: str, signal: str, max_history: int = 5):
        """
        Add a signal to recent signal history.

        Args:
            symbol: Stock symbol
            signal: "BUY", "HOLD", or "SELL"
            max_history: Maximum signals to keep
        """
        with self._lock:
            if symbol not in self._positions:
                return

            recent = self._positions[symbol].get('recent_signals', [])
            recent.append(signal)
            if len(recent) > max_history:
                recent = recent[-max_history:]

            self._positions[symbol]['recent_signals'] = recent
            self._save_state()
