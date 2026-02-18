"""
Regime Timeline Tracking Engine
Tracks evolution of HOLD signals over time using in-memory cache

RESTRICTIONS:
- ONLY for HOLD signals
- NO database schema modifications
- NO persistent storage required
- Uses in-memory cache for timeline history
- NO modifications to prediction engine
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

# In-memory cache for timeline data
# Structure: {symbol: {timestamp: metrics_snapshot}}
_timeline_cache: Dict[str, Dict[str, Dict]] = {}

# Maximum timeline history per symbol (keep last 100 snapshots)
MAX_HISTORY_LENGTH = 100

# Expiration time for stale timelines (7 days)
TIMELINE_EXPIRATION_DAYS = 7


def add_snapshot(
    symbol: str,
    drift: float,
    pressure_index: str,
    volatility_regime: str,
    quality_score: int,
    regime_state: str,
    pnl: Optional[float] = None
) -> None:
    """
    Add a new snapshot to the timeline for a symbol

    Args:
        symbol: Stock ticker
        drift: Current drift value
        pressure_index: Pressure classification
        volatility_regime: Volatility classification
        quality_score: Quality score
        regime_state: Current regime state
        pnl: Current P&L (optional)
    """
    try:
        timestamp = datetime.now().isoformat()

        # Initialize symbol cache if not exists
        if symbol not in _timeline_cache:
            _timeline_cache[symbol] = {}

        # Add snapshot
        _timeline_cache[symbol][timestamp] = {
            'drift': drift,
            'pressure_index': pressure_index,
            'volatility_regime': volatility_regime,
            'quality_score': quality_score,
            'regime_state': regime_state,
            'pnl': pnl
        }

        # Trim old snapshots if exceeds max length
        if len(_timeline_cache[symbol]) > MAX_HISTORY_LENGTH:
            # Keep only most recent snapshots
            sorted_keys = sorted(_timeline_cache[symbol].keys())
            keys_to_remove = sorted_keys[:-MAX_HISTORY_LENGTH]
            for key in keys_to_remove:
                del _timeline_cache[symbol][key]

        logger.debug(f"[TIMELINE] Added snapshot for {symbol} (total: {len(_timeline_cache[symbol])})")

    except Exception as e:
        logger.error(f"[TIMELINE] Error adding snapshot for {symbol}: {e}")


def get_timeline(symbol: str) -> Dict:
    """
    Get full timeline for a symbol

    Args:
        symbol: Stock ticker

    Returns:
        Dictionary with timeline data and summary statistics
    """
    try:
        if symbol not in _timeline_cache or not _timeline_cache[symbol]:
            return {
                'symbol': symbol,
                'snapshots': [],
                'snapshot_count': 0,
                'first_seen': None,
                'last_seen': None,
                'summary': {}
            }

        snapshots = _timeline_cache[symbol]
        sorted_timestamps = sorted(snapshots.keys())

        # Calculate summary statistics
        summary = calculate_timeline_summary(symbol, snapshots, sorted_timestamps)

        return {
            'symbol': symbol,
            'snapshots': [
                {'timestamp': ts, **snapshots[ts]}
                for ts in sorted_timestamps
            ],
            'snapshot_count': len(snapshots),
            'first_seen': sorted_timestamps[0] if sorted_timestamps else None,
            'last_seen': sorted_timestamps[-1] if sorted_timestamps else None,
            'summary': summary
        }

    except Exception as e:
        logger.error(f"[TIMELINE] Error getting timeline for {symbol}: {e}")
        return {
            'symbol': symbol,
            'snapshots': [],
            'snapshot_count': 0,
            'summary': {}
        }


def calculate_timeline_summary(
    symbol: str,
    snapshots: Dict[str, Dict],
    sorted_timestamps: List[str]
) -> Dict:
    """
    Calculate summary statistics from timeline

    Args:
        symbol: Stock ticker
        snapshots: Timeline snapshots
        sorted_timestamps: Sorted list of timestamps

    Returns:
        Summary statistics dictionary
    """
    try:
        if not sorted_timestamps:
            return {}

        # Get drift history
        drift_values = [snapshots[ts]['drift'] for ts in sorted_timestamps]
        quality_values = [snapshots[ts]['quality_score'] for ts in sorted_timestamps]

        # Calculate trends
        drift_trend = calculate_trend(drift_values)
        quality_trend = calculate_trend(quality_values)

        # Count regime changes
        regime_changes = 0
        for i in range(1, len(sorted_timestamps)):
            prev_regime = snapshots[sorted_timestamps[i-1]]['regime_state']
            curr_regime = snapshots[sorted_timestamps[i]]['regime_state']
            if prev_regime != curr_regime:
                regime_changes += 1

        # Get current vs initial metrics
        initial = snapshots[sorted_timestamps[0]]
        current = snapshots[sorted_timestamps[-1]]

        return {
            'drift_trend': drift_trend,
            'quality_trend': quality_trend,
            'regime_changes': regime_changes,
            'initial_regime': initial['regime_state'],
            'current_regime': current['regime_state'],
            'drift_change': round(current['drift'] - initial['drift'], 2),
            'quality_change': current['quality_score'] - initial['quality_score'],
            'avg_quality': round(sum(quality_values) / len(quality_values), 1),
            'avg_drift': round(sum([abs(d) for d in drift_values]) / len(drift_values), 2)
        }

    except Exception as e:
        logger.error(f"[TIMELINE] Error calculating summary for {symbol}: {e}")
        return {}


def calculate_trend(values: List[float]) -> str:
    """
    Calculate trend direction from a list of values

    Args:
        values: List of numeric values

    Returns:
        Trend direction: "Improving", "Declining", "Stable"
    """
    if len(values) < 2:
        return "Stable"

    # Simple linear trend: compare first half to second half
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / len(values[:mid])
    second_half_avg = sum(values[mid:]) / len(values[mid:])

    delta = second_half_avg - first_half_avg

    if abs(delta) < 0.5:
        return "Stable"
    elif delta > 0:
        return "Improving"
    else:
        return "Declining"


def get_previous_snapshot(symbol: str) -> Optional[Dict]:
    """
    Get the previous snapshot for a symbol (used for transition detection)

    Args:
        symbol: Stock ticker

    Returns:
        Previous snapshot or None if no history
    """
    try:
        if symbol not in _timeline_cache or len(_timeline_cache[symbol]) < 2:
            return None

        sorted_timestamps = sorted(_timeline_cache[symbol].keys())
        previous_timestamp = sorted_timestamps[-2]  # Second most recent

        return _timeline_cache[symbol][previous_timestamp]

    except Exception as e:
        logger.error(f"[TIMELINE] Error getting previous snapshot for {symbol}: {e}")
        return None


def cleanup_stale_timelines() -> int:
    """
    Remove timelines that haven't been updated in TIMELINE_EXPIRATION_DAYS

    Returns:
        Number of timelines cleaned up
    """
    try:
        cutoff_time = datetime.now() - timedelta(days=TIMELINE_EXPIRATION_DAYS)
        symbols_to_remove = []

        for symbol, snapshots in _timeline_cache.items():
            if not snapshots:
                symbols_to_remove.append(symbol)
                continue

            # Get most recent timestamp
            sorted_timestamps = sorted(snapshots.keys())
            last_timestamp = datetime.fromisoformat(sorted_timestamps[-1])

            if last_timestamp < cutoff_time:
                symbols_to_remove.append(symbol)

        # Remove stale timelines
        for symbol in symbols_to_remove:
            del _timeline_cache[symbol]

        if symbols_to_remove:
            logger.info(f"[TIMELINE] Cleaned up {len(symbols_to_remove)} stale timelines")

        return len(symbols_to_remove)

    except Exception as e:
        logger.error(f"[TIMELINE] Error cleaning up stale timelines: {e}")
        return 0


def get_cache_stats() -> Dict:
    """
    Get statistics about the timeline cache

    Returns:
        Dictionary with cache statistics
    """
    total_symbols = len(_timeline_cache)
    total_snapshots = sum(len(snapshots) for snapshots in _timeline_cache.values())

    return {
        'total_symbols': total_symbols,
        'total_snapshots': total_snapshots,
        'avg_snapshots_per_symbol': round(total_snapshots / total_symbols, 1) if total_symbols > 0 else 0
    }
