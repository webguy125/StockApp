"""
Expiration Selector for Iron Condors
Selects shortest valid expiration: try 2 weeks, else 3 weeks, else scan forward
If none found returns None
"""

from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def select_expiration(expirations: List[str]) -> Optional[str]:
    """
    Select best expiration for Iron Condor

    Strategy:
    1. Try to find expiration ~14 days out (2 weeks)
    2. If not found, try ~21 days out (3 weeks)
    3. If not found, take first expiration > 7 days out
    4. If none found, return None

    Args:
        expirations: List of expiration strings in format 'YYYYMMDD'

    Returns:
        Selected expiration string, or None if no valid expiration
    """
    if not expirations:
        logger.warning("[EXPIRATION] No expirations provided")
        return None

    try:
        today = datetime.now().date()

        # Parse all expirations and calculate days to expiration
        exp_data = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y%m%d').date()
                dte = (exp_date - today).days

                if dte > 0:  # Only consider future expirations
                    exp_data.append((exp_str, dte))
            except:
                continue

        if not exp_data:
            logger.warning("[EXPIRATION] No valid future expirations")
            return None

        # Sort by DTE
        exp_data.sort(key=lambda x: x[1])

        # Strategy 1: Find expiration closest to 14 days
        target_14d = min(exp_data, key=lambda x: abs(x[1] - 14))
        if 10 <= target_14d[1] <= 18:  # Within +/- 4 days of 2 weeks
            logger.info(f"[EXPIRATION] Selected {target_14d[0]} ({target_14d[1]} DTE) - closest to 14d target")
            return target_14d[0]

        # Strategy 2: Find expiration closest to 21 days
        target_21d = min(exp_data, key=lambda x: abs(x[1] - 21))
        if 17 <= target_21d[1] <= 25:  # Within +/- 4 days of 3 weeks
            logger.info(f"[EXPIRATION] Selected {target_21d[0]} ({target_21d[1]} DTE) - closest to 21d target")
            return target_21d[0]

        # Strategy 3: Take first expiration > 7 days out
        for exp_str, dte in exp_data:
            if dte > 7:
                logger.info(f"[EXPIRATION] Selected {exp_str} ({dte} DTE) - first valid > 7d")
                return exp_str

        # No valid expiration found
        logger.warning(f"[EXPIRATION] No suitable expiration (all < 7 DTE)")
        return None

    except Exception as e:
        logger.error(f"[EXPIRATION] Error selecting expiration: {e}")
        return None
