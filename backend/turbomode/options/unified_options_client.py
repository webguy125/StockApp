"""
Unified Options Client
======================

Provides a unified interface for fetching option chain data with automatic failover.

Data Source Priority:
1. Tradier API (primary) - Real-time market data with greeks
2. Yahoo Finance (fallback) - Free backup when Tradier unavailable

Interface matches IBKR client format for drop-in replacement:
- connect() -> bool
- disconnect() -> None
- fetch_option_chain(symbol) -> Optional[Dict]

Return Format:
{
    'expirations': ['20260220', '20260227', ...],
    'chains': {
        '20260220': {
            'calls': {
                150.0: {'bid': 2.5, 'ask': 2.6, 'mid': 2.55},
                ...
            },
            'puts': {
                150.0: {'bid': 1.8, 'ask': 1.9, 'mid': 1.85},
                ...
            }
        }
    }
}
"""

import os
import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedOptionsClient:
    """
    Unified options client with Tradier (primary) and Yahoo (fallback).

    Drop-in replacement for IBKROptionClient.
    """

    def __init__(self):
        """Initialize unified options client"""
        self.tradier_client = None
        self.yahoo_available = True
        self.connected = False

        # Try to initialize Tradier client
        try:
            from backend.turbomode.options.tradier_options_client import TradierOptionsClient
            self.tradier_client = TradierOptionsClient()
            logger.info("[UNIFIED OPTIONS] Tradier client initialized")
        except Exception as e:
            logger.warning(f"[UNIFIED OPTIONS] Tradier client unavailable: {e}")
            self.tradier_client = None

    def connect(self) -> bool:
        """
        Connect to options data sources.

        Returns:
            True if at least one source is available, False otherwise
        """
        # Tradier is always "connected" (REST API, no persistent connection)
        if self.tradier_client:
            self.connected = True
            logger.info("[UNIFIED OPTIONS] Connected (Tradier primary, Yahoo fallback)")
            return True

        # Yahoo is always available as fallback
        if self.yahoo_available:
            self.connected = True
            logger.info("[UNIFIED OPTIONS] Connected (Yahoo fallback only)")
            return True

        logger.error("[UNIFIED OPTIONS] No data sources available")
        return False

    def disconnect(self):
        """Disconnect from options data sources (no-op for REST APIs)"""
        self.connected = False
        logger.info("[UNIFIED OPTIONS] Disconnected")

    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain for a symbol using Tradier (primary) or Yahoo (fallback).

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with expirations and chains, or None if all sources fail
        """
        if not self.connected:
            logger.warning(f"[UNIFIED OPTIONS] Not connected, attempting auto-connect...")
            if not self.connect():
                return None

        # Try Tradier first (primary)
        if self.tradier_client:
            try:
                result = self._fetch_from_tradier(symbol)
                if result:
                    logger.info(f"[UNIFIED OPTIONS] Fetched {symbol} from Tradier (primary)")
                    return result
            except Exception as e:
                logger.warning(f"[UNIFIED OPTIONS] Tradier fetch failed for {symbol}: {e}")

        # Fallback to Yahoo
        if self.yahoo_available:
            try:
                result = self._fetch_from_yahoo(symbol)
                if result:
                    logger.info(f"[UNIFIED OPTIONS] Fetched {symbol} from Yahoo (fallback)")
                    return result
            except Exception as e:
                logger.warning(f"[UNIFIED OPTIONS] Yahoo fetch failed for {symbol}: {e}")

        logger.error(f"[UNIFIED OPTIONS] All sources failed for {symbol}")
        return None

    def _fetch_from_tradier(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain from Tradier and convert to unified format.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary in unified format, or None if failed
        """
        # Get expirations
        expirations_data = self.tradier_client.get_expirations(symbol)
        if not expirations_data:
            return None

        # Extract expiration dates (convert YYYY-MM-DD to YYYYMMDD format)
        expirations = []
        for exp in expirations_data:
            exp_date = exp.get('date', '')
            if exp_date:
                # Convert 2026-02-20 to 20260220
                exp_formatted = exp_date.replace('-', '')
                expirations.append(exp_formatted)

        if not expirations:
            return None

        # Fetch chains for all expirations
        chains = {}
        for exp_formatted in expirations:
            # Convert back to YYYY-MM-DD for Tradier API
            exp_date = f"{exp_formatted[0:4]}-{exp_formatted[4:6]}-{exp_formatted[6:8]}"

            chain_data = self.tradier_client.get_option_chain(symbol, exp_date, greeks=True)
            if chain_data:
                # Convert to unified format (strike -> {'bid', 'ask', 'mid'})
                calls = {}
                puts = {}

                for strike, data in chain_data.get('calls', {}).items():
                    calls[float(strike)] = {
                        'bid': data.get('bid', 0),
                        'ask': data.get('ask', 0),
                        'mid': data.get('mid', 0)
                    }

                for strike, data in chain_data.get('puts', {}).items():
                    puts[float(strike)] = {
                        'bid': data.get('bid', 0),
                        'ask': data.get('ask', 0),
                        'mid': data.get('mid', 0)
                    }

                chains[exp_formatted] = {
                    'calls': calls,
                    'puts': puts
                }

        if not chains:
            return None

        return {
            'expirations': expirations,
            'chains': chains
        }

    def _fetch_from_yahoo(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain from Yahoo Finance and convert to unified format.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary in unified format, or None if failed
        """
        try:
            import yfinance as yf
            from datetime import datetime

            ticker = yf.Ticker(symbol)

            # Get expirations
            exp_dates = ticker.options
            if not exp_dates:
                return None

            # Convert to YYYYMMDD format
            expirations = []
            for exp_str in exp_dates:
                # exp_str format: '2026-02-20'
                exp_formatted = exp_str.replace('-', '')
                expirations.append(exp_formatted)

            # Fetch chains for all expirations
            chains = {}
            for exp_formatted, exp_str in zip(expirations, exp_dates):
                try:
                    chain = ticker.option_chain(exp_str)

                    # Convert calls
                    calls = {}
                    if chain.calls is not None and not chain.calls.empty:
                        for _, row in chain.calls.iterrows():
                            strike = float(row['strike'])
                            bid = float(row.get('bid', 0))
                            ask = float(row.get('ask', 0))
                            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                            calls[strike] = {
                                'bid': bid,
                                'ask': ask,
                                'mid': mid
                            }

                    # Convert puts
                    puts = {}
                    if chain.puts is not None and not chain.puts.empty:
                        for _, row in chain.puts.iterrows():
                            strike = float(row['strike'])
                            bid = float(row.get('bid', 0))
                            ask = float(row.get('ask', 0))
                            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                            puts[strike] = {
                                'bid': bid,
                                'ask': ask,
                                'mid': mid
                            }

                    chains[exp_formatted] = {
                        'calls': calls,
                        'puts': puts
                    }

                except Exception as e:
                    logger.warning(f"[UNIFIED OPTIONS] Yahoo chain fetch failed for {symbol} {exp_str}: {e}")
                    continue

            if not chains:
                return None

            return {
                'expirations': expirations,
                'chains': chains
            }

        except Exception as e:
            logger.error(f"[UNIFIED OPTIONS] Yahoo fetch error for {symbol}: {e}")
            return None


# Singleton instance
_unified_client_instance = None

def get_unified_options_client() -> UnifiedOptionsClient:
    """
    Get singleton instance of unified options client.

    Returns:
        UnifiedOptionsClient instance
    """
    global _unified_client_instance
    if _unified_client_instance is None:
        _unified_client_instance = UnifiedOptionsClient()
    return _unified_client_instance
