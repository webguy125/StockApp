"""
Symbol Metadata - Add Sector and Market Cap Features
This adds 3 critical categorical features to reach 179 total features

Features Added:
1. sector_code (0-10): GICS sector encoded as integer
2. market_cap_tier (0-2): 0=large, 1=mid, 2=small
3. symbol_hash (0-79): Symbol position in curated list

These features help models learn sector-specific and cap-tier-specific patterns
"""

from typing import Dict, Tuple
from advanced_ml.config.core_symbols import CORE_SYMBOLS, SECTOR_CODES


class SymbolMetadata:
    """
    Provides sector and market cap metadata for symbols
    """

    def __init__(self):
        self.sector_map = {}
        self.market_cap_map = {}
        self.symbol_index_map = {}
        self._build_maps()

    def _build_maps(self):
        """Build reverse lookup maps from CORE_SYMBOLS"""
        symbol_index = 0

        for sector_name, tiers in CORE_SYMBOLS.items():
            sector_code = SECTOR_CODES[sector_name]

            for tier_name, symbols in tiers.items():
                # Map tier name to integer (0=large, 1=mid, 2=small)
                if tier_name == 'large_cap':
                    tier_code = 0
                elif tier_name == 'mid_cap':
                    tier_code = 1
                else:  # small_cap
                    tier_code = 2

                for symbol in symbols:
                    self.sector_map[symbol] = sector_code
                    self.market_cap_map[symbol] = tier_code
                    self.symbol_index_map[symbol] = symbol_index
                    symbol_index += 1

    def get_metadata_features(self, symbol: str) -> Dict[str, float]:
        """
        Get metadata features for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with 3 features:
            - sector_code: GICS sector (0-10)
            - market_cap_tier: Size category (0=large, 1=mid, 2=small)
            - symbol_hash: Index in curated list (0-79)
        """
        return {
            'sector_code': float(self.sector_map.get(symbol, 0)),
            'market_cap_tier': float(self.market_cap_map.get(symbol, 0)),
            'symbol_hash': float(self.symbol_index_map.get(symbol, 0))
        }

    def get_sector_name(self, symbol: str) -> str:
        """Get human-readable sector name"""
        sector_code = self.sector_map.get(symbol, 0)
        for name, code in SECTOR_CODES.items():
            if code == sector_code:
                return name
        return 'unknown'

    def get_market_cap_name(self, symbol: str) -> str:
        """Get human-readable market cap tier"""
        tier = self.market_cap_map.get(symbol, 0)
        return ['large_cap', 'mid_cap', 'small_cap'][tier]


# Global instance
_metadata = SymbolMetadata()


def get_symbol_metadata(symbol: str) -> Dict[str, float]:
    """
    Get metadata features for a symbol (global accessor)

    Args:
        symbol: Stock symbol

    Returns:
        Dict with sector_code, market_cap_tier, symbol_hash
    """
    return _metadata.get_metadata_features(symbol)


def get_sector_and_cap(symbol: str) -> Tuple[str, str]:
    """
    Get human-readable sector and market cap

    Returns:
        Tuple of (sector_name, market_cap_tier)
    """
    return _metadata.get_sector_name(symbol), _metadata.get_market_cap_name(symbol)
