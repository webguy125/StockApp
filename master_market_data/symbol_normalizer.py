"""
Canonical Symbol Normalization System

Establishes Yahoo Finance-style symbols as the canonical internal format across
the entire platform and provides translation layers for external data providers.

Canonical Format Rules:
    - Uppercase letters only
    - Hyphen (-) for class shares: BRK-B, BF-A
    - Hyphen (-) for crypto pairs: BTC-USD, ETH-USD
    - No spaces, dots, slashes, or lowercase
    - Examples: AAPL, GOOGL, BRK-B, BTC-USD

Provider-Specific Formats:
    - Yahoo Finance: BRK-B, BTC-USD (canonical)
    - IBKR: BRK B, BTC/USD (space for class, slash for crypto)
    - Schwab: BRK.B (dot for class)
    - Polygon: BRK.B, X:BTCUSD
    - Tiingo: btcusd (lowercase crypto pairs)
    - Kraken: XXBTZUSD (proprietary crypto codes)

Usage:
    from master_market_data.symbol_normalizer import to_canonical, is_canonical

    # Auto-correct from provider format
    symbol = to_canonical("BRK.B", provider="schwab")  # Returns: BRK-B

    # Validate canonical format
    if is_canonical(symbol):
        db.insert(symbol)  # Safe to write
    else:
        raise ValueError(f"Non-canonical symbol: {symbol}")

Last Updated: 2026-01-06
"""

import re
from typing import Optional, Tuple
from enum import Enum


class Provider(Enum):
    """Supported data providers"""
    YAHOO = "yahoo"
    IBKR = "ibkr"
    SCHWAB = "schwab"
    POLYGON = "polygon"
    TIINGO = "tiingo"
    KRAKEN = "kraken"


# ============================================================================
# CANONICAL FORMAT RULES
# ============================================================================

# Regex pattern for canonical format
CANONICAL_PATTERN = re.compile(r'^[A-Z]+(-[A-Z]+)?$')

# Known canonical crypto pairs (Yahoo Finance format)
CANONICAL_CRYPTO_PAIRS = {
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD',
    'DOGE-USD', 'DOT-USD', 'LINK-USD', 'LTC-USD', 'XRP-USD'
}

# Known canonical class shares (Yahoo Finance format)
CANONICAL_CLASS_SHARES = {
    'BRK-B', 'BRK-A', 'BF-A', 'BF-B', 'GOOGL', 'GOOG'
}


# ============================================================================
# PROVIDER-TO-CANONICAL MAPPING
# ============================================================================

def to_canonical(symbol: str, provider: str = "yahoo") -> str:
    """
    Convert provider-specific symbol format to canonical format.

    Args:
        symbol: Symbol in provider-specific format
        provider: Data provider name (yahoo, ibkr, schwab, polygon, tiingo, kraken)

    Returns:
        Canonical symbol (Yahoo Finance format)

    Raises:
        ValueError: If symbol cannot be mapped to canonical format

    Examples:
        >>> to_canonical("BRK.B", provider="schwab")
        'BRK-B'
        >>> to_canonical("BRK B", provider="ibkr")
        'BRK-B'
        >>> to_canonical("BTC/USD", provider="ibkr")
        'BTC-USD'
        >>> to_canonical("btcusd", provider="tiingo")
        'BTC-USD'
    """
    provider = provider.lower()

    if provider == "yahoo":
        return _from_yahoo(symbol)
    elif provider == "ibkr":
        return _from_ibkr(symbol)
    elif provider == "schwab":
        return _from_schwab(symbol)
    elif provider == "polygon":
        return _from_polygon(symbol)
    elif provider == "tiingo":
        return _from_tiingo(symbol)
    elif provider == "kraken":
        return _from_kraken(symbol)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _from_yahoo(symbol: str) -> str:
    """Yahoo Finance -> Canonical (identity transform)"""
    # Yahoo Finance IS canonical - just validate and normalize case
    symbol = symbol.upper().strip()

    if is_canonical(symbol):
        return symbol
    else:
        raise ValueError(f"Invalid Yahoo Finance symbol format: {symbol}")


def _from_ibkr(symbol: str) -> str:
    """
    IBKR -> Canonical

    IBKR Format:
        - Class shares: "BRK B" (space between ticker and class)
        - Crypto pairs: "BTC/USD" (slash separator)
        - Regular stocks: "AAPL"

    Canonical Format:
        - Class shares: "BRK-B" (hyphen)
        - Crypto pairs: "BTC-USD" (hyphen)
        - Regular stocks: "AAPL"
    """
    symbol = symbol.upper().strip()

    # Handle crypto pairs: BTC/USD -> BTC-USD
    if '/' in symbol:
        parts = symbol.split('/')
        if len(parts) == 2:
            canonical = f"{parts[0]}-{parts[1]}"
            if is_canonical(canonical):
                return canonical

    # Handle class shares with space: BRK B -> BRK-B
    if ' ' in symbol:
        parts = symbol.split(' ')
        if len(parts) == 2 and len(parts[1]) == 1:  # Single letter class
            canonical = f"{parts[0]}-{parts[1]}"
            if is_canonical(canonical):
                return canonical

    # Regular stock - validate and return
    if is_canonical(symbol):
        return symbol

    raise ValueError(f"Cannot map IBKR symbol to canonical: {symbol}")


def _from_schwab(symbol: str) -> str:
    """
    Schwab -> Canonical

    Schwab Format:
        - Class shares: "BRK.B" (dot separator)
        - Regular stocks: "AAPL"

    Canonical Format:
        - Class shares: "BRK-B" (hyphen)
        - Regular stocks: "AAPL"
    """
    symbol = symbol.upper().strip()

    # Handle class shares: BRK.B -> BRK-B
    if '.' in symbol:
        parts = symbol.split('.')
        if len(parts) == 2 and len(parts[1]) == 1:  # Single letter class
            canonical = f"{parts[0]}-{parts[1]}"
            if is_canonical(canonical):
                return canonical

    # Regular stock - validate and return
    if is_canonical(symbol):
        return symbol

    raise ValueError(f"Cannot map Schwab symbol to canonical: {symbol}")


def _from_polygon(symbol: str) -> str:
    """
    Polygon -> Canonical (PLACEHOLDER - implement when Polygon is integrated)

    Polygon Format:
        - Stocks: "AAPL", "BRK.B"
        - Crypto: "X:BTCUSD" (X: prefix for crypto)

    Canonical Format:
        - Stocks: "AAPL", "BRK-B"
        - Crypto: "BTC-USD"
    """
    symbol = symbol.upper().strip()

    # Handle crypto with X: prefix
    if symbol.startswith('X:'):
        crypto_code = symbol[2:]  # Remove X: prefix
        # Map common crypto codes (expand as needed)
        if crypto_code == 'BTCUSD':
            return 'BTC-USD'
        elif crypto_code == 'ETHUSD':
            return 'ETH-USD'
        elif crypto_code == 'SOLUSD':
            return 'SOL-USD'

    # Handle class shares: BRK.B -> BRK-B
    if '.' in symbol:
        parts = symbol.split('.')
        if len(parts) == 2 and len(parts[1]) == 1:
            canonical = f"{parts[0]}-{parts[1]}"
            if is_canonical(canonical):
                return canonical

    # Regular stock
    if is_canonical(symbol):
        return symbol

    raise ValueError(f"Cannot map Polygon symbol to canonical: {symbol}")


def _from_tiingo(symbol: str) -> str:
    """
    Tiingo -> Canonical (PLACEHOLDER - implement when Tiingo is integrated)

    Tiingo Format:
        - Stocks: "AAPL" (uppercase)
        - Crypto: "btcusd" (lowercase, no separator)

    Canonical Format:
        - Stocks: "AAPL"
        - Crypto: "BTC-USD"
    """
    # Tiingo uses lowercase for crypto
    lower_symbol = symbol.lower().strip()

    # Map common crypto pairs (expand as needed)
    crypto_map = {
        'btcusd': 'BTC-USD',
        'ethusd': 'ETH-USD',
        'solusd': 'SOL-USD',
        'adausd': 'ADA-USD',
        'avaxusd': 'AVAX-USD',
        'dogeusd': 'DOGE-USD',
        'dotusd': 'DOT-USD',
        'linkusd': 'LINK-USD',
        'ltcusd': 'LTC-USD',
        'xrpusd': 'XRP-USD'
    }

    if lower_symbol in crypto_map:
        return crypto_map[lower_symbol]

    # Try uppercase stock symbol
    upper_symbol = symbol.upper().strip()
    if is_canonical(upper_symbol):
        return upper_symbol

    raise ValueError(f"Cannot map Tiingo symbol to canonical: {symbol}")


def _from_kraken(symbol: str) -> str:
    """
    Kraken -> Canonical (PLACEHOLDER - implement when Kraken is integrated)

    Kraken Format:
        - Proprietary crypto codes: "XXBTZUSD" (Bitcoin)

    Canonical Format:
        - "BTC-USD"
    """
    symbol = symbol.upper().strip()

    # Map Kraken proprietary codes (expand as needed)
    kraken_map = {
        'XXBTZUSD': 'BTC-USD',
        'XETHZUSD': 'ETH-USD',
        'SOLUSD': 'SOL-USD'
    }

    if symbol in kraken_map:
        return kraken_map[symbol]

    raise ValueError(f"Cannot map Kraken symbol to canonical: {symbol}")


# ============================================================================
# CANONICAL-TO-PROVIDER REVERSE MAPPING (for API calls)
# ============================================================================

def from_canonical(symbol: str, provider: str) -> str:
    """
    Convert canonical symbol to provider-specific format (for API calls).

    Args:
        symbol: Canonical symbol (Yahoo Finance format)
        provider: Target provider name

    Returns:
        Provider-specific symbol format

    Examples:
        >>> from_canonical("BRK-B", provider="ibkr")
        'BRK B'
        >>> from_canonical("BTC-USD", provider="ibkr")
        'BTC/USD'
    """
    provider = provider.lower()

    if not is_canonical(symbol):
        raise ValueError(f"Input must be canonical symbol: {symbol}")

    if provider == "yahoo":
        return symbol  # Already canonical

    elif provider == "ibkr":
        # BRK-B -> BRK B, BTC-USD -> BTC/USD
        if symbol in CANONICAL_CLASS_SHARES:
            return symbol.replace('-', ' ')
        elif symbol in CANONICAL_CRYPTO_PAIRS:
            return symbol.replace('-', '/')
        return symbol

    elif provider == "schwab":
        # BRK-B -> BRK.B
        if symbol in CANONICAL_CLASS_SHARES:
            return symbol.replace('-', '.')
        return symbol

    elif provider == "polygon":
        # BRK-B -> BRK.B, BTC-USD -> X:BTCUSD
        if symbol in CANONICAL_CRYPTO_PAIRS:
            crypto_code = symbol.replace('-', '')
            return f"X:{crypto_code}"
        elif symbol in CANONICAL_CLASS_SHARES:
            return symbol.replace('-', '.')
        return symbol

    elif provider == "tiingo":
        # BTC-USD -> btcusd (lowercase crypto)
        if symbol in CANONICAL_CRYPTO_PAIRS:
            return symbol.replace('-', '').lower()
        return symbol

    elif provider == "kraken":
        # BTC-USD -> XXBTZUSD (proprietary codes)
        kraken_map = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD'
        }
        return kraken_map.get(symbol, symbol)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# VALIDATION AND AUTO-CORRECTION
# ============================================================================

def is_canonical(symbol: str) -> bool:
    """
    Check if symbol is in canonical format.

    Canonical format rules:
        - Uppercase letters only
        - Optional hyphen (-) for class shares or crypto pairs
        - No spaces, dots, slashes, or lowercase
        - Pattern: ^[A-Z]+(-[A-Z]+)?$

    Args:
        symbol: Symbol to validate

    Returns:
        True if canonical, False otherwise

    Examples:
        >>> is_canonical("AAPL")
        True
        >>> is_canonical("BRK-B")
        True
        >>> is_canonical("BTC-USD")
        True
        >>> is_canonical("BRK.B")
        False
        >>> is_canonical("brk-b")
        False
    """
    if not symbol or not isinstance(symbol, str):
        return False

    return bool(CANONICAL_PATTERN.match(symbol))


def auto_correct(symbol: str, provider: Optional[str] = None) -> Tuple[str, bool, str]:
    """
    Attempt to auto-correct symbol to canonical format.

    This function is forgiving - used at ingestion edges to auto-fix
    common format issues before they enter the system.

    Args:
        symbol: Symbol in any format
        provider: Optional provider hint (helps disambiguation)

    Returns:
        Tuple of (canonical_symbol, was_corrected, correction_message)

    Examples:
        >>> auto_correct("brk-b")
        ('BRK-B', True, 'AUTO_CORRECTED: Case normalized')

        >>> auto_correct("BRK.B", provider="schwab")
        ('BRK-B', True, 'AUTO_CORRECTED: BRK.B -> BRK-B (Schwab class share)')

        >>> auto_correct("AAPL")
        ('AAPL', False, 'Already canonical')
    """
    original = symbol

    # Already canonical - no correction needed
    if is_canonical(symbol):
        return (symbol, False, "Already canonical")

    # Try uppercase normalization first
    symbol_upper = symbol.upper().strip()
    if is_canonical(symbol_upper):
        return (symbol_upper, True, "AUTO_CORRECTED: Case normalized")

    # If provider specified, try provider-specific mapping
    if provider:
        try:
            canonical = to_canonical(original, provider=provider)
            msg = f"AUTO_CORRECTED: {original} -> {canonical} ({provider} format)"
            return (canonical, True, msg)
        except ValueError:
            pass  # Fall through to generic auto-correct

    # Generic auto-correction attempts

    # Try replacing dot with hyphen (Schwab class shares)
    if '.' in symbol:
        candidate = symbol.replace('.', '-').upper()
        if is_canonical(candidate):
            return (candidate, True, f"AUTO_CORRECTED: {original} -> {candidate} (dot -> hyphen)")

    # Try replacing space with hyphen (IBKR class shares)
    if ' ' in symbol:
        candidate = symbol.replace(' ', '-').upper()
        if is_canonical(candidate):
            return (candidate, True, f"AUTO_CORRECTED: {original} -> {candidate} (space -> hyphen)")

    # Try replacing slash with hyphen (IBKR crypto)
    if '/' in symbol:
        candidate = symbol.replace('/', '-').upper()
        if is_canonical(candidate):
            return (candidate, True, f"AUTO_CORRECTED: {original} -> {candidate} (slash -> hyphen)")

    # Unable to auto-correct
    return (original, False, f"FAILED: Cannot auto-correct '{original}' to canonical format")


def validate_and_normalize(symbol: str, provider: str = "yahoo",
                           strict: bool = False) -> str:
    """
    Validate and normalize symbol for database writes.

    This is the primary function called before any DB insert/update.

    Args:
        symbol: Symbol in any format
        provider: Data provider (for correct translation)
        strict: If True, raise error on non-canonical. If False, auto-correct.

    Returns:
        Canonical symbol (safe to write to DB)

    Raises:
        ValueError: If strict=True and symbol is non-canonical or unmappable

    Examples:
        # Forgiving mode (ingestion edges)
        >>> validate_and_normalize("BRK.B", provider="schwab", strict=False)
        'BRK-B'  # Auto-corrected with warning logged

        # Strict mode (core systems)
        >>> validate_and_normalize("BRK.B", provider="yahoo", strict=True)
        ValueError: Non-canonical symbol: BRK.B
    """
    # If already canonical, return immediately
    if is_canonical(symbol):
        return symbol

    # Strict mode - no auto-correction allowed
    if strict:
        raise ValueError(f"Non-canonical symbol rejected (strict mode): {symbol}")

    # Forgiving mode - attempt provider-specific translation
    try:
        canonical = to_canonical(symbol, provider=provider)
        # Note: Caller should log this correction to data_quality_log
        return canonical
    except ValueError as e:
        # Unable to map - raise error even in forgiving mode
        raise ValueError(f"Cannot normalize symbol '{symbol}' from {provider}: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_provider_formats(symbol: str) -> dict:
    """
    Get all provider-specific formats for a canonical symbol.

    Args:
        symbol: Canonical symbol

    Returns:
        Dict mapping provider names to their symbol formats

    Example:
        >>> get_provider_formats("BRK-B")
        {
            'yahoo': 'BRK-B',
            'ibkr': 'BRK B',
            'schwab': 'BRK.B',
            'polygon': 'BRK.B',
            'tiingo': 'BRK-B',
            'kraken': 'BRK-B'
        }
    """
    if not is_canonical(symbol):
        raise ValueError(f"Input must be canonical: {symbol}")

    formats = {}
    for provider in Provider:
        try:
            formats[provider.value] = from_canonical(symbol, provider.value)
        except (ValueError, NotImplementedError):
            formats[provider.value] = symbol  # Fallback to canonical

    return formats


if __name__ == '__main__':
    # Test canonical validation
    print("=" * 80)
    print("CANONICAL SYMBOL NORMALIZER - TEST SUITE")
    print("=" * 80)
    print()

    print("1. CANONICAL VALIDATION TESTS:")
    print("-" * 80)
    test_cases = [
        ("AAPL", True),
        ("BRK-B", True),
        ("BTC-USD", True),
        ("GOOGL", True),
        ("BRK.B", False),
        ("brk-b", False),
        ("BRK B", False),
        ("BTC/USD", False),
    ]

    for symbol, expected in test_cases:
        result = is_canonical(symbol)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: is_canonical('{symbol}') = {result} (expected {expected})")

    print()
    print("2. PROVIDER MAPPING TESTS:")
    print("-" * 80)

    # Schwab mappings
    print("  Schwab -> Canonical:")
    print(f"    BRK.B  -> {to_canonical('BRK.B', provider='schwab')}")
    print(f"    AAPL   -> {to_canonical('AAPL', provider='schwab')}")

    # IBKR mappings
    print("  IBKR -> Canonical:")
    print(f"    BRK B    -> {to_canonical('BRK B', provider='ibkr')}")
    print(f"    BTC/USD  -> {to_canonical('BTC/USD', provider='ibkr')}")
    print(f"    AAPL     -> {to_canonical('AAPL', provider='ibkr')}")

    # Tiingo mappings
    print("  Tiingo -> Canonical:")
    print(f"    btcusd -> {to_canonical('btcusd', provider='tiingo')}")
    print(f"    AAPL   -> {to_canonical('AAPL', provider='tiingo')}")

    print()
    print("3. AUTO-CORRECTION TESTS:")
    print("-" * 80)

    correction_tests = [
        "brk-b",      # Case normalization
        "BRK.B",      # Dot to hyphen
        "BRK B",      # Space to hyphen
        "BTC/USD",    # Slash to hyphen
        "AAPL",       # Already canonical
    ]

    for test_symbol in correction_tests:
        canonical, corrected, msg = auto_correct(test_symbol)
        print(f"  {test_symbol:12s} -> {canonical:12s} | {msg}")

    print()
    print("4. REVERSE MAPPING TESTS (Canonical -> Provider):")
    print("-" * 80)

    canonical_symbols = ["BRK-B", "BTC-USD", "AAPL"]

    for symbol in canonical_symbols:
        print(f"\n  {symbol}:")
        formats = get_provider_formats(symbol)
        for provider, format_str in formats.items():
            print(f"    {provider:12s} -> {format_str}")

    print()
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
