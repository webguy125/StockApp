"""
Test Production Scanner with Fast Mode

Tests the complete integrated system:
1. Fast Mode model loading
2. Adaptive SL/TP calculation
3. Position management
4. Hysteresis and persistence logic
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.turbomode.overnight_scanner import ProductionScanner
import logging

# Set logging to DEBUG to see all details
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def test_scanner():
    """Test scanner with a few symbols from different sectors"""

    logger.info("=" * 80)
    logger.info("PRODUCTION SCANNER TEST (FAST MODE)")
    logger.info("=" * 80)

    # Initialize scanner with 1D horizon
    scanner = ProductionScanner(horizon='1d')

    # Test symbols from different sectors
    test_symbols = [
        'AAPL',  # Technology
        'JPM',   # Financials
        'JNJ',   # Healthcare
        'HD',    # Consumer Discretionary
        'XOM',   # Energy
    ]

    logger.info(f"\nTesting {len(test_symbols)} symbols...")

    results = []
    for symbol in test_symbols:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TESTING: {symbol}")
        logger.info(f"{'=' * 60}")

        signal = scanner.scan_symbol(symbol)

        if signal:
            logger.info(f"\n[SIGNAL GENERATED]")
            logger.info(f"  Symbol: {signal['symbol']}")
            logger.info(f"  Type: {signal['signal_type']}")
            logger.info(f"  Confidence: {signal['confidence']:.2%}")
            logger.info(f"  Entry: ${signal['entry_price']:.2f}")
            logger.info(f"  Stop: ${signal['stop_price']:.2f}")
            logger.info(f"  Target: ${signal['target_price']:.2f}")
            logger.info(f"  Sector: {signal['sector']}")
            logger.info(f"  ATR: ${signal['atr']:.2f}")

            # Calculate R-multiple
            stop_distance = abs(signal['entry_price'] - signal['stop_price'])
            target_distance = abs(signal['target_price'] - signal['entry_price'])
            reward_ratio = target_distance / stop_distance if stop_distance > 0 else 0

            logger.info(f"  Stop Distance: ${stop_distance:.2f} (1R)")
            logger.info(f"  Target Distance: ${target_distance:.2f} ({reward_ratio:.1f}R)")

            results.append(signal)
        else:
            logger.info(f"\n[NO SIGNAL] - Below threshold or position already exists")

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Symbols tested: {len(test_symbols)}")
    logger.info(f"Signals generated: {len(results)}")

    if results:
        buy_signals = [r for r in results if r['signal_type'] == 'BUY']
        sell_signals = [r for r in results if r['signal_type'] == 'SELL']

        logger.info(f"  BUY signals: {len(buy_signals)}")
        logger.info(f"  SELL signals: {len(sell_signals)}")

        logger.info(f"\nSignals:")
        for signal in results:
            logger.info(f"  {signal['symbol']}: {signal['signal_type']} @ {signal['confidence']:.2%}")

    # Check active positions
    logger.info(f"\n{'=' * 80}")
    logger.info("ACTIVE POSITIONS")
    logger.info(f"{'=' * 80}")

    active_positions = scanner.position_manager.get_all_active_positions()
    logger.info(f"Active positions: {len(active_positions)}")

    for symbol, pos in active_positions.items():
        logger.info(f"\n{symbol} - {pos['position'].upper()}")
        logger.info(f"  Entry: ${pos['entry_price']:.2f}")
        logger.info(f"  Stop: ${pos['stop_price']:.2f}")
        logger.info(f"  Target: ${pos['target_price']:.2f}")
        logger.info(f"  Sector: {pos['sector']}")
        logger.info(f"  Horizon: {pos['horizon']}")
        logger.info(f"  Confidence: {pos['entry_confidence']:.2%}")
        logger.info(f"  Partials: 1R={pos['partial_1R_done']}, 2R={pos['partial_2R_done']}, 3R={pos['partial_3R_done']}")

    logger.info(f"\n{'=' * 80}")
    logger.info("TEST COMPLETE")
    logger.info(f"{'=' * 80}")


if __name__ == '__main__':
    test_scanner()
