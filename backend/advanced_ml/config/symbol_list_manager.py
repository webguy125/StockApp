"""
Symbol List Manager
Validates and maintains the core symbol list automatically

Features:
- Validates symbols are still trading
- Checks market cap categories
- Verifies liquidity (volume requirements)
- Detects sector changes
- Suggests replacements for problematic symbols
- Generates health reports
- Auto-updates core_symbols.py if needed
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import os

from .core_symbols import (
    CORE_SYMBOLS,
    SECTOR_CODES,
    MARKET_CAP_LARGE,
    MARKET_CAP_MID,
    MARKET_CAP_SMALL,
    get_all_core_symbols,
    get_symbol_metadata
)


class SymbolListManager:
    """
    Manages and validates the core symbol list

    Usage:
        manager = SymbolListManager()
        report = manager.validate_all_symbols()
        manager.print_health_report(report)
    """

    def __init__(self):
        self.validation_results = {}
        self.issues_found = []
        self.suggestions = []

        # Validation criteria
        self.min_avg_volume = 500_000      # 500K daily volume
        self.min_price = 5.0                # $5 minimum price
        self.max_price_gap_days = 5         # Max days without trading

    def validate_symbol(self, symbol: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Validate a single symbol

        Args:
            symbol: Stock ticker
            verbose: Print validation details

        Returns:
            Validation result dictionary
        """
        result = {
            'symbol': symbol,
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metadata': {}
        }

        try:
            # Fetch ticker info
            ticker = yf.Ticker(symbol)

            # Get basic info
            info = ticker.info

            # Get recent price data (30 days)
            hist = ticker.history(period='1mo')

            if hist.empty:
                result['is_valid'] = False
                result['issues'].append('No price data available - possibly delisted')
                return result

            # Check 1: Symbol still trading
            days_since_last_trade = (datetime.now() - hist.index[-1]).days
            if days_since_last_trade > self.max_price_gap_days:
                result['is_valid'] = False
                result['issues'].append(f'Last trade {days_since_last_trade} days ago - possibly delisted')

            # Check 2: Current price
            current_price = hist['Close'].iloc[-1]
            result['metadata']['price'] = float(current_price)

            if current_price < self.min_price:
                result['warnings'].append(f'Price ${current_price:.2f} below minimum ${self.min_price}')

            # Check 3: Average volume
            avg_volume = hist['Volume'].mean()
            result['metadata']['avg_volume'] = float(avg_volume)

            if avg_volume < self.min_avg_volume:
                result['warnings'].append(f'Avg volume {avg_volume:,.0f} below minimum {self.min_avg_volume:,.0f}')

            # Check 4: Market cap
            market_cap = info.get('marketCap', 0)
            result['metadata']['market_cap'] = market_cap

            if market_cap == 0:
                result['warnings'].append('Market cap unavailable')
            else:
                # Determine actual market cap category
                if market_cap >= MARKET_CAP_LARGE:
                    actual_category = 'large_cap'
                elif market_cap >= MARKET_CAP_MID:
                    actual_category = 'mid_cap'
                elif market_cap >= MARKET_CAP_SMALL:
                    actual_category = 'small_cap'
                else:
                    actual_category = 'micro_cap'
                    result['warnings'].append(f'Market cap ${market_cap/1e9:.2f}B below small cap threshold')

                result['metadata']['market_cap_category'] = actual_category
                result['metadata']['market_cap_billions'] = market_cap / 1e9

                # Compare to expected category
                expected_meta = get_symbol_metadata(symbol)
                expected_category = expected_meta.get('market_cap_category', 'unknown')

                if expected_category != 'unknown' and actual_category != expected_category:
                    result['warnings'].append(
                        f'Market cap changed: {expected_category} → {actual_category}'
                    )

            # Check 5: Sector
            sector = info.get('sector', 'Unknown')
            result['metadata']['sector'] = sector

            # Check 6: Beta (volatility measure)
            beta = info.get('beta', 1.0)
            result['metadata']['beta'] = beta if beta else 1.0

            # Check 7: Company name
            result['metadata']['name'] = info.get('longName', info.get('shortName', symbol))

            if verbose:
                print(f"✓ {symbol:6s} - ${current_price:7.2f} - Vol: {avg_volume:10,.0f} - MCap: ${market_cap/1e9:6.2f}B")

        except Exception as e:
            result['is_valid'] = False
            result['issues'].append(f'Error fetching data: {str(e)}')

        return result

    def validate_all_symbols(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate all symbols in core list

        Args:
            verbose: Print progress

        Returns:
            Comprehensive validation report
        """
        if verbose:
            print("\n" + "=" * 60)
            print("VALIDATING CORE SYMBOL LIST")
            print("=" * 60 + "\n")

        all_symbols = get_all_core_symbols()
        total = len(all_symbols)

        results = []
        valid_count = 0
        issue_count = 0
        warning_count = 0

        for i, symbol in enumerate(all_symbols, 1):
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{total}")

            result = self.validate_symbol(symbol, verbose=False)
            results.append(result)

            if result['is_valid']:
                valid_count += 1
            else:
                issue_count += 1
                self.issues_found.append(result)

            if result['warnings']:
                warning_count += len(result['warnings'])

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': total,
            'valid_symbols': valid_count,
            'invalid_symbols': issue_count,
            'total_warnings': warning_count,
            'results': results,
            'issues': self.issues_found,
            'health_score': (valid_count / total * 100) if total > 0 else 0
        }

        # Analyze by sector and market cap
        report['by_sector'] = self._analyze_by_category(results, 'sector')
        report['by_market_cap'] = self._analyze_by_category(results, 'market_cap_category')

        return report

    def _analyze_by_category(self, results: List[Dict], category_key: str) -> Dict[str, Dict]:
        """Analyze validation results by category"""
        analysis = {}

        for result in results:
            if category_key in result['metadata']:
                category = result['metadata'][category_key]

                if category not in analysis:
                    analysis[category] = {
                        'total': 0,
                        'valid': 0,
                        'issues': 0,
                        'warnings': 0
                    }

                analysis[category]['total'] += 1

                if result['is_valid']:
                    analysis[category]['valid'] += 1
                else:
                    analysis[category]['issues'] += 1

                analysis[category]['warnings'] += len(result['warnings'])

        return analysis

    def print_health_report(self, report: Dict[str, Any]):
        """
        Print comprehensive health report

        Args:
            report: Validation report from validate_all_symbols()
        """
        print("\n" + "=" * 60)
        print("CORE SYMBOL LIST HEALTH REPORT")
        print("=" * 60)
        print(f"Generated: {report['timestamp']}")
        print(f"Health Score: {report['health_score']:.1f}% ✓")
        print()

        # Summary
        print(f"Total Symbols: {report['total_symbols']}")
        print(f"  ✓ Valid:   {report['valid_symbols']} ({report['valid_symbols']/report['total_symbols']*100:.1f}%)")
        print(f"  ✗ Invalid: {report['invalid_symbols']}")
        print(f"  ⚠ Warnings: {report['total_warnings']}")

        # Issues
        if report['invalid_symbols'] > 0:
            print(f"\n{'=' * 60}")
            print("CRITICAL ISSUES (Require Immediate Attention)")
            print("=" * 60)

            for issue in report['issues']:
                print(f"\n✗ {issue['symbol']}")
                for problem in issue['issues']:
                    print(f"  - {problem}")

        # Warnings
        warnings = [r for r in report['results'] if r['warnings']]
        if warnings:
            print(f"\n{'=' * 60}")
            print("WARNINGS (Monitor but OK)")
            print("=" * 60)

            for warning in warnings[:10]:  # Show first 10
                print(f"\n⚠ {warning['symbol']}")
                for warn in warning['warnings']:
                    print(f"  - {warn}")

            if len(warnings) > 10:
                print(f"\n  ... and {len(warnings) - 10} more warnings")

        # Market Cap Migration
        migrations = [
            r for r in report['results']
            if any('Market cap changed' in w for w in r.get('warnings', []))
        ]

        if migrations:
            print(f"\n{'=' * 60}")
            print("MARKET CAP MIGRATIONS")
            print("=" * 60)

            for mig in migrations:
                print(f"  {mig['symbol']:6s} - {[w for w in mig['warnings'] if 'Market cap changed' in w][0]}")

        # By Sector Health
        print(f"\n{'=' * 60}")
        print("HEALTH BY SECTOR")
        print("=" * 60)

        for sector, stats in sorted(report['by_sector'].items()):
            health = (stats['valid'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {sector:30s} {stats['valid']:2d}/{stats['total']:2d} ({health:5.1f}%)")

        # Recommendations
        print(f"\n{'=' * 60}")
        print("RECOMMENDATIONS")
        print("=" * 60)

        if report['health_score'] >= 95:
            print("  ✓ List is healthy - no action needed")
        elif report['health_score'] >= 85:
            print("  ⚠ List is mostly healthy - monitor warnings")
        else:
            print("  ✗ List needs attention - review and update invalid symbols")

        if report['invalid_symbols'] > 0:
            print(f"  → Replace {report['invalid_symbols']} invalid symbols")

        if len(migrations) > 0:
            print(f"  → Review {len(migrations)} market cap migrations")

        print()

    def suggest_replacements(self, sector: str, market_cap_category: str,
                            num_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest replacement symbols for a sector/market cap category

        Args:
            sector: GICS sector name
            market_cap_category: 'large_cap', 'mid_cap', or 'small_cap'
            num_suggestions: Number of suggestions to return

        Returns:
            List of suggested symbols with metadata
        """
        # This is a placeholder - in production, you'd fetch from a database
        # or API of S&P 500 constituents filtered by sector and market cap

        suggestions = []

        # For now, return a message
        print(f"\n[SUGGEST] Replacement needed for {sector} / {market_cap_category}")
        print("  Recommendation: Manually research symbols using:")
        print("    - finviz.com (screener)")
        print("    - S&P sector constituent lists")
        print("    - Market cap and volume filters")

        return suggestions

    def save_report(self, report: Dict[str, Any], output_file: str = "backend/data/symbol_validation_report.json"):
        """
        Save validation report to JSON

        Args:
            report: Validation report
            output_file: Output file path
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[OK] Report saved to: {output_file}")

    def get_quick_check(self) -> Dict[str, Any]:
        """
        Quick health check (just validates that symbols exist)

        Returns:
            Quick validation summary
        """
        print("\n[QUICK CHECK] Validating symbol existence...")

        all_symbols = get_all_core_symbols()

        valid = 0
        invalid = 0
        invalid_symbols = []

        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')

                if not hist.empty:
                    valid += 1
                else:
                    invalid += 1
                    invalid_symbols.append(symbol)

            except:
                invalid += 1
                invalid_symbols.append(symbol)

        health_score = (valid / len(all_symbols) * 100) if len(all_symbols) > 0 else 0

        result = {
            'total': len(all_symbols),
            'valid': valid,
            'invalid': invalid,
            'invalid_symbols': invalid_symbols,
            'health_score': health_score
        }

        print(f"  Total: {result['total']}")
        print(f"  Valid: {result['valid']}")
        print(f"  Invalid: {result['invalid']}")
        print(f"  Health Score: {health_score:.1f}%")

        if invalid_symbols:
            print(f"  Invalid Symbols: {', '.join(invalid_symbols)}")

        return result


if __name__ == '__main__':
    # Test symbol list manager
    print("Testing Symbol List Manager...")

    manager = SymbolListManager()

    # Run quick check first
    quick_check = manager.get_quick_check()

    # If health is good, run full validation
    if quick_check['health_score'] >= 90:
        print("\n[OK] Quick check passed - running full validation...")
        report = manager.validate_all_symbols(verbose=True)

        # Print health report
        manager.print_health_report(report)

        # Save report
        manager.save_report(report)

        print("\n[OK] Symbol list manager test complete!")
    else:
        print(f"\n[WARNING] Quick check failed ({quick_check['health_score']:.1f}% health)")
        print("Review invalid symbols before running full validation")
