"""
TurboMode Schema Guardrail
Enforces canonical schema and prevents contamination of turbomode.db

This module validates that turbomode.db contains ONLY the canonical TurboMode tables
and raises hard errors if any forbidden or unknown tables are detected.

Author: TurboMode Core Engine
Date: 2026-01-06
"""

import sqlite3
import os
import sys
from typing import List, Dict, Set, Tuple
import fnmatch


# Canonical TurboMode Schema (IMMUTABLE)
CANONICAL_ALLOWED_TABLES = {
    'active_signals',
    'config_audit_log',
    'feature_store',
    'model_metadata',
    'price_data',
    'sector_stats',
    'signal_history',
    'trades',
    'training_runs'
}

# Forbidden table patterns (IMMUTABLE)
FORBIDDEN_TABLE_PATTERNS = [
    'drift_monitoring',
    'drift_logs',
    'advanced_ml_*',
    'ml_drift_*',
    'prediction_drift_*',
    'training_samples_old',
    'positions',
    'outcomes',
    'signals_old',
    'backtest_results',
    'temp_*',
    'cache_*',
    'debug_*',
    'sandbox_*',
    'test_*'
]


class SchemaContaminationError(Exception):
    """Raised when turbomode.db contains forbidden or unknown tables"""
    pass


def get_existing_tables(db_path: str) -> Set[str]:
    """
    Get list of all tables in database

    Args:
        db_path: Path to turbomode.db

    Returns:
        Set of table names
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    return tables


def check_forbidden_tables(tables: Set[str]) -> List[str]:
    """
    Check if any tables match forbidden patterns

    Args:
        tables: Set of table names

    Returns:
        List of forbidden tables found
    """
    forbidden_found = []

    for table in tables:
        for pattern in FORBIDDEN_TABLE_PATTERNS:
            if fnmatch.fnmatch(table, pattern):
                forbidden_found.append(table)
                break

    return sorted(forbidden_found)


def check_unknown_tables(tables: Set[str]) -> List[str]:
    """
    Check if any tables are not in the canonical allowed list

    Args:
        tables: Set of table names

    Returns:
        List of unknown tables found
    """
    unknown = tables - CANONICAL_ALLOWED_TABLES
    return sorted(unknown)


def validate_schema(db_path: str, strict: bool = True) -> Dict[str, any]:
    """
    Validate turbomode.db schema against canonical schema

    Args:
        db_path: Path to turbomode.db
        strict: If True, raise error on violation. If False, only warn.

    Returns:
        Validation report dictionary

    Raises:
        SchemaContaminationError: If schema violations detected (strict mode only)
    """
    if not os.path.exists(db_path):
        # Database doesn't exist yet - this is OK
        return {
            'status': 'OK',
            'message': 'Database does not exist yet (will be created)',
            'violations': []
        }

    # Get existing tables
    existing_tables = get_existing_tables(db_path)

    # Check for violations
    forbidden_tables = check_forbidden_tables(existing_tables)
    unknown_tables = check_unknown_tables(existing_tables)

    # Remove forbidden tables from unknown list (avoid double-counting)
    unknown_tables = [t for t in unknown_tables if t not in forbidden_tables]

    # Build violation report
    violations = []

    if forbidden_tables:
        violations.append({
            'type': 'FORBIDDEN_TABLES',
            'severity': 'CRITICAL',
            'tables': forbidden_tables,
            'message': f'Found {len(forbidden_tables)} FORBIDDEN tables that match contamination patterns'
        })

    if unknown_tables:
        violations.append({
            'type': 'UNKNOWN_TABLES',
            'severity': 'WARNING',
            'tables': unknown_tables,
            'message': f'Found {len(unknown_tables)} tables not in canonical schema'
        })

    # Generate report
    if violations:
        report = {
            'status': 'VIOLATION',
            'db_path': db_path,
            'existing_tables': sorted(existing_tables),
            'allowed_tables': sorted(CANONICAL_ALLOWED_TABLES),
            'violations': violations,
            'total_violations': len(forbidden_tables) + len(unknown_tables)
        }

        # Print violation report
        print_violation_report(report)

        # Raise error if strict mode
        if strict:
            error_msg = f"SCHEMA CONTAMINATION DETECTED: {len(forbidden_tables)} forbidden + {len(unknown_tables)} unknown tables"
            raise SchemaContaminationError(error_msg)

        return report

    else:
        # No violations - schema is clean
        return {
            'status': 'OK',
            'db_path': db_path,
            'existing_tables': sorted(existing_tables),
            'allowed_tables': sorted(CANONICAL_ALLOWED_TABLES),
            'violations': [],
            'message': 'Schema validated - all tables are canonical'
        }


def print_violation_report(report: Dict[str, any]) -> None:
    """
    Print formatted violation report

    Args:
        report: Validation report dictionary
    """
    print()
    print("=" * 80)
    print("TURBOMODE SCHEMA GUARDRAIL - VIOLATION DETECTED")
    print("=" * 80)
    print(f"Database: {report['db_path']}")
    print(f"Status: {report['status']}")
    print()

    for violation in report['violations']:
        print(f"[{violation['severity']}] {violation['type']}")
        print(f"  {violation['message']}")
        print(f"  Tables:")
        for table in violation['tables']:
            print(f"    - {table}")
        print()

    print("Canonical Allowed Tables:")
    for table in report['allowed_tables']:
        status = "OK" if table in report['existing_tables'] else "MISSING"
        print(f"  [{status}] {table}")

    print()
    print("=" * 80)
    print("ACTION REQUIRED:")
    print("  1. Identify which script created the forbidden/unknown tables")
    print("  2. Remove all AdvancedML imports from that script")
    print("  3. Delete contaminated tables or rebuild turbomode.db")
    print("  4. Re-run with clean schema")
    print("=" * 80)
    print()


def cleanup_forbidden_tables(db_path: str, dry_run: bool = True) -> Dict[str, any]:
    """
    Remove forbidden tables from turbomode.db

    Args:
        db_path: Path to turbomode.db
        dry_run: If True, only report what would be deleted. If False, actually delete.

    Returns:
        Cleanup report dictionary
    """
    existing_tables = get_existing_tables(db_path)
    forbidden_tables = check_forbidden_tables(existing_tables)

    if not forbidden_tables:
        return {
            'status': 'CLEAN',
            'message': 'No forbidden tables to clean up',
            'deleted': []
        }

    print()
    print("=" * 80)
    print("TURBOMODE SCHEMA CLEANUP")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETION'}")
    print()
    print(f"Forbidden tables to delete: {len(forbidden_tables)}")
    for table in forbidden_tables:
        print(f"  - {table}")
    print()

    if dry_run:
        print("[DRY RUN] No tables deleted. Run with dry_run=False to delete.")
        print("=" * 80)
        return {
            'status': 'DRY_RUN',
            'would_delete': forbidden_tables,
            'deleted': []
        }

    # Actually delete forbidden tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    deleted = []
    for table in forbidden_tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            deleted.append(table)
            print(f"[DELETED] {table}")
        except Exception as e:
            print(f"[ERROR] Failed to delete {table}: {e}")

    conn.commit()
    conn.close()

    print()
    print(f"[OK] Deleted {len(deleted)} forbidden tables")
    print("=" * 80)

    return {
        'status': 'CLEANED',
        'deleted': deleted,
        'message': f'Successfully deleted {len(deleted)} forbidden tables'
    }


def clean_schema(db_path: str, auto_confirm: bool = False) -> Dict[str, any]:
    """
    Automatically clean schema by removing ALL non-canonical tables

    This removes:
    - Forbidden tables (matching contamination patterns)
    - Unknown tables (not in canonical list)

    Args:
        db_path: Path to turbomode.db
        auto_confirm: If True, skip confirmation prompt

    Returns:
        Cleanup report dictionary
    """
    if not os.path.exists(db_path):
        return {
            'status': 'OK',
            'message': 'Database does not exist (nothing to clean)',
            'deleted': []
        }

    existing_tables = get_existing_tables(db_path)
    forbidden_tables = check_forbidden_tables(existing_tables)
    unknown_tables = check_unknown_tables(existing_tables)

    # Remove forbidden from unknown (avoid double-counting)
    unknown_tables = [t for t in unknown_tables if t not in forbidden_tables]

    all_to_delete = forbidden_tables + unknown_tables

    if not all_to_delete:
        return {
            'status': 'CLEAN',
            'message': 'Schema is already clean - no tables to delete',
            'deleted': []
        }

    print()
    print("=" * 80)
    print("TURBOMODE AUTO-CLEAN SCHEMA")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()
    print(f"Tables to DELETE: {len(all_to_delete)}")
    if forbidden_tables:
        print(f"  Forbidden: {len(forbidden_tables)}")
        for table in forbidden_tables:
            print(f"    - {table}")
    if unknown_tables:
        print(f"  Unknown: {len(unknown_tables)}")
        for table in unknown_tables:
            print(f"    - {table}")
    print()

    if not auto_confirm:
        response = input("Proceed with deletion? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("[ABORT] User cancelled cleanup")
            return {
                'status': 'CANCELLED',
                'message': 'User cancelled cleanup',
                'deleted': []
            }

    # Delete all non-canonical tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    deleted = []
    for table in all_to_delete:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            deleted.append(table)
            print(f"[DELETED] {table}")
        except Exception as e:
            print(f"[ERROR] Failed to delete {table}: {e}")

    conn.commit()
    conn.close()

    print()
    print(f"[OK] Deleted {len(deleted)} non-canonical tables")
    print("=" * 80)

    return {
        'status': 'CLEANED',
        'deleted': deleted,
        'message': f'Successfully deleted {len(deleted)} non-canonical tables'
    }


def restore_canonical_schema(db_path: str) -> Dict[str, any]:
    """
    Ensure all canonical tables exist with correct structure

    This creates any missing canonical tables but does NOT drop existing data.

    Args:
        db_path: Path to turbomode.db

    Returns:
        Restoration report dictionary
    """
    from turbomode.database_schema import TurboModeDB

    print()
    print("=" * 80)
    print("TURBOMODE RESTORE CANONICAL SCHEMA")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    # Initialize TurboMode schema (creates missing tables only)
    turbomode_db = TurboModeDB(db_path)

    existing_tables = get_existing_tables(db_path)
    missing_tables = CANONICAL_ALLOWED_TABLES - existing_tables
    present_tables = CANONICAL_ALLOWED_TABLES & existing_tables

    print(f"Canonical tables present: {len(present_tables)}")
    for table in sorted(present_tables):
        print(f"  [OK] {table}")

    if missing_tables:
        print(f"\nCanonical tables created: {len(missing_tables)}")
        for table in sorted(missing_tables):
            print(f"  [NEW] {table}")
    else:
        print("\nAll canonical tables already exist")

    print("=" * 80)

    return {
        'status': 'RESTORED',
        'present_tables': sorted(present_tables),
        'created_tables': sorted(missing_tables),
        'message': f'Schema restored: {len(missing_tables)} tables created, {len(present_tables)} already existed'
    }


def run_guardrail(db_path: str, auto_clean: bool = True, auto_restore: bool = True) -> Dict[str, any]:
    """
    Complete guardrail workflow: validate → clean → restore → validate

    This is the main entry point for schema guardrail protection.

    Args:
        db_path: Path to turbomode.db
        auto_clean: If True, automatically clean contamination without prompting
        auto_restore: If True, automatically restore missing canonical tables

    Returns:
        Guardrail execution report

    Raises:
        SchemaContaminationError: If schema cannot be cleaned
    """
    print()
    print("=" * 80)
    print("TURBOMODE SCHEMA GUARDRAIL - COMPLETE WORKFLOW")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Auto-clean: {auto_clean}")
    print(f"Auto-restore: {auto_restore}")
    print()

    report = {
        'db_path': db_path,
        'steps': []
    }

    # STEP 1: Initial validation
    print("[STEP 1] Initial Validation")
    try:
        validation_result = validate_schema(db_path, strict=False)
        report['steps'].append({
            'step': 'initial_validation',
            'status': validation_result['status'],
            'violations': validation_result.get('violations', [])
        })

        if validation_result['status'] == 'OK':
            print("[OK] Schema is clean - no action needed")
            report['final_status'] = 'CLEAN'
            return report

        print(f"[DETECTED] {len(validation_result.get('violations', []))} violation(s)")

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        report['steps'].append({
            'step': 'initial_validation',
            'status': 'ERROR',
            'error': str(e)
        })
        raise

    # STEP 2: Clean contamination
    print("\n[STEP 2] Clean Contamination")
    try:
        clean_result = clean_schema(db_path, auto_confirm=auto_clean)
        report['steps'].append({
            'step': 'clean_schema',
            'status': clean_result['status'],
            'deleted': clean_result.get('deleted', [])
        })

        if clean_result['status'] == 'CANCELLED':
            print("[ABORT] Cleanup cancelled by user")
            report['final_status'] = 'CANCELLED'
            return report

        print(f"[OK] Cleaned {len(clean_result.get('deleted', []))} tables")

    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        report['steps'].append({
            'step': 'clean_schema',
            'status': 'ERROR',
            'error': str(e)
        })
        raise

    # STEP 3: Restore canonical schema
    if auto_restore:
        print("\n[STEP 3] Restore Canonical Schema")
        try:
            restore_result = restore_canonical_schema(db_path)
            report['steps'].append({
                'step': 'restore_schema',
                'status': restore_result['status'],
                'created_tables': restore_result.get('created_tables', [])
            })

            print(f"[OK] Restored {len(restore_result.get('created_tables', []))} tables")

        except Exception as e:
            print(f"[ERROR] Restore failed: {e}")
            report['steps'].append({
                'step': 'restore_schema',
                'status': 'ERROR',
                'error': str(e)
            })
            raise

    # STEP 4: Final validation
    print("\n[STEP 4] Final Validation")
    try:
        final_validation = validate_schema(db_path, strict=True)
        report['steps'].append({
            'step': 'final_validation',
            'status': final_validation['status']
        })

        if final_validation['status'] == 'OK':
            print("[OK] Schema validation PASSED - database is clean")
            report['final_status'] = 'CLEAN'
        else:
            print("[ERROR] Schema validation FAILED after cleanup")
            report['final_status'] = 'FAILED'
            raise SchemaContaminationError("Schema still contaminated after cleanup")

    except SchemaContaminationError:
        raise
    except Exception as e:
        print(f"[ERROR] Final validation failed: {e}")
        report['steps'].append({
            'step': 'final_validation',
            'status': 'ERROR',
            'error': str(e)
        })
        raise

    print()
    print("=" * 80)
    print("[SUCCESS] GUARDRAIL COMPLETE - DATABASE IS CLEAN")
    print("=" * 80)
    print()

    return report


if __name__ == '__main__':
    """
    CLI for schema validation and cleanup
    """
    import argparse

    parser = argparse.ArgumentParser(description='TurboMode Schema Guardrail')
    parser.add_argument('--db', type=str, default='backend/data/turbomode.db',
                        help='Path to turbomode.db')
    parser.add_argument('--validate', action='store_true',
                        help='Validate schema only (no cleanup)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up forbidden tables only')
    parser.add_argument('--run-guardrail', action='store_true',
                        help='Run complete guardrail workflow (validate → clean → restore → validate)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode (report only, no deletion)')
    parser.add_argument('--non-strict', action='store_true',
                        help='Non-strict mode (warn but do not raise error)')
    parser.add_argument('--no-auto-clean', action='store_true',
                        help='Disable automatic cleanup (prompt for confirmation)')
    parser.add_argument('--no-restore', action='store_true',
                        help='Skip schema restoration step')

    args = parser.parse_args()

    # Convert relative path to absolute
    if not os.path.isabs(args.db):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(project_root, args.db)
    else:
        db_path = args.db

    if args.run_guardrail:
        # Complete guardrail workflow (RECOMMENDED)
        try:
            result = run_guardrail(
                db_path,
                auto_clean=not args.no_auto_clean,
                auto_restore=not args.no_restore
            )
            print(f"\n[SUCCESS] Guardrail complete: {result['final_status']}")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Guardrail failed: {e}")
            sys.exit(1)

    elif args.cleanup:
        # Cleanup mode only
        result = cleanup_forbidden_tables(db_path, dry_run=args.dry_run)
        print(f"\n[RESULT] {result['message']}")
        sys.exit(0 if result['status'] in ['CLEAN', 'CLEANED'] else 1)

    elif args.validate:
        # Validate mode only
        try:
            result = validate_schema(db_path, strict=not args.non_strict)
            print(f"\n[OK] {result.get('message', 'Schema validation passed')}")
            sys.exit(0)
        except SchemaContaminationError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)

    else:
        # Default: run complete guardrail workflow
        try:
            result = run_guardrail(db_path, auto_clean=True, auto_restore=True)
            print(f"\n[SUCCESS] Guardrail complete: {result['final_status']}")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Guardrail failed: {e}")
            sys.exit(1)
