#!/usr/bin/env python

"""
Create a fresh turbomode.db using the current database_schema.py.

This script lives in:
    C:\StockApp\backend\turbomode\database_rebuild

It simply instantiates TurboModeDB(), which:
    - Creates backend/data/turbomode.db if missing
    - Applies the REAL-DB-derived schema
    - Ensures all tables and indexes exist
"""

import os
import sys
from pathlib import Path

# Resolve project root (C:\StockApp)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import TurboModeDB from the canonical schema file
from backend.turbomode.database_schema import TurboModeDB


def main():
    print("=" * 80)
    print("Creating fresh turbomode.db using TurboModeDB()")
    print("=" * 80)

    # Explicit db_path required because the auto-generated schema removed the default
    db_path = os.path.join(project_root, "backend", "data", "turbomode.db")

    db = TurboModeDB(db_path=db_path)

    print("[OK] turbomode.db created at:")
    print(f"     {db.db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()