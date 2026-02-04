"""
Level-2 Deep Scheduler Audit Script

Goal:
Go beyond basic file/entrypoint checks and aggressively validate:

- Actual imports and function calls in unified_scheduler.py
- APScheduler job registration and target functions
- Working directory and path usage
- Parameters passed into core 14-day tasks
- Hidden legacy references (including commented-out code)
- Relative vs absolute path risks
- Any accidental use of unified pipeline script

This script is intentionally paranoid.
"""

import os
import re
import ast
from typing import Dict, List, Tuple

SCHEDULER_PATH = "backend/unified_scheduler.py"
CONFIG_PATH = "backend/scheduler_config.json"

# Expected mapping from scheduler function -> fully qualified target
EXPECTED_CALLS = {
    "run_ingestion": {
        "must_contain": ["ingestion", "master_market_data"],
        "must_not_contain": ["backtest_generator", "label_1d_5pct"]
    },
    "run_orchestrator": {
        "must_contain": ["train_all_sectors_optimized_orchestrator"],
        "must_not_contain": ["train_1d", "train_percent", "advanced_ml"]
    },
    "run_overnight_scanner": {
        "must_contain": ["overnight_scanner", "ProductionScanner"],
        "must_not_contain": ["horizon='1d'", "horizon=\"1d\"", "old_scanner"]
    },
    "run_backtest_generator": {
        "must_contain": ["turbomode_backtest", "TurboModeBacktest"],
        "must_not_contain": ["backtest_generator", "label_1d_5pct", "percent_threshold"]
    },
    "run_adaptive_ranking": {
        "must_contain": ["adaptive_stock_ranker"],
        "must_not_contain": ["old_ranking", "percent_threshold"]
    },
    "run_drift_monitor": {
        "must_contain": ["drift"],
        "must_not_contain": ["backtest_generator", "label_1d_5pct"]
    },
    "run_weekly_maintenance": {
        "must_contain": ["maintenance", "cleanup"],
        "must_not_contain": ["backtest_generator", "label_1d_5pct"]
    }
}

LEGACY_PATTERNS_CODE = [
    r"backtest_generator",
    r"label_1d_5pct",
    r"label_2d",
    r"horizon\s*=\s*['\"]1d['\"]",
    r"trained_5pct",
    r"trained_10pct",
    r"advanced_ml",
    r"percent_threshold",
    r"train_1d",
    r"train_percent",
    r"scanner_v1",
    r"old_scanner",
    r"legacy_",
    r"deprecated_",
    r"run_full_production_pipeline_14d"
]

RELATIVE_PATH_PATTERNS = [
    r"subprocess\.run\(",
    r"os\.system\(",
    r"os\.chdir\(",
]

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def find_function_blocks(source: str) -> Dict[str, str]:
    """
    Roughly extract function bodies by name using AST.
    """
    tree = ast.parse(source)
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            # naive end: next function or EOF
            functions[node.name] = node
    # Map to source slices
    lines = source.splitlines()
    fn_blocks: Dict[str, str] = {}
    fn_names = list(functions.keys())
    for i, name in enumerate(fn_names):
        node = functions[name]
        start = node.lineno - 1
        if i + 1 < len(fn_names):
            next_node = functions[fn_names[i + 1]]
            end = next_node.lineno - 1
        else:
            end = len(lines)
        fn_blocks[name] = "\n".join(lines[start:end])
    return fn_blocks

def scan_legacy_patterns(source: str) -> List[str]:
    hits = []
    for pattern in LEGACY_PATTERNS_CODE:
        if re.search(pattern, source):
            hits.append(pattern)
    return hits

def scan_relative_path_risks(source: str) -> List[str]:
    hits = []
    for pattern in RELATIVE_PATH_PATTERNS:
        if re.search(pattern, source):
            hits.append(pattern)
    return hits

def check_expected_calls(fn_blocks: Dict[str, str]) -> List[str]:
    issues = []
    for fn_name, expectations in EXPECTED_CALLS.items():
        block = fn_blocks.get(fn_name)
        if block is None:
            issues.append(f"Function {fn_name} not found in unified_scheduler.py")
            continue

        must = expectations.get("must_contain", [])
        must_not = expectations.get("must_not_contain", [])

        for token in must:
            if token not in block:
                issues.append(f"{fn_name}: expected to contain '{token}' but it was missing")

        for token in must_not:
            if token in block:
                issues.append(f"{fn_name}: must NOT contain '{token}' but it was found")
    return issues

def extract_scheduler_jobs(source: str) -> List[Tuple[int, str]]:
    """
    Look for add_job calls to see what functions are actually scheduled.
    """
    jobs = []
    for i, line in enumerate(source.splitlines(), start=1):
        if "add_job(" in line:
            jobs.append((i, line.strip()))
    return jobs

def main():
    print_header("LEVEL-2 DEEP SCHEDULER AUDIT")

    if not os.path.exists(SCHEDULER_PATH):
        print(f"Missing unified scheduler file: {SCHEDULER_PATH}")
        return

    code = read_file(SCHEDULER_PATH)
    fn_blocks = find_function_blocks(code)

    # 1. Check expected calls inside each run_* function
    print_header("FUNCTION CALL & IMPORT VALIDATION")
    call_issues = check_expected_calls(fn_blocks)
    if call_issues:
        for issue in call_issues:
            print(issue)
    else:
        print("All run_* functions contain expected imports/calls and no obvious legacy tokens")

    # 2. Scan entire file for legacy contamination (including commented-out code)
    print_header("GLOBAL LEGACY CONTAMINATION SCAN")
    legacy_hits = scan_legacy_patterns(code)
    if legacy_hits:
        print("Legacy patterns found:")
        for pattern in legacy_hits:
            print(f"   - {pattern}")
    else:
        print("No legacy patterns detected in unified_scheduler.py")

    # 3. Scan for relative path / subprocess / chdir risks
    print_header("RELATIVE PATH & SUBPROCESS RISK SCAN")
    rel_hits = scan_relative_path_risks(code)
    if rel_hits:
        print("Potential relative path / subprocess risks found:")
        for pattern in rel_hits:
            print(f"   - Pattern: {pattern}")
        print("  Review these manually to ensure absolute paths / correct cwd are enforced.")
    else:
        print("No obvious subprocess/os.chdir/os.system usage detected")

    # 4. Inspect APScheduler job registration
    print_header("APSCHEDULER JOB REGISTRATION SCAN")
    jobs = extract_scheduler_jobs(code)
    if not jobs:
        print("No add_job(...) calls found. If scheduler uses another registration mechanism, review manually.")
    else:
        for lineno, line in jobs:
            print(f"Line {lineno}: {line}")
        print("\nReview that each add_job targets the correct run_* function and cron config matches scheduler_config.json.")

    print_header("SUMMARY")
    if not call_issues and not legacy_hits:
        print("LEVEL-2 AUDIT RESULT: PASS (no structural or legacy issues detected)")
    else:
        print("LEVEL-2 AUDIT RESULT: ATTENTION REQUIRED (see issues above)")

if __name__ == "__main__":
    main()
