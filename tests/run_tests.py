#!/usr/bin/env python3
"""Fallback runner (no pytest needed): python tests/run_tests.py"""
import importlib
import pathlib
import sys
import traceback

root = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(root), str(root.parent)]
passed = failed = 0
for path in sorted(root.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        try:
            getattr(module, name)()
            passed += 1
            print(f"PASS {path.stem}::{name}")
        except Exception:
            failed += 1
            print(f"FAIL {path.stem}::{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
