#!/usr/bin/env python3
"""Run exactly what CI runs, and fail the way CI fails.

    python scripts/check.py          # lint, format, tests
    python scripts/check.py --fix    # fix what can be fixed first

CI lints `src tests`, checks formatting of `src tests docs/conf.py`, and runs
pytest. Running those by hand in three commands is easy to get subtly wrong —
formatting only `src` after editing a test, say — and the miss is not visible
until a red run minutes later. This runs the same three, stops at the first
failure, and returns a real exit code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (what it is, the command) — mirroring .github/workflows/ci.yml.
CHECKS: tuple[tuple[str, list[str]], ...] = (
    ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests"]),
    ("black", [sys.executable, "-m", "black", "--check", "src", "tests", "docs/conf.py"]),
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
)

FIXES: tuple[tuple[str, list[str]], ...] = (
    (
        "ruff --fix",
        [sys.executable, "-m", "ruff", "check", "--fix", "src", "tests", "scripts", "examples"],
    ),
    (
        "black",
        [sys.executable, "-m", "black", "src", "tests", "scripts", "examples", "docs/conf.py"],
    ),
)


def run(label: str, command: list[str]) -> int:
    print(f"\n--- {label} " + "-" * (60 - len(label)))
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fix", action="store_true", help="format and autofix first")
    args = parser.parse_args()

    if args.fix:
        for label, command in FIXES:
            run(label, command)

    for label, command in CHECKS:
        code = run(label, command)
        if code != 0:
            print(f"\n{label} failed — CI would fail here too.", file=sys.stderr)
            return code

    print("\n전부 통과. CI도 같은 것을 봅니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
