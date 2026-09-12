#!/usr/bin/env python3
"""Leave a text report as a comment on the commit that produced it.

    python scripts/post_report.py report.txt "Verify capabilities"

Job logs take their time to publish and a job summary cannot be read back
through the API, so a report that only lives in either is a report nobody can
fetch when they need it. A commit comment is readable the moment it exists, by
anyone who can read the repository.

Needs ``GITHUB_TOKEN`` (the workflow's own token, with ``contents: write``),
``GITHUB_REPOSITORY`` and ``GITHUB_SHA`` — all three are set for it in Actions.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: GitHub rejects very large comment bodies; keep the tail, which is where the
#: totals and the failures are.
LIMIT = 60_000


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: post_report.py <file> [title]", file=sys.stderr)
        return 2

    report = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else "Report"
    if not report.is_file():
        print(f"{report} does not exist — nothing to post.", file=sys.stderr)
        return 0

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (token and repo and sha):
        print("not running in Actions (no token/repo/sha) — skipping the comment.")
        return 0

    text = report.read_text(encoding="utf-8", errors="replace")
    if len(text) > LIMIT:
        text = "... (앞부분 생략)\n" + text[-LIMIT:]
    body = f"### {title} — `{sha[:7]}`\n\n```\n{text}\n```\n"

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/commits/{sha}/comments",
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ovkit-verify",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            print(f"report posted to the commit ({response.status}).")
    except urllib.error.HTTPError as exc:
        # Never fail the job over the notification: the report itself already
        # went to the log.
        print(f"could not post the report: HTTP {exc.code} {exc.reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
