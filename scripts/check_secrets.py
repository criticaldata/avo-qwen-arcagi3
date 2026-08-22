#!/usr/bin/env python3
"""Fail if anything that looks like an API key is present in tracked files.

Used by CI and the pre-commit hook. Deliberately dependency-free.
"""

from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [
    # Cerebras keys
    re.compile(r"\bcsk-[A-Za-z0-9]{16,}"),
    # OpenAI-style keys
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    # Anthropic
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"),
    # Generic assignments of key-like names (compound names like GITHUB_TOKEN or
    # AGENTOPS_API_KEY included) to non-placeholder values
    re.compile(
        r"(?i)[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD)\b\s*[=:]\s*"
        r"['\"]?(?!\s*$)(?!\$\{?)(?!<)(?!your[-_])(?!changeme)(?!dummy)(?!test-key)(?!secrets\.)"
        r"[A-Za-z0-9_\-]{16,}"
    ),
    # UUID-shaped values assigned to api-key-ish names/headers (ARC keys are uuids)
    re.compile(
        r"(?i)(api[_-]?key|x-api-key)[\"']?\s*[=:]\s*[\"']?"
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ),
    # GitHub tokens (classic + fine-grained)
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    # AWS access keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]

SKIP_SUFFIXES = (".png", ".jpg", ".gif", ".pdf", ".lock", ".ipynb")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in out.stdout.splitlines() if f and not f.endswith(SKIP_SUFFIXES)]


def main() -> int:
    hits = []
    for path in tracked_files():
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    for pat in PATTERNS:
                        if pat.search(line):
                            hits.append(f"{path}:{lineno}: matches {pat.pattern[:40]}...")
        except OSError:
            continue
    # this file legitimately contains the patterns themselves
    hits = [h for h in hits if not h.startswith("scripts/check_secrets.py:")]
    if hits:
        print("Potential secrets found — refusing to proceed:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    print(f"secret scan clean ({len(tracked_files())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
