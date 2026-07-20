#!/usr/bin/env python3
"""Reject oversized Git blobs without adding repository-specific exceptions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class OversizedBlob:
    source: str
    size: int
    path: str


def _git(
    root: Path,
    *args: str,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _worktree_objects(root: Path, limit: int) -> list[OversizedBlob]:
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    failures: list[OversizedBlob] = []
    for raw_path in listing.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if size >= limit:
            failures.append(OversizedBlob("worktree", size, relative))
    return failures


def _history_objects(root: Path, base: str | None, limit: int) -> list[OversizedBlob]:
    if base is not None:
        verified = _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", check=False)
        if verified.returncode != 0:
            raise ValueError(f"base revision does not exist: {base}")
        revision = f"{base}..HEAD"
    else:
        revision = "HEAD"
    objects = _git(root, "rev-list", "--objects", revision).stdout
    if not objects.strip():
        return []
    checked = _git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)",
        input_text=objects,
    ).stdout
    failures: list[OversizedBlob] = []
    for line in checked.splitlines():
        fields = line.split(" ", 3)
        if len(fields) < 3 or fields[1] != "blob":
            continue
        size = int(fields[2])
        if size < limit:
            continue
        path = fields[3] if len(fields) == 4 and fields[3] else fields[0]
        failures.append(OversizedBlob("history", size, path))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base",
        help="scan only commits after this revision; omit to scan all history reachable from HEAD",
    )
    parser.add_argument("--limit-mib", type=float, default=50.0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    limit = int(args.limit_mib * 1024**2)
    try:
        failures = _worktree_objects(root, limit)
        failures.extend(_history_objects(root, args.base, limit))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"blob validation failed to run: {exc}", file=sys.stderr)
        return 2
    unique = sorted(set(failures), key=lambda item: (item.path, item.source, item.size))
    if unique:
        print(f"repository contains objects at or above {args.limit_mib:g} MiB:", file=sys.stderr)
        for item in unique:
            print(f"  {item.source}: {item.size} bytes  {item.path}", file=sys.stderr)
        return 1
    print(f"blob validation passed (limit {args.limit_mib:g} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
