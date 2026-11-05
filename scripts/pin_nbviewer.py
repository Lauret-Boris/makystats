#!/usr/bin/env python3
"""Pin nbviewer links of the documentation site to one commit SHA.

Why
---
nbviewer caches rendered notebooks *per URL*. A link ending in ``/blob/main/...``
keeps the same URL forever, so nbviewer can keep serving an old render. A link
ending in ``/blob/<sha>/...`` is a new URL for every commit, hence a fresh render.

What it does
------------
Only URLs of the form

    https://nbviewer.org/github/<OWNER>/<REPO>/blob/main/...

are rewritten to ``.../blob/<sha>/...`` in ``*.html`` files. GitHub and Colab
links, README links and everything else are left byte-for-byte untouched.

The source files in ``website/`` keep ``main`` on purpose: a hard-coded SHA there
would be wrong after the next commit. The pinning happens at deploy time only.

Usage
-----
    # build a pinned copy (recommended, safe locally)
    python scripts/pin_nbviewer.py --src website --dst _site --sha "$(git rev-parse HEAD)"

    # CI only: rewrite the runner's checkout in place (needs CI=true, set by GitHub Actions)
    python scripts/pin_nbviewer.py --src website --in-place --sha "$GITHUB_SHA"

Exit codes: 0 ok, 1 verification failed, 2 bad arguments / unsafe situation.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

OWNER = "Lauret-Boris"
REPO = "makystats"

_BASE = rf"https://nbviewer\.org/github/{re.escape(OWNER)}/{re.escape(REPO)}/blob/"
# Matches only the moving ref "main" as a whole path segment.
PIN_RE = re.compile(rf"({_BASE})main(?=/)")
# Used for the post-check: any nbviewer link of this repo that still points at main.
LEFTOVER_RE = re.compile(rf"{_BASE}main/")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_sha(cli_value: str | None) -> str:
    sha = cli_value or os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            sha = ""
    sha = sha.strip().lower()
    if not SHA_RE.match(sha):
        raise SystemExit(
            "error: need a full 40-character lowercase hex commit SHA "
            f"(got {sha!r}). Pass --sha or run inside a git checkout."
        )
    return sha


def pin_text(text: str, sha: str) -> tuple[str, int]:
    """Return (new_text, number_of_links_pinned)."""
    return PIN_RE.subn(lambda m: m.group(1) + sha, text)


def process_tree(root: Path, sha: str) -> tuple[int, int, list[str]]:
    """Rewrite every *.html under root. Returns (files_changed, links_pinned, leftovers)."""
    files_changed = links_pinned = 0
    leftovers: list[str] = []
    for path in sorted(root.rglob("*.html")):
        # newline="" keeps CRLF/LF exactly as they are.
        with open(path, "r", encoding="utf-8", newline="") as fh:
            original = fh.read()
        updated, n = pin_text(original, sha)
        if n:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(updated)
            files_changed += 1
            links_pinned += n
        if LEFTOVER_RE.search(updated):
            leftovers.append(str(path))
    return files_changed, links_pinned, leftovers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="website", help="site folder (default: website)")
    ap.add_argument("--dst", help="write a pinned COPY here (must not exist yet)")
    ap.add_argument("--in-place", action="store_true", help="rewrite --src itself (CI only)")
    ap.add_argument("--sha", help="40-char commit SHA (default: $GITHUB_SHA, else git rev-parse HEAD)")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.is_dir():
        print(f"error: --src {src} is not a directory", file=sys.stderr)
        return 2
    if bool(args.dst) == bool(args.in_place):
        print("error: choose exactly one of --dst or --in-place", file=sys.stderr)
        return 2

    sha = resolve_sha(args.sha)

    if args.in_place:
        if os.environ.get("CI", "").lower() != "true":
            print(
                "error: --in-place would modify your source files. It only runs when CI=true "
                "(GitHub Actions sets it). Locally, use --dst.",
                file=sys.stderr,
            )
            return 2
        target = src
    else:
        dst = Path(args.dst)
        if dst.exists():
            print(f"error: --dst {dst} already exists; refusing to overwrite or delete it", file=sys.stderr)
            return 2
        try:
            if dst.resolve() == src.resolve() or src.resolve() in dst.resolve().parents:
                print("error: --dst must be outside --src", file=sys.stderr)
                return 2
        except OSError as exc:  # pragma: no cover
            print(f"error: cannot resolve paths: {exc}", file=sys.stderr)
            return 2
        shutil.copytree(src, dst)
        target = dst

    files_changed, links_pinned, leftovers = process_tree(target, sha)

    print(f"sha        : {sha}")
    print(f"target     : {target}")
    print(f"pinned     : {links_pinned} nbviewer link(s) in {files_changed} file(s)")

    if leftovers:
        print("error: nbviewer links still pointing at 'main' in:", file=sys.stderr)
        for p in leftovers:
            print(f"  {p}", file=sys.stderr)
        return 1
    if links_pinned == 0:
        # Not fatal in itself (idempotent re-run), but worth a loud notice in CI logs.
        print("warning: no nbviewer link needed pinning (already pinned, or none present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
