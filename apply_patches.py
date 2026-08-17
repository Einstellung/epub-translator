#!/usr/bin/env python3
"""Re-apply the local fixes this project keeps on top of its installed dependencies.

Two bugs in `epub_translator` are fatal for a book-length run and are fixed here rather
than upstream:

  * `tiktoken` refuses to encode a literal ``<|endoftext|>``. A book *about* language
    models quotes such markers as ordinary prose, so the run dies partway through with
    ``ValueError: Encountered text corresponding to disallowed special token`` and
    cannot be resumed past it.
  * the retry predicate does not treat 429 (rate limit), 500 or 529 as retryable, so
    ``EPUB_TRANSLATOR_RETRY_TIMES`` never gets a chance to do its job on those.

The fixes live in ``patches/*.patch`` as ordinary unified diffs against the pristine
wheel contents. `uv sync` (and especially `uv sync --reinstall`) restores the pristine
files, so run this afterwards::

    uv run python apply_patches.py

It is safe to run any number of times: an already-patched file is detected and skipped.
If a patch no longer fits - because the dependency was upgraded - it says so loudly and
exits non-zero rather than leaving a half-patched environment behind.

    uv run python apply_patches.py --check    # report only, change nothing

Note for anyone editing a site-packages file by hand instead: uv installs packages by
*hardlinking* out of ``~/.cache/uv/archive-v0/``, so editing a file in place also edits
the shared uv cache, and every other project on the machine silently inherits the edit.
This script always writes a fresh temp file and ``os.replace``s it into position, which
breaks the hardlink and leaves the cache pristine.
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path

PATCHES_DIR = Path(__file__).resolve().parent / "patches"


class PatchError(RuntimeError):
    pass


@dataclass
class Hunk:
    old_start: int  # 1-based, as written in the @@ header
    old_lines: list[str]
    new_lines: list[str]


@dataclass
class Patch:
    path: Path  # source .patch file
    target: str  # target path relative to site-packages
    hunks: list[Hunk]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _strip_prefix(path: str) -> str:
    """`a/epub_translator/llm/error.py` -> `epub_translator/llm/error.py`."""
    path = path.split("\t")[0].strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def parse_patch(path: Path) -> Patch:
    lines = path.read_text(encoding="utf-8").splitlines()
    target: str | None = None
    hunks: list[Hunk] = []
    current: Hunk | None = None

    for lineno, line in enumerate(lines, 1):
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            if target is not None:
                raise PatchError(f"{path.name}:{lineno}: more than one target file in a single patch")
            target = _strip_prefix(line[4:])
            continue
        if line.startswith("@@"):
            try:
                old_span = line.split("@@")[1].strip().split(" ")[0]  # e.g. "-16,9"
                old_start = int(old_span.lstrip("-").split(",")[0])
            except (IndexError, ValueError) as err:
                raise PatchError(f"{path.name}:{lineno}: cannot parse hunk header {line!r}") from err
            current = Hunk(old_start=old_start, old_lines=[], new_lines=[])
            hunks.append(current)
            continue
        if current is None:
            continue  # preamble before the first hunk
        if line.startswith("\\"):
            raise PatchError(
                f"{path.name}:{lineno}: patches without a trailing newline are not supported ({line!r})"
            )
        if not line:
            # `diff -u` writes a context line that is empty as a bare empty line.
            current.old_lines.append("")
            current.new_lines.append("")
        elif line[0] == " ":
            current.old_lines.append(line[1:])
            current.new_lines.append(line[1:])
        elif line[0] == "-":
            current.old_lines.append(line[1:])
        elif line[0] == "+":
            current.new_lines.append(line[1:])
        else:
            raise PatchError(f"{path.name}:{lineno}: unexpected line {line!r}")

    if target is None:
        raise PatchError(f"{path.name}: no `+++ <file>` header found")
    if not hunks:
        raise PatchError(f"{path.name}: no hunks found")
    return Patch(path=path, target=target, hunks=hunks)


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------


def _find(haystack: list[str], needle: list[str], hint: int) -> int | None:
    """Index of `needle` inside `haystack`, preferring the occurrence nearest `hint`."""
    if not needle:
        return None
    matches = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i : i + len(needle)] == needle]
    if not matches:
        return None
    return min(matches, key=lambda i: abs(i - hint))


def transform(lines: list[str], hunks: list[Hunk], reverse: bool) -> list[str]:
    """Return `lines` with every hunk applied, or raise PatchError.

    Matching is exact - no fuzz. A dependency upgrade that reflows the surrounding code
    must fail here so the patch gets regenerated, rather than land somewhere plausible.
    """
    result = list(lines)
    offset = 0
    for n, hunk in enumerate(hunks, 1):
        old = hunk.new_lines if reverse else hunk.old_lines
        new = hunk.old_lines if reverse else hunk.new_lines
        hint = max(0, hunk.old_start - 1 + offset)
        at = _find(result, old, hint)
        if at is None:
            raise PatchError(f"hunk #{n} does not match (expected near line {hunk.old_start})")
        result[at : at + len(old)] = new
        offset += len(new) - len(old)
    return result


def _write(target: Path, lines: list[str]) -> None:
    """Replace `target` atomically, breaking any hardlink into the uv cache."""
    text = "\n".join(lines) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def site_packages() -> Path:
    if sys.prefix == sys.base_prefix:
        raise PatchError(
            "not running inside the project virtualenv, refusing to patch system packages.\n"
            "  Run it as:  uv run python apply_patches.py\n"
            "         or:  .venv/bin/python apply_patches.py"
        )
    return Path(sysconfig.get_path("purelib"))


# status codes returned by apply_one()
APPLIED = "applied"
ALREADY = "already applied"


def apply_one(patch: Patch, root: Path, check_only: bool) -> str:
    target = root / patch.target
    if not target.is_file():
        raise PatchError(f"target file not found: {target}")
    lines = target.read_text(encoding="utf-8").splitlines()

    # Idempotency: if the reverse patch fits, the file already carries the fix.
    try:
        transform(lines, patch.hunks, reverse=True)
        return ALREADY
    except PatchError:
        pass

    try:
        patched = transform(lines, patch.hunks, reverse=False)
    except PatchError as err:
        raise PatchError(
            f"{patch.path.name}: cannot apply to {patch.target}: {err}\n"
            f"  The file is neither pristine nor already patched. Most likely the "
            f"dependency was upgraded.\n"
            f"  Regenerate the patch against the new version before running a book.",
        ) from None

    if not check_only:
        _write(target, patched)
    return APPLIED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would happen and exit non-zero if anything is unpatched; write nothing",
    )
    args = parser.parse_args()

    try:
        root = site_packages()
    except PatchError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    patch_files = sorted(PATCHES_DIR.glob("*.patch"))
    if not patch_files:
        print(f"error: no patches found in {PATCHES_DIR}", file=sys.stderr)
        return 2

    print(f"site-packages: {root}")
    failures: list[str] = []
    pending = 0

    for path in patch_files:
        try:
            patch = parse_patch(path)
            status = apply_one(patch, root, check_only=args.check)
        except PatchError as err:
            failures.append(str(err))
            print(f"  FAIL     {path.name}")
            continue
        if status == APPLIED:
            pending += 1
            verb = "would patch" if args.check else "patched"
            print(f"  {verb:<12} {path.name} -> {patch.target}")
        else:
            print(f"  {'ok':<12} {path.name} -> {patch.target} (already applied)")

    if failures:
        sys.stdout.flush()  # keep the per-patch log above the error block when piped
        print("", file=sys.stderr)
        for message in failures:
            print(f"error: {message}", file=sys.stderr)
        return 1

    if args.check:
        if pending:
            sys.stdout.flush()
            print(f"\n{pending} patch(es) not applied. Run: uv run python apply_patches.py", file=sys.stderr)
            return 1
        print("\nall patches already applied")
        return 0

    print(f"\ndone ({pending} newly applied, {len(patch_files) - pending} already in place)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
