"""Repair MathML namespaces in a translated EPUB.

Why this exists: epub-translator re-serialises XHTML through a parser that
rewrites each self-contained `<math xmlns="...MathML">` element into a
prefixed `<m:math>` form relying on a namespace declaration on the root
<html>. That rewrite is not understood by many EPUB readers, which then treat
`<m:mtable>`/`<m:mtr>`/`<m:mtd>` as unknown tags and drop the table structure,
so every matrix collapses onto a single line.

The fix is mechanical and lossless: for each `<m:math>...</m:math>` block,
re-add `xmlns="http://www.w3.org/1998/Math/MathML"` to the root and strip the
`m:` prefix from every tag inside, restoring the default-namespaced form that
pandoc originally produced and that readers render correctly.

The formula *content* is untouched (the translator already skips math, so the
text is byte-identical before and after translation) — only the namespace
serialisation is repaired.

Usage:
    uv run python fix_epub_math.py output/book.zh.epub
    uv run python fix_epub_math.py output/book.zh.epub -o output/book.fixed.epub
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

MATHML_NS = "http://www.w3.org/1998/Math/MathML"


def _fix_xhtml(html: str) -> tuple[str, int]:
    """Return (fixed_html, blocks_fixed)."""
    count = 0

    def fix_block(m: re.Match) -> str:
        nonlocal count
        count += 1
        block = m.group(0)
        # re-add xmlns to the root <m:math ...>
        block = block.replace(
            "<m:math", f'<math xmlns="{MATHML_NS}"', 1
        )
        # strip the m: prefix from every remaining tag (open and close)
        block = re.sub(r"<(/?)m:", r"<\1", block)
        return block

    fixed = re.sub(r"<m:math\b.*?</m:math>", fix_block, html, flags=re.S)
    return fixed, count


def fix_epub(src: Path, dst: Path) -> int:
    """Repair MathML namespaces in an EPUB. Returns total blocks fixed."""
    src = src.resolve()
    dst = dst.resolve()
    work = dst.parent / f".{dst.stem}.mathfix"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)

    total = 0
    for xhtml in list(work.rglob("*.xhtml")) + list(work.rglob("*.html")):
        html = xhtml.read_text(encoding="utf-8")
        if "<m:math" not in html:
            continue
        fixed, n = _fix_xhtml(html)
        if n:
            xhtml.write_text(fixed, encoding="utf-8")
            total += n

    # repackage: mimetype stored first, rest deflated
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w") as zf:
        mt = work / "mimetype"
        if mt.exists():
            zf.write(mt, "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in sorted(work.rglob("*")):
            if path.is_dir() or path.name == "mimetype":
                continue
            arc = path.relative_to(work).as_posix()
            zf.write(path, arc, compress_type=zipfile.ZIP_DEFLATED)

    shutil.rmtree(work)
    return total


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repair MathML namespaces in a translated EPUB.")
    p.add_argument("epub", type=Path, help="translated EPUB path")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output path (default: overwrite input in place)",
    )
    args = p.parse_args(argv)
    if not args.epub.is_file():
        sys.exit(f"error: EPUB not found: {args.epub}")
    dst = args.output or args.epub
    n = fix_epub(args.epub, dst)
    print(f"repaired {n} MathML block(s) -> {dst}")


if __name__ == "__main__":
    main()
