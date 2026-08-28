r"""Plumbing shared by the three MASK -> TRANSLATE -> RESTORE modules.

`mask_math.py`, `mask_code.py` and `mask_table.py` all do the same thing to an
EPUB: unzip it, rewrite the XHTML documents, zip it back up. They differ only in
*what* they replace and with which placeholder. Everything below is the part
that does not differ, kept in one place so a fix (a new content-file extension,
an EPUB packaging rule, a bug in the depth counter) lands in all three at once
instead of in one of three copies.

`content_files` / `repackage` were duplicated verbatim in `mask_math.py` and
`mask_code.py`; `tag_end` / `element_end` lived in `mask_code.py` and were
already tag-agnostic. They were moved here unchanged — the behaviour of the
maskers that call them is identical to before the move.
"""

import re
import zipfile
from pathlib import Path


def content_files(root: Path) -> list[Path]:
    return sorted(
        p
        for ext in ("*.xhtml", "*.html", "*.htm")
        for p in root.rglob(ext)
    )


def repackage(work: Path, dst: Path) -> None:
    """Zip `work` into a valid EPUB: mimetype stored first & uncompressed."""
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        mt = work / "mimetype"
        if mt.exists():
            zf.writestr("mimetype", mt.read_bytes(), compress_type=zipfile.ZIP_STORED)
        for path in sorted(work.rglob("*")):
            if path.is_dir() or path.name == "mimetype":
                continue
            zf.write(path, path.relative_to(work).as_posix())


def tag_end(html: str, start: int) -> int:
    """Index of the `>` closing the tag that starts at `start`, or -1.

    Quote-aware, so an attribute value containing `>` cannot end the tag early.
    """
    quote: str | None = None
    for i in range(start, len(html)):
        ch = html[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == ">":
            return i
    return -1


def element_end(html: str, start: int, tag: str) -> int | None:
    """Index just past the element of type `tag` that opens at `start`.

    Depth-counting, so `<code>a<code>b</code>c</code>` (illegal but seen in the
    wild) is captured as ONE element instead of being cut at the first `</code>`.
    Returns None when the element is never closed; the caller then leaves it
    alone rather than masking a broken span.
    """
    gt = tag_end(html, start)
    if gt < 0:
        return None
    if html[gt - 1] == "/":  # <code/> — self-closing, nothing to match
        return gt + 1

    open_re = re.compile(rf"<{tag}\b", re.I)
    close_re = re.compile(rf"</{tag}\s*>", re.I)
    depth, pos = 1, gt + 1
    while depth > 0:
        close_m = close_re.search(html, pos)
        if close_m is None:
            return None
        open_m = open_re.search(html, pos)
        if open_m is not None and open_m.start() < close_m.start():
            inner_gt = tag_end(html, open_m.start())
            if inner_gt < 0:
                return None
            if html[inner_gt - 1] != "/":
                depth += 1
            pos = inner_gt + 1
        else:
            depth -= 1
            pos = close_m.end()
    return pos
