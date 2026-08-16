r"""Protect MathML from the translator with a MASK -> TRANSLATE -> RESTORE wrapper.

Why this exists
---------------
`epub_translator` does not pass MathML through untouched. Its
`translation/xml_interrupter.py` detects every `<math>` element, converts it to
LaTeX with the `mathml2latex` package (which floods stderr with
`Unknown Tag appeared!! ... semantics`, because it does not understand the
MathML `<semantics>` wrapper), and hands the LaTeX to the LLM. On the way back:

  * inline math is re-inserted as the ORIGINAL element but re-serialised with an
    `m:` namespace prefix (`<m:math><m:semantics>...`), which many EPUB readers
    refuse to render, so formulas *look* flattened; and
  * DISPLAY / block math (numbered equations, matrices) frequently leaks into the
    output as literal LaTeX text such as `$$\mathcal{Y}_{h} = ...$$` or, worse,
    an empty `$\boxed{ \begin{array}{r l} \end{array} }$` — the structure is lost.

The clean, library-agnostic fix is to never let the translator see the math at
all: replace every `<math>...</math>` element in the source XHTML with an inert
alphanumeric sentinel token *before* translation, then substitute the original,
verbatim, default-namespaced MathML back in *after* translation.

The sentinel (`MATHPLACEHOLDER0001X`) is pure uppercase-letters + digits with no
whitespace or punctuation, so (a) the translator's whitespace/punctuation
normalisation never splits it and (b) the LLM copies it through verbatim the way
it copies identifiers/codes. Restoration is tolerant: it also matches tokens the
LLM lightly mangled (case change, stray spaces around the number).

In `submit: append-block` mode each source block is kept verbatim AND followed by
its translation, so every token appears at least twice; the verbatim original
copy guarantees restoration even if the LLM altered the token in the translated
copy. All occurrences are restored.
"""

import re
import shutil
import zipfile
from pathlib import Path

MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# One <math>...</math> element (MathML never nests <math>, so non-greedy is safe).
_MATH_RE = re.compile(r"<math\b.*?</math\s*>", re.S | re.I)

_TOKEN_PREFIX = "MATHPLACEHOLDER"
_TOKEN_SUFFIX = "X"


def _token(index: int) -> str:
    return f"{_TOKEN_PREFIX}{index:04d}{_TOKEN_SUFFIX}"


# Tolerant matcher used on the way back: exact token, but also tolerates a case
# change and/or stray whitespace the LLM may have introduced around the number.
_TOKEN_RE = re.compile(
    rf"{_TOKEN_PREFIX}\s*0*(\d+)\s*{_TOKEN_SUFFIX}",
    re.I,
)


def _content_files(root: Path) -> list[Path]:
    return sorted(
        p
        for ext in ("*.xhtml", "*.html", "*.htm")
        for p in root.rglob(ext)
    )


def _repackage(work: Path, dst: Path) -> None:
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


def mask_epub(src: Path, dst: Path) -> tuple[dict[int, str], int]:
    """Replace every <math> element in `src` with a sentinel token.

    Returns (mapping index->original MathML string, number of elements masked).
    Writes the masked EPUB to `dst`.
    """
    src, dst = src.resolve(), dst.resolve()
    work = dst.parent / f".{dst.stem}.mathmask"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)

    mapping: dict[int, str] = {}
    counter = 0

    for html_path in _content_files(work):
        # bytes, not read_text(): text mode would silently rewrite CRLF line
        # endings to LF and break the byte-for-byte guarantee.
        html = html_path.read_bytes().decode("utf-8")
        if "<math" not in html:
            continue

        def repl(m: re.Match) -> str:
            nonlocal counter
            counter += 1
            mapping[counter] = _normalize_math(m.group(0))
            return _token(counter)

        new_html = _MATH_RE.sub(repl, html)
        html_path.write_bytes(new_html.encode("utf-8"))

    _repackage(work, dst)
    shutil.rmtree(work, ignore_errors=True)
    return mapping, counter


def _normalize_math(math: str) -> str:
    """Ensure the restored <math> carries the default MathML namespace so
    readers render it (pandoc output already does; this is belt-and-braces)."""
    if "xmlns=" in math.split(">", 1)[0]:
        return math
    return math.replace("<math", f'<math xmlns="{MATHML_NS}"', 1)


def restore_text(text: str, mapping: dict[int, str]) -> tuple[str, int]:
    """Replace every sentinel token in `text` with its original MathML.
    Returns (restored_text, occurrences_restored)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        idx = int(m.group(1))
        math = mapping.get(idx)
        if math is None:
            return m.group(0)  # unknown id: leave untouched
        count += 1
        return math

    return _TOKEN_RE.sub(repl, text), count


def restore_epub(src: Path, dst: Path, mapping: dict[int, str]) -> int:
    """Restore all sentinel tokens in `src` back to MathML, write to `dst`.
    Returns total occurrences restored."""
    src, dst = src.resolve(), dst.resolve()
    work = dst.parent / f".{dst.stem}.mathrestore"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(work)

    total = 0
    for html_path in _content_files(work):
        html = html_path.read_bytes().decode("utf-8")
        if _TOKEN_PREFIX not in html.upper():
            continue
        new_html, n = restore_text(html, mapping)
        if n:
            html_path.write_bytes(new_html.encode("utf-8"))
            total += n

    _repackage(work, dst)
    shutil.rmtree(work, ignore_errors=True)
    return total
