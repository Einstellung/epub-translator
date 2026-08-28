r"""Keep source code out of the translator with a MASK -> TRANSLATE -> RESTORE wrapper.

Why this exists
---------------
In a technical book, code must never be translated: function names, variable
names, string literals and shell output are not prose. Asking the LLM nicely
("do not translate code") is not a guarantee — one chapter of *Hands-On Large
Language Models* alone carries 1,366 `<code>` elements and 34 `<pre>` listings;
a single slip renames an identifier in the Chinese text, and re-assembling a
paragraph that is 90% inline tags is exactly where the translator's inline
segment mapper is most likely to drop or reorder markup.

The reliable fix is the one `mask_math.py` already uses for MathML: make the
model unable to see the code at all. Before translation every code fragment is
swapped for an inert placeholder; after translation the ORIGINAL bytes are put
back verbatim, so the code in the output is byte-for-byte the code in the
source — entities, whitespace, syntax-highlight spans and all.

Two kinds of code, two kinds of placeholder
-------------------------------------------
* **Inline `<code>`** (outside `<pre>`) becomes a text sentinel
  `CODEPLACEHOLDER00001X` — pure uppercase letters + digits, no whitespace or
  punctuation, so the translator's whitespace/punctuation normalisation cannot
  split it and the LLM copies it through the way it copies identifiers. The
  sentinel sits *inside* the sentence, which is what we want: the surrounding
  prose is still translated, with the code pinned at its original position.

* **`<pre>` blocks** become an EMPTY element `<pre data-codemask="7"></pre>`
  instead. A text sentinel would be wrong here: `epub_translator` only visits
  elements that contain text, and in `submit: append-block` mode it appends a
  translated copy of every block it visits — a `<pre>` holding a sentinel would
  therefore be emitted TWICE, i.e. every code listing duplicated. With no text
  at all the block never enters the translator's segment mapping, so it is
  passed through untouched and restored exactly once, in place.
  Nested `<code>` inside a `<pre>` is part of the block and is NOT masked
  separately (the scanner consumes the whole `<pre>` element).

Restoration is deliberately tolerant of the LLM: the sentinel matcher also
accepts a case change or stray whitespace around the number. In `append-block`
mode each source block is kept verbatim AND followed by its translation, so an
inline sentinel appears at least twice; the verbatim original copy guarantees
restoration even if the model altered the token in the translated copy. ALL
occurrences are restored (this is the trap `mask_math.py` documents: restoring
only the first occurrence loses the code in the translated half of the book).

Stacking with `mask_math.py` and `mask_table.py`
------------------------------------------------
The maskers are mutually inert: a `MATHPLACEHOLDER...X` sentinel is plain
text containing no `<pre>`/`<code>` markup, so this module cannot swallow or
split it, and neither `CODEPLACEHOLDER...X` nor `<pre data-codemask="N">`
contains a `<math>` element, so `mask_math` cannot touch them. They may
therefore be applied in either order, but they must be UNDONE in the reverse
order of application (LIFO): whichever masker ran last may have stored the
other's sentinels inside its mapping (e.g. a `<math>` inside a `<pre>` is
already a math sentinel by the time the `<pre>` is captured), and those
sentinels only come back into the document when that mapping is restored. The
same argument extends to `mask_table.py`, which runs after this module and
captures whole `<table>` elements — including any code placeholder already
sitting in a cell. `translate_book.py` masks math -> code -> table and restores
table -> code -> math.
"""

import re
import shutil
import zipfile
from pathlib import Path

from mask_common import content_files, element_end, repackage

# Opening tag of a maskable element. \b keeps <preamble>/<codex> out.
_OPEN_RE = re.compile(r"<(pre|code)\b", re.I)

_TOKEN_PREFIX = "CODEPLACEHOLDER"
_TOKEN_SUFFIX = "X"

# Attribute carrying the index of a masked <pre> block. Not "id": the translator
# runs deduplicate_ids_in_element() over every chapter and would rewrite it.
_BLOCK_ATTR = "data-codemask"


def _token(index: int) -> str:
    # 5 digits: a full technical book runs to several thousand fragments
    # (1,366 in a single chapter of the reference book).
    return f"{_TOKEN_PREFIX}{index:05d}{_TOKEN_SUFFIX}"


def _block_placeholder(index: int) -> str:
    return f'<pre {_BLOCK_ATTR}="{index}"></pre>'


# Tolerant matcher used on the way back. Both placeholder kinds are matched in
# ONE pass so that restored code is never rescanned (a listing that happens to
# print the literal string "CODEPLACEHOLDER00001X" must not be re-substituted).
#   group 1 - a masked <pre> block. The placeholder survives an XML round-trip,
#             which re-serialises an empty element as `<pre data-codemask="7" />`;
#             that form and `...></pre>` are both accepted.
#   group 2 - an inline sentinel: the exact token, but also tolerating a case
#             change and/or stray whitespace the LLM may have put around the
#             number.
_RESTORE_RE = re.compile(
    rf'<pre\b[^>]*\b{_BLOCK_ATTR}="(\d+)"[^>]*>(?:\s*</pre\s*>)?'
    rf"|{_TOKEN_PREFIX}\s*0*(\d+)\s*{_TOKEN_SUFFIX}",
    re.I,
)


def _mask_html(html: str, mapping: dict[int, str], counter: int) -> tuple[str, int]:
    """Mask every top-level <pre>/<code> element in `html`.

    Scans left to right and jumps past each element it captures, so a `<code>`
    nested in a `<pre>` is swallowed by the `<pre>` and never masked twice.
    """
    out: list[str] = []
    pos = 0
    while True:
        m = _OPEN_RE.search(html, pos)
        if m is None:
            break
        tag = m.group(1).lower()
        end = element_end(html, m.start(), tag)
        if end is None:  # unbalanced markup: leave it exactly as it is
            out.append(html[pos : m.end()])
            pos = m.end()
            continue
        counter += 1
        mapping[counter] = html[m.start() : end]
        out.append(html[pos : m.start()])
        out.append(_block_placeholder(counter) if tag == "pre" else _token(counter))
        pos = end
    out.append(html[pos:])
    return "".join(out), counter


def mask_epub(source: Path, dest: Path) -> tuple[dict[int, str], int]:
    """Replace every <pre> block and inline <code> element in `source` with an
    inert placeholder.

    Returns (mapping index->original markup, number of elements masked).
    Writes the masked EPUB to `dest`.
    """
    source, dest = source.resolve(), dest.resolve()
    work = dest.parent / f".{dest.stem}.codemask"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(source) as zf:
        zf.extractall(work)

    mapping: dict[int, str] = {}
    counter = 0

    for html_path in content_files(work):
        # bytes, not read_text(): text mode would silently rewrite CRLF line
        # endings to LF and break the byte-for-byte guarantee.
        html = html_path.read_bytes().decode("utf-8")
        if "<pre" not in html.lower() and "<code" not in html.lower():
            continue
        new_html, counter = _mask_html(html, mapping, counter)
        if new_html != html:
            html_path.write_bytes(new_html.encode("utf-8"))

    repackage(work, dest)
    shutil.rmtree(work, ignore_errors=True)
    return mapping, counter


def restore_text(text: str, mapping: dict[int, str]) -> tuple[str, int]:
    """Replace every placeholder in `text` with its original markup.
    Returns (restored_text, occurrences_restored)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        original = mapping.get(int(m.group(1) or m.group(2)))
        if original is None:
            return m.group(0)  # unknown id: leave untouched
        count += 1
        return original

    return _RESTORE_RE.sub(repl, text), count


def restore_epub(target: Path, out: Path, mapping: dict[int, str]) -> int:
    """Restore all placeholders in `target` back to code, write to `out`.
    Returns total occurrences restored."""
    target, out = target.resolve(), out.resolve()
    work = out.parent / f".{out.stem}.coderestore"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(target) as zf:
        zf.extractall(work)

    total = 0
    for html_path in content_files(work):
        html = html_path.read_bytes().decode("utf-8")
        upper = html.upper()
        if _TOKEN_PREFIX not in upper and _BLOCK_ATTR.upper() not in upper:
            continue
        new_html, n = restore_text(html, mapping)
        if n:
            html_path.write_bytes(new_html.encode("utf-8"))
            total += n

    repackage(work, out)
    shutil.rmtree(work, ignore_errors=True)
    return total
