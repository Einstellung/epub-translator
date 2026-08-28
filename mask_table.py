r"""Keep tables out of the translator with a MASK -> TRANSLATE -> RESTORE wrapper.

Why this exists
---------------
A table is the one structure `submit: append-block` cannot survive. The
translator's inline-tag set (`epub_translator/xml/inline.py:_HTML_INLINE_TAGS`)
does not contain `table`, `tr`, `td` or `th`, so every single cell counts as its
own BLOCK. In append-block mode each block is emitted verbatim and then followed
by a translated copy of itself — inside a table that means a translated `<td>`
is appended after every original `<td>`, so:

  * the column count doubles. A 3-column `<thead>` comes back with 6 cells and
    every reader lays the table out wrong from the header row down; the observed
    damage is in `OEBPS/ch02.html` of the shipped
    `Hands-On大语言模型_第1-3章_中英对照.epub`.
  * numeric and identifier columns get "translated" anyway. Model sizes, token
    counts, release years, model names — the majority of cells in a technical
    book's tables carry nothing to translate, so the duplicate cell is pure
    noise (and pure spend).
  * the reading experience collapses. Even where a cell holds real prose, the
    bilingual pairing that works so well for paragraphs is unreadable when the
    two languages are interleaved cell by cell across a grid.

The decision is therefore: **a table is not translated at all**. It is removed
from the translator's view before translation and put back verbatim afterwards,
so the table in the bilingual output is byte-for-byte the table in the source.
The table's *caption* normally lives outside the element (O'Reilly emits
`<h6>Table 3-1. …</h6>` or a `<p>` before the `<table>`), so it is untouched by
this module and gets translated like any other paragraph — which is what the
reader needs to navigate the table.

Why the placeholder is an EMPTY element
---------------------------------------
The stand-in is `<table data-tablemask="7"></table>` — a block element with NO
text, exactly like the `<pre>` placeholder in `mask_code.py` and for the same
reason. `epub_translator` only visits elements that contain text, and in
append-block mode it appends a translated copy of every block it visits; a
placeholder carrying a text sentinel would therefore be emitted TWICE and the
restored table would appear twice in the book. With no text at all the
placeholder never enters the translator's segment mapping: it is passed through
untouched and restored exactly once, in place.

The attribute is `data-tablemask`, not `id`: the translator runs
`deduplicate_ids_in_element()` over every chapter and would rewrite an `id`.
Restoration also accepts `<table data-tablemask="7" />`, because an XML
round-trip re-serialises an empty element in that short form.

Nested `<table>` (a table used for layout inside a cell — rare, but it happens)
is captured as ONE element by the depth counter in `mask_common.element_end()`,
never cut at the first `</table>`. An unclosed `<table>` is left exactly as it
is rather than masked as a broken span.

All occurrences are restored, not just the first: append-block keeps the source
block and appends its translation, so a placeholder that ever did reach the
translator's output could appear more than once, and restoring only the first
would lose the table from the translated half of the book (the trap
`mask_math.py` documents).

Stacking with `mask_math.py` and `mask_code.py`
-----------------------------------------------
The three maskers are mutually inert: a `MATHPLACEHOLDER…X` / `CODEPLACEHOLDER…X`
sentinel is plain text with no `<table>` markup, and an empty
`<table data-tablemask="N">` contains neither `<math>` nor `<pre>`/`<code>`. They
may be applied in any order, but they must be UNDONE in the reverse order of
application (LIFO): whichever masker ran last may have stored the others'
placeholders inside its own mapping — a `<pre>` inside a table cell is already a
`<pre data-codemask="N">` by the time the table is captured — and those
placeholders only come back into the document when that mapping is restored.
`translate_book.py` masks math -> code -> table and restores table -> code ->
math.
"""

import re
import shutil
import zipfile
from pathlib import Path

from mask_common import content_files, element_end, repackage

# Opening tag of a maskable element. \b keeps <tablefoot>/<tabledata> out.
_OPEN_RE = re.compile(r"<table\b", re.I)
_TAG = "table"

# Attribute carrying the index of a masked <table>. Not "id": the translator
# runs deduplicate_ids_in_element() over every chapter and would rewrite it.
_BLOCK_ATTR = "data-tablemask"


def _block_placeholder(index: int) -> str:
    return f'<table {_BLOCK_ATTR}="{index}"></table>'


# Tolerant matcher used on the way back. The placeholder survives an XML
# round-trip, which re-serialises an empty element as
# `<table data-tablemask="7" />`; that form and `...></table>` are both accepted.
# There is no text sentinel to match here — a table is never masked inline.
_RESTORE_RE = re.compile(
    rf'<{_TAG}\b[^>]*\b{_BLOCK_ATTR}="(\d+)"[^>]*>(?:\s*</{_TAG}\s*>)?',
    re.I,
)


def _mask_html(html: str, mapping: dict[int, str], counter: int) -> tuple[str, int]:
    """Mask every top-level <table> element in `html`.

    Scans left to right and jumps past each element it captures, so a `<table>`
    nested inside another `<table>` is swallowed by the outer one and never
    masked twice.
    """
    out: list[str] = []
    pos = 0
    while True:
        m = _OPEN_RE.search(html, pos)
        if m is None:
            break
        end = element_end(html, m.start(), _TAG)
        if end is None:  # unbalanced markup: leave it exactly as it is
            out.append(html[pos : m.end()])
            pos = m.end()
            continue
        counter += 1
        mapping[counter] = html[m.start() : end]
        out.append(html[pos : m.start()])
        out.append(_block_placeholder(counter))
        pos = end
    out.append(html[pos:])
    return "".join(out), counter


def mask_epub(source: Path, dest: Path) -> tuple[dict[int, str], int]:
    """Replace every <table> element in `source` with an empty placeholder.

    Returns (mapping index->original markup, number of elements masked).
    Writes the masked EPUB to `dest`.
    """
    source, dest = source.resolve(), dest.resolve()
    work = dest.parent / f".{dest.stem}.tablemask"
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
        if "<table" not in html.lower():
            continue
        new_html, counter = _mask_html(html, mapping, counter)
        if new_html != html:
            html_path.write_bytes(new_html.encode("utf-8"))

    repackage(work, dest)
    shutil.rmtree(work, ignore_errors=True)
    return mapping, counter


def restore_text(text: str, mapping: dict[int, str]) -> tuple[str, int]:
    """Replace every placeholder in `text` with its original table markup.
    Returns (restored_text, occurrences_restored)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        original = mapping.get(int(m.group(1)))
        if original is None:
            return m.group(0)  # unknown id: leave untouched
        count += 1
        return original

    return _RESTORE_RE.sub(repl, text), count


def restore_epub(target: Path, out: Path, mapping: dict[int, str]) -> int:
    """Restore all placeholders in `target` back to tables, write to `out`.
    Returns total occurrences restored."""
    target, out = target.resolve(), out.resolve()
    work = out.parent / f".{out.stem}.tablerestore"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(target) as zf:
        zf.extractall(work)

    total = 0
    for html_path in content_files(work):
        html = html_path.read_bytes().decode("utf-8")
        if _BLOCK_ATTR.upper() not in html.upper():
            continue
        new_html, n = restore_text(html, mapping)
        if n:
            html_path.write_bytes(new_html.encode("utf-8"))
            total += n

    repackage(work, out)
    shutil.rmtree(work, ignore_errors=True)
    return total
