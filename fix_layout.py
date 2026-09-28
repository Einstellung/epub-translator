#!/usr/bin/env python3
"""fix_layout.py — repair the bilingual layout of an append-block EPUB.

`epub_translator`'s append-block mode clones every translated leaf block and
inserts the clone right after the original, attributes and all. That is right
for a paragraph or a heading and wrong in four places this pass fixes:

  1. **Language.** The clone keeps the source block's attributes and sits under
     `<html xml:lang="en">`, so every Chinese paragraph is declared English.
     Each translation block gets `lang="zh"` (plus `xml:lang="zh"` when the
     document uses `xml:lang`). A block already tagged `zh*` (e.g. by
     `polish_cjk.py`) is left as it is.
  2. **List items.** The translation of an `<li>` is a second `<li>`: an extra
     bullet, and an `<ol>` numbers the translations too. The translation moves
     into the original item as a `<div lang="zh">`, so the list keeps its items
     and its numbering.
  3. **Notes.** A footnote that is itself the leaf block (`<aside
     epub:type="footnote">text</aside>`) is cloned into a second note: two
     notes for one reference, and in readers that hide footnote asides the
     translated one is never shown. It moves into the original note the same
     way. Inside a note the clone also repeats the note's backlink (`1`, `*`),
     so the translation drops that copy: one note, one backlink.
  4. **TOC numbering.** Older runs translated the TOC in append-text form,
     which repeats a section number: `I. INTRODUCTION I. 引言`. The second
     number is dropped: `I. INTRODUCTION 引言`. Current runs leave the TOC
     untranslated (patches/epub_translator-epub-toc.patch), so this only
     matters for books translated before that patch.

A translation block is recognised structurally, not by script alone: the
element right after an original, with the same tag and the same attributes
(the id may carry the translator's `__translated` suffix; `polish_cjk.py`'s
class and lang are ignored), where the original has no CJK characters and the
clone is CJK-dominant. Blocks the translator wrote in append-text form (text
and child blocks mixed in one element) have no clone and are left alone.

Like `polish_cjk.py`, nothing is re-serialised: the parser only yields source
offsets and every change is a point edit on the raw text, so everything this
pass does not touch is byte-identical. Running it on its own output changes
nothing. Order relative to `polish_cjk.py` does not matter.

Usage:
    uv run python fix_layout.py IN.epub OUT.epub
    uv run python fix_layout.py IN.epub --stats        # count only, no write
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import zipfile
from pathlib import Path

from polish_cjk import (
    BLOCK_TAGS,
    OPAQUE_TAGS,
    _Node,
    _stream,
    _Tree,
    is_cjk_script,
)

TARGET_LANG = "zh"
THRESHOLD = 0.15  # same weighted CJK share as polish_cjk.py
CJK_WEIGHT = 2.0

_ID_SUFFIX = re.compile(r"__translated(?:_\d+)?$")
_ATTR = re.compile(r"""([^\s=/>]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+))?""")
_WS = " \t\r\n\f\v"

# A single note, whose clone must not be a second note.
NOTE_TOKENS = frozenset(
    "footnote endnote rearnote note doc-footnote doc-endnote".split()
)
BACKLINK_TOKENS = frozenset("backlink doc-backlink".split())

# A section number worth de-duplicating: `1`, `2.3`, `1.`, `I.`, `A.`, `IV:`.
# A bare letter or roman numeral needs its punctuation — `I` and `A` are words.
_NUMBER = re.compile(r"\d+(?:[.\-]\d+)*[.:)]?|(?:[IVXLCDM]+|[A-Za-z])[.:)]")


# ── parsing helpers ─────────────────────────────────────────────────────


def attrs_of(node: _Node, text: str) -> dict[str, str]:
    raw = text[node.raw_start:node.raw_end]
    m = re.match(r"<[^\s/>]+", raw)
    body = raw[m.end() if m else 0:].rstrip(">").rstrip("/")
    out: dict[str, str] = {}
    for am in _ATTR.finditer(body):
        val = am.group(2) or ""
        if val[:1] in "\"'":
            val = val[1:-1]
        out[am.group(1).lower()] = val
    return out


def _tokens(attrs: dict[str, str], *names: str) -> set[str]:
    out: set[str] = set()
    for n in names:
        out.update(attrs.get(n, "").lower().split())
    return out


def is_note(attrs: dict[str, str], tokens=NOTE_TOKENS) -> bool:
    return bool(_tokens(attrs, "epub:type", "type", "role") & tokens)


def _clone_key(attrs: dict[str, str]) -> tuple:
    """What a clone shares with its original once the translator's and
    polish_cjk.py's own additions are taken out."""
    key = {}
    for name, val in attrs.items():
        if name in ("lang", "xml:lang"):
            continue
        if name == "class":
            val = " ".join(c for c in val.split() if c != "zh-translation")
            if not val:
                continue
        if name == "id":
            val = _ID_SUFFIX.sub("", val)
        key[name] = val
    return tuple(sorted(key.items()))


def _visible(node: _Node, text: str) -> str:
    return "".join(ch for ch, _ in _stream(node, text))


def _cjk_share(s: str) -> tuple[int, float]:
    cjk = sum(1 for ch in s if is_cjk_script(ch))
    latin = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    w = cjk * CJK_WEIGHT
    return cjk, (w / (w + latin) if w + latin else 0.0)


def is_leaf(node: _Node) -> bool:
    """A block holding only text and inline markup — the translator's unit."""
    for kind, child in node.children:
        if kind != "node":
            continue
        if child.tag in BLOCK_TAGS:
            return False
        if child.tag not in OPAQUE_TAGS and not is_leaf(child):
            return False
    return True


def is_translation_of(src: _Node, tr: _Node, text: str) -> bool:
    if src.tag != tr.tag or src.tag not in BLOCK_TAGS:
        return False
    if src.end_end is None or tr.end_end is None:
        return False
    if not (is_leaf(src) and is_leaf(tr)):
        return False
    if _clone_key(attrs_of(src, text)) != _clone_key(attrs_of(tr, text)):
        return False
    src_cjk, _ = _cjk_share(_visible(src, text))
    tr_cjk, share = _cjk_share(_visible(tr, text))
    return src_cjk == 0 and tr_cjk > 0 and share >= THRESHOLD


def pairs(node: _Node, text: str, ancestors: tuple[_Node, ...] = ()):
    """Yield (original, translation, ancestors) for every clone pair."""
    kids = node.children
    i = 0
    while i < len(kids):
        kind, child = kids[i]
        if kind == "node":
            # the next element sibling, across whitespace only
            j = i + 1
            while j < len(kids) and kids[j][0] == "text" and \
                    not text[kids[j][1][0]:kids[j][1][1]].strip(_WS):
                j += 1
            if j < len(kids) and kids[j][0] == "node" and \
                    is_translation_of(child, kids[j][1], text):
                yield child, kids[j][1], ancestors + (node,)
                i = j + 1
                continue
            yield from pairs(child, text, ancestors + (node,))
        i += 1


# ── edits ───────────────────────────────────────────────────────────────


def _leading_links(node: _Node, text: str) -> list[_Node]:
    """`<a>` elements before the first visible character of `node`."""
    found: list[_Node] = []

    def walk(n: _Node) -> bool:  # True once visible text has been seen
        for kind, child in n.children:
            if kind == "text":
                if text[child[0]:child[1]].strip(_WS):
                    return True
            elif kind == "opaque":
                return True
            elif child.tag == "a":
                found.append(child)
                return True
            elif walk(child):
                return True
        return False

    walk(node)
    return found


def _all_links(node: _Node):
    for kind, child in node.children:
        if kind == "node":
            if child.tag == "a":
                yield child
            yield from _all_links(child)


def _backlinks(node: _Node, text: str) -> list[_Node]:
    out = list(_leading_links(node, text))
    for a in _all_links(node):
        if a not in out and is_note(attrs_of(a, text), BACKLINK_TOKENS):
            out.append(a)
    return out


def _removal_span(tr: _Node, link: _Node, text: str) -> tuple[int, int]:
    """The link, widened to the inline wrappers that hold nothing else
    (`<span class="footnoteNum"><a>*</a> </span>`)."""

    def path(n: _Node, target: _Node) -> list[_Node] | None:
        for kind, child in n.children:
            if kind != "node":
                continue
            if child is target:
                return [child]
            sub = path(child, target)
            if sub is not None:
                return [child] + sub
        return None

    chain = path(tr, link) or [link]
    span = (link.raw_start, link.end_end)
    for wrapper in reversed(chain[:-1]):
        others = [
            c for kind, c in wrapper.children
            if not (kind == "node" and c is chain[chain.index(wrapper) + 1])
            and not (kind == "text" and not text[c[0]:c[1]].strip(_WS))
        ]
        if others or wrapper.end_end is None:
            break
        span = (wrapper.raw_start, wrapper.end_end)
    return span


def backlink_removals(src: _Node, tr: _Node, text: str) -> list[tuple[int, int]]:
    hrefs = {attrs_of(a, text).get("href") for a in _backlinks(src, text)}
    hrefs.discard(None)
    out = []
    for a in _backlinks(tr, text):
        if attrs_of(a, text).get("href") in hrefs and a.end_end is not None:
            out.append(_removal_span(tr, a, text))
    return out


def _set_lang(raw: str, use_xml_lang: bool) -> str | None:
    """Give a start tag lang (and xml:lang) = zh. None when already zh."""
    m = re.match(r"(<[^\s/>]+)(.*?)(/?>)\Z", raw, re.S)
    if not m:
        return None
    head, attrs, close = m.groups()
    changed = False
    for name in ("lang", "xml:lang") if use_xml_lang else ("lang",):
        am = re.search(rf"""(?<![\w:-]){re.escape(name)}\s*=\s*(["'])(.*?)\1""", attrs, re.S)
        if am is None:
            attrs = attrs.rstrip() + f' {name}="{TARGET_LANG}"'
            changed = True
        elif not am.group(2).lower().startswith(TARGET_LANG):
            attrs = attrs[:am.start(2)] + TARGET_LANG + attrs[am.end(2):]
            changed = True
    return head + attrs + close if changed else None


def _merged_div(tr: _Node, text: str, removals, use_xml_lang: bool) -> str:
    attrs = attrs_of(tr, text)
    lang = attrs.get("lang", "")
    if not lang.lower().startswith(TARGET_LANG):
        lang = TARGET_LANG
    xml_lang = attrs.get("xml:lang", "")
    if not xml_lang.lower().startswith(TARGET_LANG):
        xml_lang = lang
    parts = [f'<div lang="{lang}"']
    if use_xml_lang or "xml:lang" in attrs:
        parts.append(f' xml:lang="{xml_lang}"')
    if "zh-translation" in attrs.get("class", "").split():
        parts.append(' class="zh-translation"')
    if attrs.get("id"):
        parts.append(f' id="{attrs["id"]}"')
    inner_start, inner_end = tr.raw_end, tr.end_start
    inner = text[inner_start:inner_end]
    for s, e in sorted(removals, reverse=True):
        inner = inner[:s - inner_start] + inner[e - inner_start:]
    return "".join(parts) + ">" + inner + "</div>"


def dedupe_number(stream) -> tuple[int, int] | None:
    """Offsets of ` 1.` to drop in `1. Introduction 1. 引言`, or None."""
    s = "".join(ch for ch, _ in stream)
    m = re.match(r"\s*(\S+)\s", s)
    if not m or not _NUMBER.fullmatch(m.group(1)):
        return None
    num = m.group(1)
    # The last repeat before the first CJK character is where the translation
    # starts; an earlier one is part of the English title.
    found = None
    for dm in re.finditer(r"\s+" + re.escape(num) + r"(?=\s)", s[m.end() - 1:]):
        a, b = m.end() - 1 + dm.start(), m.end() - 1 + dm.end()
        if any(is_cjk_script(c) for c in s[:a]):
            break
        if any(is_cjk_script(c) for c in s[b:]):
            found = (a, b)
    if found is None:
        return None
    offs = [o for _, o in stream[found[0]:found[1]]]
    if None in offs or offs != list(range(offs[0], offs[0] + len(offs))):
        return None
    return offs[0], offs[-1] + 1


def _toc_entries(node: _Node, in_nav: bool = False):
    for kind, child in node.children:
        if kind != "node":
            continue
        if (child.tag == "a" and in_nav) or (child.tag == "text" and node.tag == "navlabel"):
            yield child
        else:
            yield from _toc_entries(child, in_nav or child.tag == "nav")


def plan_edits(text: str) -> tuple[list[tuple[int, int, str]], collections.Counter]:
    """The point edits for one document: (start, end, replacement)."""
    tree = _Tree(text)
    stats: collections.Counter = collections.Counter()
    edits: list[tuple[int, int, str]] = []
    use_xml_lang = re.search(r"(?<![\w:-])xml:lang\s*=", text) is not None

    for src, tr, ancestors in pairs(tree.root, text):
        src_attrs = attrs_of(src, text)
        in_note = is_note(src_attrs) or any(
            is_note(attrs_of(a, text)) for a in ancestors)
        removals = backlink_removals(src, tr, text) if in_note else []
        if src.tag == "li" or is_note(src_attrs):
            div = _merged_div(tr, text, removals, use_xml_lang)
            edits.append((src.end_start, src.end_start, div))
            edits.append((src.end_end, tr.end_end, ""))
            stats["notes merged" if is_note(src_attrs) else "list items merged"] += 1
            stats["backlinks dropped"] += len(removals)
            continue
        new_tag = _set_lang(text[tr.raw_start:tr.raw_end], use_xml_lang)
        if new_tag is not None:
            edits.append((tr.raw_start, tr.raw_end, new_tag))
            stats["lang set"] += 1
        for s, e in removals:
            edits.append((s, e, ""))
        stats["backlinks dropped"] += len(removals)

    for entry in _toc_entries(tree.root):
        span = dedupe_number(_stream(entry, text))
        if span is not None:
            edits.append((span[0], span[1], ""))
            stats["toc numbers deduplicated"] += 1
    return edits, stats


def process_document(text: str) -> tuple[str, collections.Counter]:
    edits, stats = plan_edits(text)
    for s, e, repl in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        text = text[:s] + repl + text[e:]
    return text, stats


# ── EPUB ────────────────────────────────────────────────────────────────

_SUFFIXES = (".xhtml", ".html", ".htm", ".ncx")


def fix_epub(src: Path, dst: Path | None) -> collections.Counter:
    """Fix every content document of `src` into `dst` (None: count only).

    The zip is rewritten entry by entry in the source order with the source
    timestamps, `mimetype` first and stored, so untouched entries are
    byte-identical and the output is deterministic.
    """
    total: collections.Counter = collections.Counter()
    with zipfile.ZipFile(src) as zin:
        infos = zin.infolist()
        payload = {i.filename: zin.read(i) for i in infos}
    for name, data in payload.items():
        if not name.lower().endswith(_SUFFIXES):
            continue
        text = data.decode("utf-8")
        new_text, stats = process_document(text)
        total.update(stats)
        if new_text != text:
            payload[name] = new_text.encode("utf-8")
            total["documents changed"] += 1
    if dst is None:
        return total
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to overwrite the input; give another output path")
    dst.parent.mkdir(parents=True, exist_ok=True)
    infos.sort(key=lambda i: i.filename != "mimetype")
    with zipfile.ZipFile(dst, "w") as zout:
        for info in infos:
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.external_attr = info.external_attr
            zi.internal_attr = info.internal_attr
            zi.create_system = info.create_system
            zi.comment = info.comment
            ct = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(zi, payload[info.filename], compress_type=ct)
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="bilingual EPUB (append-block)")
    ap.add_argument("dst", type=Path, nargs="?", help="output EPUB")
    ap.add_argument("--stats", action="store_true", help="count only, write nothing")
    args = ap.parse_args(argv)
    if not args.stats and args.dst is None:
        ap.error("an output path is required (or --stats)")
    total = fix_epub(args.src, None if args.stats else args.dst)
    for key in ("lang set", "list items merged", "notes merged",
                "backlinks dropped", "toc numbers deduplicated", "documents changed"):
        print(f"  {key:26} {total[key]}")
    if not args.stats:
        print(f"  -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
