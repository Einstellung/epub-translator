"""Cut an EPUB down to a subset of its spine (e.g. "preface + chapters 1-3").

Why this exists
---------------
Translating a whole 300-page book costs real money and hours of wall clock, so
for smoke tests / sample bilingual builds you want a small but *valid* EPUB that
contains only the first few chapters. Deleting the unwanted XHTML files is the
easy 10%; the other 90% is everything that keeps pointing at them. This tool
handles the parts that actually break readers and downstream tooling:

* **OPF surgery is textual, not a re-serialisation.** The `<metadata>` block,
  the `prefix=`/`version=` attributes, `linear="no"` on an `<itemref>`, custom
  `properties=` — all of it is preserved byte-for-byte, because the OPF is only
  *parsed* to decide what to drop and then *edited* by excising the exact
  `<item>` / `<itemref>` / `<reference>` substrings. Re-emitting the XML through
  ElementTree would reorder attributes, mangle namespace prefixes and drop the
  original quoting style.
* **Both tables of contents.** EPUB3 books ship an XHTML nav document (the
  manifest item with `properties="nav"`, usually *not* in the spine) *and*, for
  EPUB2 readers, a `toc.ncx`. Both are pruned. Nav `<li>`s that pointed at a
  dropped document are removed whole — nested sub-lists and all — instead of
  leaving an empty `<li></li>` shell that renders as a blank TOC row. NCX
  `<navPoint>`s are dropped the same way and then `playOrder` is renumbered
  1..N in document order, since the spec requires it to be gapless.
* **Resource pruning is what actually shrinks the file.** Images are ~99% of a
  technical EPUB's bytes. The kept documents are scanned for `<img src>`,
  `srcset` candidate lists, SVG `<image xlink:href>`, `<a href>` pointing at a
  binary, inline `style="...url()..."`, `<source>`, `<link href>` and
  `<script src>`; every stylesheet reachable from them is scanned for `url()`
  too (fonts and background images). Anything with an image media type that is
  not in that union is dropped. The cover image is force-kept even though no
  body text references it (it is reached only via `<meta name="cover">` /
  `properties="cover-image"`, and losing it gives you a coverless book).
* **Dangling cross-references are degraded, not left to rot.** Chapter 2 says
  "see Chapter 9" with an `<a href="ch09.html#...">`. Pointing at a file that no
  longer exists makes epubcheck fail and some readers throw. Such anchors are
  unwrapped to their visible text (`<a href="ch09.html">Chapter 9</a>` ->
  `Chapter 9`). Links to *kept* documents, in-page `#fragment` links and
  external `http(s):`/`mailto:` links are left untouched.
* **ZIP layout matters.** `mimetype` must be the first entry and stored
  uncompressed, or the file is not a valid EPUB (OCF checks the magic bytes at
  a fixed offset). Everything else is deflated. Original entry order is
  otherwise preserved.

Usage
-----
    uv run python truncate_epub.py book.epub --keep-spine 0-8 -o small.epub
    uv run python truncate_epub.py book.epub --keep-spine 0-3,6,7 -o small.epub
    uv run python truncate_epub.py book.epub --keep-idref cover,preface-id534 -o small.epub
    uv run python truncate_epub.py book.epub --list          # just show the spine

Non-image resources (CSS, fonts) are always kept: they are small, and a font
that is only referenced from a `@font-face` in a stylesheet variant is easy to
lose by accident. Pass `--prune-all` to prune unreferenced non-image resources
too.
"""

import argparse
import posixpath
import re
import sys
import zipfile
from urllib.parse import unquote, urldefrag
from xml.etree import ElementTree as ET

OPF_NS = "http://www.idpf.org/2007/opf"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

XHTML_MEDIA = {"application/xhtml+xml", "text/html"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def resolve(base_path: str, href: str) -> str:
    """Resolve an href found inside `base_path` to a zip-relative path."""
    href = urldefrag(href)[0]
    if not href:
        return ""
    href = unquote(href)
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_path), href))


def is_external(href: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", href)) or href.startswith("//")


def parse_ranges(spec: str) -> set[int]:
    """'0-8,12' -> {0,...,8,12}"""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


# --------------------------------------------------------------------------- #
# OPF reading (parse to decide) and editing (text surgery to preserve bytes)
# --------------------------------------------------------------------------- #


def find_opf(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    el = root.find(f".//{{{CONTAINER_NS}}}rootfile")
    if el is None or not el.get("full-path"):
        raise SystemExit("container.xml has no rootfile/@full-path")
    return el.get("full-path")


class Opf:
    def __init__(self, text: str, path: str):
        self.text = text
        self.path = path
        root = ET.fromstring(text)
        self.items: list[dict] = []
        for it in root.iter(f"{{{OPF_NS}}}item"):
            self.items.append(
                {
                    "id": it.get("id"),
                    "href": it.get("href"),
                    "media": it.get("media-type", ""),
                    "properties": it.get("properties", "") or "",
                    "path": resolve(path, it.get("href", "")),
                }
            )
        self.by_id = {i["id"]: i for i in self.items}
        self.spine: list[str] = [
            r.get("idref") for r in root.iter(f"{{{OPF_NS}}}itemref")
        ]
        # <meta name="cover" content="ID"/>
        self.cover_ids: set[str] = set()
        for m in root.iter(f"{{{OPF_NS}}}meta"):
            if (m.get("name") or "").lower() == "cover" and m.get("content"):
                self.cover_ids.add(m.get("content"))
        for i in self.items:
            if "cover-image" in i["properties"].split():
                self.cover_ids.add(i["id"])
        self.nav_ids: set[str] = {
            i["id"] for i in self.items if "nav" in i["properties"].split()
        }
        self.ncx_ids: set[str] = {
            i["id"] for i in self.items if i["media"] == "application/x-dtbncx+xml"
        }


_ITEM_RE = re.compile(r"<item\b[^>]*?(?:/>|>.*?</item\s*>)", re.S)
_ITEMREF_RE = re.compile(r"<itemref\b[^>]*?(?:/>|>.*?</itemref\s*>)", re.S)
_REFERENCE_RE = re.compile(r"<reference\b[^>]*?(?:/>|>.*?</reference\s*>)", re.S)
_ATTR = lambda tag, name: (  # noqa: E731
    (re.search(rf'\b{name}\s*=\s*"([^"]*)"', tag) or re.search(rf"\b{name}\s*=\s*'([^']*)'", tag))
)


def _attr(tag: str, name: str) -> str | None:
    m = _ATTR(tag, name)
    return m.group(1) if m else None


def rewrite_opf(opf: Opf, keep_ids: set[str], kept_paths: set[str]) -> str:
    """Excise dropped <item>/<itemref>/<reference> elements, keep the rest verbatim."""
    text = opf.text

    def drop_item(m: re.Match) -> str:
        return "" if _attr(m.group(0), "id") not in keep_ids else m.group(0)

    def drop_itemref(m: re.Match) -> str:
        return "" if _attr(m.group(0), "idref") not in keep_ids else m.group(0)

    def drop_reference(m: re.Match) -> str:
        href = _attr(m.group(0), "href") or ""
        if is_external(href):
            return m.group(0)
        return m.group(0) if resolve(opf.path, href) in kept_paths else ""

    # scope each regex to its own section so an <item> in <metadata> is untouched
    def in_section(open_tag: str, close_tag: str, fn, rx) -> None:
        nonlocal text
        a = text.find(open_tag)
        b = text.find(close_tag)
        if a == -1 or b == -1:
            return
        text = text[:a] + rx.sub(fn, text[a:b]) + text[b:]

    in_section("<manifest", "</manifest>", drop_item, _ITEM_RE)
    in_section("<spine", "</spine>", drop_itemref, _ITEMREF_RE)
    in_section("<guide", "</guide>", drop_reference, _REFERENCE_RE)
    # an emptied <guide> is invalid (needs >=1 reference) -> remove it entirely
    text = re.sub(r"<guide\b[^>]*>\s*</guide\s*>", "", text)
    return text


# --------------------------------------------------------------------------- #
# reference scanning
# --------------------------------------------------------------------------- #

_SCAN_PATTERNS = [
    re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<image\b[^>]*?\b(?:xlink:)?href\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<source\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<link\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<script\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<audio\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<video\b[^>]*?\b(?:src|poster)\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
    re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", re.I | re.S),
]
_SRCSET_RE = re.compile(r"\bsrcset\s*=\s*[\"']([^\"']+)[\"']", re.I)
_URL_RE = re.compile(r"url\(\s*[\"']?([^\"')]+)[\"']?\s*\)", re.I)


def scan_refs(doc_path: str, text: str, is_css: bool) -> set[str]:
    """All zip-relative paths referenced from one document."""
    out: set[str] = set()

    def add(href: str) -> None:
        if not href or href.startswith("#") or is_external(href):
            return
        p = resolve(doc_path, href)
        if p:
            out.add(p)

    if is_css:
        for m in _URL_RE.finditer(text):
            add(m.group(1).strip("\"'"))
        return out

    for rx in _SCAN_PATTERNS:
        for m in rx.finditer(text):
            add(m.group(1))
    for m in _SRCSET_RE.finditer(text):
        for cand in m.group(1).split(","):
            cand = cand.strip().split()  # "img.png 2x" -> "img.png"
            if cand:
                add(cand[0])
    for m in _URL_RE.finditer(text):  # inline style="background:url(...)"
        add(m.group(1).strip("\"'"))
    return out


# --------------------------------------------------------------------------- #
# dangling-link degradation
# --------------------------------------------------------------------------- #

# NOTE: `(?<!/)` skips SELF-CLOSING `<a ... id="idx1"/>` tags (O'Reilly books use
# them as index-term markers). Without it the "opening" half of such a tag pairs
# with the *next* `</a>`, swallowing a real link and leaving it undegraded.
_A_RE = re.compile(r"<a\b([^>]*?)(?<!/)>(.*?)</a\s*>", re.I | re.S)


def degrade_links(doc_path: str, text: str, kept_paths: set[str]) -> tuple[str, int]:
    """Unwrap <a> elements whose target document no longer exists."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        href = _attr("<a" + m.group(1) + ">", "href")
        if href is None or href.startswith("#") or is_external(href):
            return m.group(0)
        target = resolve(doc_path, href)
        if not target or target in kept_paths:
            return m.group(0)
        n += 1
        return m.group(2)

    return _A_RE.sub(repl, text), n


# --------------------------------------------------------------------------- #
# nested-block pruning shared by the nav document and the NCX
# --------------------------------------------------------------------------- #


def _blocks(text: str, tag: str, start: int, end: int) -> list[tuple[int, int]]:
    """Top-level (outermost) <tag>...</tag> spans inside text[start:end]."""
    open_rx = re.compile(rf"<{tag}\b", re.I)
    close_rx = re.compile(rf"</{tag}\s*>", re.I)
    spans: list[tuple[int, int]] = []
    i = start
    while True:
        m = open_rx.search(text, i, end)
        if not m:
            return spans
        depth = 0
        j = m.start()
        while j < end:
            om = open_rx.search(text, j, end)
            cm = close_rx.search(text, j, end)
            if cm is None:
                return spans  # malformed; give up on the rest
            if om is not None and om.start() < cm.start():
                depth += 1
                j = om.end()
            else:
                depth -= 1
                j = cm.end()
                if depth == 0:
                    spans.append((m.start(), j))
                    i = j
                    break
        else:
            return spans


def prune_nested(
    text: str,
    tag: str,
    target_of,
    dead,
    start: int = 0,
    end: int | None = None,
) -> tuple[str, int]:
    """Recursively drop <tag> blocks whose own link points at a dead document.

    `target_of(block)` returns the block's own href (or None). Blocks that keep
    no link at all after pruning their children are dropped too, so no empty
    shells survive.
    """
    if end is None:
        end = len(text)
    removed = 0
    spans = _blocks(text, tag, start, end)
    # rebuild right-to-left so earlier offsets stay valid
    for a, b in reversed(spans):
        block = text[a:b]
        href = target_of(block)
        if href is not None and dead(href):
            text = text[:a] + text[b:]
            removed += 1
            continue
        # recurse into children of this block
        inner_start = block.find(">") + 1
        new_block, sub = prune_nested(
            block, tag, target_of, dead, inner_start, len(block)
        )
        removed += sub
        if not re.search(r"<(?:a|content)\b", new_block, re.I):
            text = text[:a] + text[b:]
            removed += 1
            continue
        text = text[:a] + new_block + text[b:]
    return text, removed


def prune_nav(nav_path: str, text: str, kept_paths: set[str]) -> tuple[str, int]:
    def target_of(block: str) -> str | None:
        m = re.search(r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", block, re.I | re.S)
        return m.group(1) if m else None

    def dead(href: str) -> bool:
        if href.startswith("#") or is_external(href):
            return False
        return resolve(nav_path, href) not in kept_paths

    return prune_nested(text, "li", target_of, dead)


def prune_ncx(ncx_path: str, text: str, kept_paths: set[str]) -> tuple[str, int, int]:
    def target_of(block: str) -> str | None:
        m = re.search(
            r"<content\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", block, re.I | re.S
        )
        return m.group(1) if m else None

    def dead(href: str) -> bool:
        if href.startswith("#") or is_external(href):
            return False
        return resolve(ncx_path, href) not in kept_paths

    text, removed = prune_nested(text, "navPoint", target_of, dead)
    text, removed_pt = prune_nested(text, "pageTarget", target_of, dead)
    removed += removed_pt

    # playOrder must be a gapless 1..N sequence in document order
    counter = [0]

    def renumber(m: re.Match) -> str:
        counter[0] += 1
        return f'playOrder="{counter[0]}"'

    text = re.sub(r'playOrder\s*=\s*"\d+"', renumber, text)
    return text, removed, counter[0]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Truncate an EPUB to a subset of its spine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="source EPUB (never modified)")
    ap.add_argument("-o", "--output", help="destination EPUB")
    ap.add_argument(
        "--keep-spine",
        help="spine indices to keep, e.g. '0-8' or '0-3,6,7' (0-based)",
    )
    ap.add_argument(
        "--keep-idref",
        help="comma-separated spine idrefs to keep instead of indices",
    )
    ap.add_argument(
        "--prune-all",
        action="store_true",
        help="also drop unreferenced non-image resources (CSS, fonts)",
    )
    ap.add_argument("--list", action="store_true", help="print the spine and exit")
    args = ap.parse_args()

    src = zipfile.ZipFile(args.input)
    opf_path = find_opf(src)
    opf = Opf(src.read(opf_path).decode("utf-8"), opf_path)
    sizes = {i.filename: i.file_size for i in src.infolist()}

    if args.list:
        print(f"OPF: {opf_path}   spine: {len(opf.spine)} items")
        for n, idref in enumerate(opf.spine):
            it = opf.by_id.get(idref, {})
            print(
                f"{n:3d}  {idref:<22} {it.get('href','?'):<24} "
                f"{sizes.get(it.get('path',''), 0):>8}"
            )
        return 0

    if not args.output:
        ap.error("-o/--output is required (unless --list)")
    if bool(args.keep_spine) == bool(args.keep_idref):
        ap.error("give exactly one of --keep-spine / --keep-idref")

    if args.keep_spine:
        want = parse_ranges(args.keep_spine)
        bad = want - set(range(len(opf.spine)))
        if bad:
            ap.error(f"spine index out of range: {sorted(bad)}")
        keep_spine = [r for n, r in enumerate(opf.spine) if n in want]
    else:
        want_ids = [s.strip() for s in args.keep_idref.split(",") if s.strip()]
        unknown = [s for s in want_ids if s not in opf.spine]
        if unknown:
            ap.error(f"idref not in spine: {unknown}")
        keep_spine = [r for r in opf.spine if r in want_ids]  # keep spine order

    dropped_spine = [r for r in opf.spine if r not in set(keep_spine)]
    print(f"source     : {args.input}")
    print(f"OPF        : {opf_path}")
    print(f"spine      : keep {len(keep_spine)} / {len(opf.spine)} documents")
    for idref in keep_spine:
        print(f"  keep  {idref:<22} {opf.by_id[idref]['href']}")
    print(f"  drop  {len(dropped_spine)}: "
          f"{', '.join(opf.by_id[r]['href'] for r in dropped_spine)}")

    # ---- 1. which documents survive -------------------------------------- #
    keep_ids = set(keep_spine) | opf.nav_ids | opf.ncx_ids | opf.cover_ids
    dropped_paths = {opf.by_id[r]["path"] for r in dropped_spine if r in opf.by_id}

    # ---- 2. collect every resource the kept documents reference ----------- #
    #     scanned on the ORIGINAL text, so an <a href="fig.png"> keeps its image
    raw_text: dict[str, str] = {}
    referenced: set[str] = set()
    frontier = []
    for idref in keep_spine:
        p = opf.by_id[idref]["path"]
        if opf.by_id[idref]["media"] not in XHTML_MEDIA:
            continue
        raw_text[p] = src.read(p).decode("utf-8")
        frontier.append((p, raw_text[p], False))
    # the nav document's own resources (its stylesheet) count too
    for i in opf.nav_ids:
        if i in opf.by_id:
            p = opf.by_id[i]["path"]
            frontier.append((p, src.read(p).decode("utf-8"), False))

    seen_css: set[str] = set()
    while frontier:
        p, text, is_css = frontier.pop()
        for ref in scan_refs(p, text, is_css):
            referenced.add(ref)
            if ref.lower().endswith(".css") and ref not in seen_css and ref in sizes:
                seen_css.add(ref)
                frontier.append((ref, src.read(ref).decode("utf-8"), True))

    # ---- 3. decide the final manifest ------------------------------------ #
    final_ids: set[str] = set()
    for it in opf.items:
        i = it["id"]
        if i in keep_ids:
            final_ids.add(i)
        elif it["path"] in dropped_paths:
            continue  # a dropped spine document is never resurrected by a link
        elif it["path"] in referenced:
            final_ids.add(i)
        elif not it["media"].startswith("image/") and not args.prune_all:
            final_ids.add(i)  # keep CSS/fonts by default

    kept_paths = {opf.by_id[i]["path"] for i in final_ids}
    kept_paths.add(opf_path)

    # ---- 4. degrade links that now point at nothing ----------------------- #
    new_text: dict[str, str] = {}
    dangling_total = 0
    dangling_by_doc: dict[str, int] = {}
    for p, raw in raw_text.items():
        fixed, n = degrade_links(p, raw, kept_paths)
        if n:
            dangling_by_doc[posixpath.basename(p)] = n
            dangling_total += n
        new_text[p] = fixed

    n_img_before = sum(1 for it in opf.items if it["media"].startswith("image/"))
    n_img_after = sum(
        1 for it in opf.items if it["id"] in final_ids and it["media"].startswith("image/")
    )
    print(f"manifest   : keep {len(final_ids)} / {len(opf.items)} items "
          f"(images {n_img_after} / {n_img_before}, "
          f"dropped {n_img_before - n_img_after})")
    print(f"dangling   : unwrapped {dangling_total} <a> in "
          f"{len(dangling_by_doc)} documents "
          f"({', '.join(f'{k}:{v}' for k, v in dangling_by_doc.items())})")

    # ---- 5. rewrite OPF / nav / ncx --------------------------------------- #
    new_text[opf_path] = rewrite_opf(opf, final_ids, kept_paths)

    for i in opf.nav_ids & final_ids:
        p = opf.by_id[i]["path"]
        text = src.read(p).decode("utf-8")
        text, removed = prune_nav(p, text, kept_paths)
        new_text[p] = text
        print(f"nav        : {opf.by_id[i]['href']} -> removed {removed} <li>")
    for i in opf.ncx_ids & final_ids:
        p = opf.by_id[i]["path"]
        text = src.read(p).decode("utf-8")
        text, removed, total = prune_ncx(p, text, kept_paths)
        new_text[p] = text
        print(f"ncx        : {opf.by_id[i]['href']} -> removed {removed} navPoint, "
              f"renumbered playOrder 1..{total}")

    # ---- 6. write the zip: mimetype first, stored ------------------------- #
    keep_entry = lambda n: (  # noqa: E731
        n == "mimetype" or n.startswith("META-INF/") or n in kept_paths
    )
    order = [i for i in src.infolist() if keep_entry(i.filename)]
    order.sort(key=lambda i: i.filename != "mimetype")  # mimetype first

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as out:
        for info in order:
            name = info.filename
            if name in new_text:
                data = new_text[name].encode("utf-8")
            else:
                data = src.read(name)
            if name == "mimetype":
                out.writestr(
                    zipfile.ZipInfo("mimetype"), b"application/epub+zip",
                    compress_type=zipfile.ZIP_STORED,
                )
            else:
                out.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)

    import os

    before, after = os.path.getsize(args.input), os.path.getsize(args.output)
    print(f"entries    : {len(src.namelist())} -> {len(order)} "
          f"(dropped {len(src.namelist()) - len(order)})")
    print(f"bytes      : {before:,} -> {after:,} "
          f"({after / before:.1%} of original)")
    print(f"wrote      : {args.output}")
    src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
