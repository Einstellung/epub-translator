r"""Detect a book's front matter so the translator can skip it by default.

Why this exists
---------------
Cover, praise/dedication, title page, copyright page, table of contents,
preface and the bare "Part I" divider pages are not prose anyone wants a
bilingual rendering of — translating them burns tokens, and an LLM rewriting a
copyright notice or an ISBN block is actively bad. Until now `translate_book.py`
only had the hand-maintained `exclude_spine_ids` list, so every new book needed
a human to read its OPF first. `skip_front_matter: true` (the default) makes it
automatic: this module returns the spine ids to drop, and `translate_book.py`
unions them with whatever the config lists by hand.

Design: layered, authoritative first, and *never* silent
-------------------------------------------------------
A false negative (one extra page translated) is harmless; a false positive
(a real chapter silently skipped) is a serious defect. So the rules are ordered
by how much the source file actually *tells* us, and the outcome of every single
spine document is printed with the layer that decided it:

  L1  nav `landmarks` — `epub:type="cover" / "toc" / "titlepage" / ...` label
      individual documents, and `epub:type="bodymatter"` names where the body
      starts, which is the single most useful signal a book can carry.
  L2  OPF `<guide>` — `type="cover" / "title-page" / "copyright-page" / ...`,
      and `type="text"` for "start of text".

      Both "the body starts here" pointers are *only* trusted when they do not
      aim at a document that is itself specifically classified front matter,
      because shipped books get this wrong in both directions: O'Reilly's
      HTMLBook emits `<reference type="text" href="titlepage01.html"/>` (the
      title page would become the body, and nothing before chapter 1 would be
      skipped), while Penguin Random House aims `landmarks bodymatter` at the
      dedication (the dedication, epigraph, contents and preface would all be
      translated). A pointer into declared front matter is reported as
      DISTRUSTED and the next layer decides.
  L3  the document's own `epub:type` (`titlepage`, `copyright-page`,
      `dedication`, `preface`, `toc`, ...), read off `<body>` and the first few
      structural elements inside it, so a stray `epub:type="note"` on a footnote
      deep in a chapter cannot vote.
  L4  spine id / href naming patterns (`cover`, `halftitle`, `praise`,
      `alsoby`, `colophon`, ...), size-capped so a long document can never be
      classified on its filename alone.
  L5  size heuristics: a `part`-typed or `part\d+`-named document that is a few
      hundred bytes of nothing but a heading is a divider page, not a chapter;
      and a generically `frontmatter`-typed document that is tiny is a card
      page ("Also by ...", a dedication).

Two hard safety rails on top of the layers:

  * **Prefix only.** Apart from the tiny part-divider rule and the contents
    page (see below), a document is excluded only if it sits *before* the body
    start — i.e. front matter is a contiguous prefix of the spine. A chapter
    halfway through the book called `the_cover_story.xhtml` is therefore
    untouchable, and "the body begins at the first document we could not
    classify" is the fallback when the book ships no landmarks at all.
  * **Generic `frontmatter` never excludes on its own.** Publishers routinely
    tag *everything* before chapter one with `epub:type="frontmatter"` — in
    "Reentry" that includes a 7,000-word narrative Prologue. Only specific
    categories (cover, titlepage, copyright, dedication, toc, preface, ...)
    count, and a large unclassified document before the body start is kept.

The contents page: never translated, wherever it sits
-----------------------------------------------------
A table of contents is a list of the chapter titles, which the translated book
re-renders from the chapters themselves; translating it is pure waste. So it is
the one category excluded *regardless of position* — it does not have to sit in
the front-matter prefix, and a book that buries its contents page behind the
declared body start still gets it skipped.

That makes the "is this really a ToC?" test the load-bearing part, because the
claim is checked against the document instead of believed. "Reentry" ships
`<reference type="toc" href="Prologue.xhtml"/>` — its 7,000-word narrative
Prologue — and that one bad attribute would otherwise both skip the Prologue
*and*, by contradicting the landmarks, poison the body-start detection. The
test is "short *and* link-dense": see TOC_MAX_TEXT / TOC_MIN_LINK_RATIO.

Run it standalone to audit a book without translating anything:

    uv run python front_matter.py "input/Some Book.epub"
"""

from __future__ import annotations

import posixpath
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- categories

# epub:type values that identify a *specific* kind of front matter. The generic
# "frontmatter" is deliberately absent: see the module docstring.
FRONT_EPUB_TYPES = {
    "cover": "cover",
    "titlepage": "title page",
    "halftitlepage": "half title page",
    "fulltitlepage": "title page",
    "copyright-page": "copyright page",
    "colophon": "colophon",
    "imprint": "imprint",
    "dedication": "dedication",
    "epigraph": "epigraph",
    "toc": "table of contents",
    "toc-brief": "table of contents",
    "landmarks": "landmarks",
    "preface": "preface",
    "foreword": "foreword",
    "acknowledgments": "acknowledgments",  # only counts inside the front prefix
    "seriespage": "series page",
    "other-credits": "credits page",
    "contributors": "contributors page",
    "revision-history": "revision history",
}

# <guide> reference types (EPUB 2 / OPF 2) that identify front matter.
FRONT_GUIDE_TYPES = {
    "cover": "cover",
    "title-page": "title page",
    "titlepage": "title page",
    "copyright-page": "copyright page",
    "copyright": "copyright page",
    "dedication": "dedication",
    "epigraph": "epigraph",
    "acknowledgements": "acknowledgments",
    "acknowledgments": "acknowledgments",
    "toc": "table of contents",
    "preface": "preface",
    "foreword": "foreword",
    "colophon": "colophon",
}

# spine-id / filename patterns. Matched as whole words against the id and the
# href stem, and only applied to short documents (see NAME_MAX_TEXT).
NAME_PATTERNS: list[tuple[str, str]] = [
    (r"cover", "cover"),
    (r"coverpage|cover[-_]?image", "cover"),
    (r"title[-_]?page|titlepage|half[-_]?title|halftitle|frontcover", "title page"),
    (r"copyright|imprint|colophon|legal|impressum", "copyright page"),
    (r"dedication|epigraph", "dedication / epigraph"),
    (r"praise|acclaim|advance[-_]?praise|blurbs?", "praise page"),
    (r"also[-_]?by|alsoby|other[-_]?books|by[-_]?the[-_]?same", "also-by page"),
    (r"series[-_]?page|about[-_]?the[-_]?series|ad[-_]?card|bookad", "series / ad page"),
    (r"toc|contents|tableofcontents|table[-_]?of[-_]?contents", "table of contents"),
    (r"preface|foreword|prf|pref|frontmatter|front[-_]?matter", "preface / foreword"),
    # "acknowledgments" is already a FRONT_EPUB_TYPES category, but no pattern
    # here matched it, so a book that ships no epub:type at all (Manning's
    # retail EPUBs) ended its front-matter prefix on the acknowledgments page
    # and translated everything after it. Trade books put acknowledgments in the
    # *back* — measured at spine #17-#23 in Reentry, Why We Remember, The
    # Experience Machine and The President's Book of Secrets — and the
    # prefix-only rail is what keeps those untouched.
    (r"acknowledge?ments?", "acknowledgments"),
]

# Names that are only safe on a *tiny* page. Two kinds live here:
#
#   * publisher shorthand — Penguin Random House & friends name files
#     `..._cvi_r1`, `_tp_`, `_cop_`, `_ded_`, `_epi_`; these abbreviations are
#     too short to be unambiguous (`epi` is equally "epigraph" and "epilogue");
#   * plain English words that are *also* ordinary vocabulary — `title`.
#
# Both would be reckless in NAME_PATTERNS, which allows up to NAME_MAX_TEXT
# (20,000) characters. Here they only apply to documents small enough
# (ABBREV_MAX_TEXT, 1,000 chars) that no prose could be hiding in them.
#
# Bare `title`: Manning's retail EPUBs call the title page `title.xhtml`
# (measured in "Build a Reasoning Model": 220 chars) while NAME_PATTERNS only
# knows `titlepage` / `title_page` / `halftitle`. That book carries no
# epub:type, no role, no nav document and therefore no landmarks, and a
# `<guide>` with a single `type="cover"` entry — L1/L2/L3 are all silent, so the
# filename is the only signal there is. Missing it did not cost one page: the
# prefix scan stopped dead at #1 and let the copyright page, dedication,
# contents, preface and everything else through. `title` is not in
# NAME_PATTERNS because a real chapter can easily contain the word ("The Title
# Fight"); at 1,000 chars it cannot.
ABBREV_PATTERNS: list[tuple[str, str]] = [
    (r"cvi|cvr|cov", "cover"),
    (r"tp|htp|hftp|tpg|title", "title page"),
    (r"cop|copy|cpr", "copyright page"),
    (r"ded", "dedication"),
    (r"epi|epg", "epigraph"),
    (r"con|cnt", "table of contents"),
]

# Reference apparatus: an index, glossary or bibliography is *naturally*
# link-dense — measured 0.29-0.86 across this corpus, i.e. right in genuine-ToC
# territory — so "short and link-dense" cannot tell one from a contents page.
# A layer that calls one of these the table of contents is therefore not
# believed, at any length.
#
# The veto only ever moves a document from SKIP to keep, so over-matching it is
# the harmless direction; that is why the name half can afford to be broad.
REFERENCE_APPARATUS_TYPES = {"index", "glossary", "bibliography"}
REFERENCE_APPARATUS_NAMES = r"index|idx|glossary|bibliography|biblio"

# Document types that must never be treated as front matter, whatever else says.
BODY_EPUB_TYPES = {"bodymatter", "chapter", "prologue", "introduction", "epilogue"}

# DPUB-ARIA spells the same vocabulary with a `doc-` prefix (`role="doc-toc"`),
# and shipped books mix the two spellings *within one book*: "Why We Remember"
# tags its dedication `epub:type="dedication" role="doc-dedication"` but its
# contents page only `role="doc-toc"`, so the contents page went unrecognised
# while the dedication was caught. Both spellings are therefore folded into one
# canonical form before any lookup.
#
# Safe by construction, because the prefix is a pure spelling difference and the
# two vocabularies agree on meaning: every `doc-*` role that names body text
# (`doc-chapter`, `doc-introduction`, `doc-prologue`, `doc-epilogue`) folds onto
# a BODY_EPUB_TYPES entry, which makes the *keep* side stronger too; the
# front-matter roles (`doc-toc`, `doc-cover`, `doc-colophon`, `doc-epigraph`,
# `doc-foreword`, `doc-preface`, ...) fold onto the right FRONT_EPUB_TYPES
# entry; and the rest (`doc-index`, `doc-glossary`, `doc-endnotes`,
# `doc-bibliography`, `doc-appendix`, ...) fold onto names no table contains, so
# they simply do not vote — exactly as their unprefixed spellings do not.
_DOC_ROLE_PREFIX = "doc-"

# A name-pattern match on a document longer than this many characters of text is
# ignored (a real chapter can share a word with a front-matter filename).
NAME_MAX_TEXT = 20_000
# An unclassified document longer than this is kept even when it sits before a
# publisher-declared bodymatter start (prefer translating one page too many).
UNCLASSIFIED_MAX_TEXT = 20_000
# A "Part I" divider page is a heading and nothing else.
PART_DIVIDER_MAX_TEXT = 600
PART_DIVIDER_MAX_BODY_TEXT = 200  # text left after removing the headings
# Ambiguous short names (ABBREV_PATTERNS) only count for pages this small.
ABBREV_MAX_TEXT = 1_000
# Generic epub:type="frontmatter" only counts for tiny card pages.
FRONTMATTER_TINY_MAX_TEXT = 1_000
# "This is the table of contents" is only believed for a document that actually
# reads like one: **short and link-dense**. Re-measured over every EPUB on this
# machine (17 files, 305 spine documents, `input/` and `output/`) *after* the
# self-closing-anchor bug in _LINK_RE was fixed — the numbers below are the
# corrected ones, and they look nothing like the numbers this comment used to
# quote, because that bug was inflating every prose document's link ratio.
#
# Only documents some layer actually *claims* are a ToC are ever measured, so
# these are the samples that matter:
#
#   claimed as ToC                                     chars   link ratio
#   Reentry, Contents (genuine)                          425         0.96
#   Presidents Book of Secrets, Contents (genuine)       527         0.92
#   The Experience Machine, toc (genuine)                691         0.90
#   Why We Remember, inlinetoc (genuine)                1005         0.46
#   Build a Reasoning Model, contents (genuine)         7160         1.00
#   Active Inference, Contents, bilingual (genuine)    13404         1.00
#   Reentry, Prologue mislabelled `type="toc"`          7347         0.00
#   Reentry, ditto in the bilingual output             10008         0.00
#
# and, as the upper guard, the densest documents in the corpus that are *not*
# contents pages:
#
#   not a ToC                                          chars   link ratio
#   Build a Reasoning Model, index                     16394         0.86
#   Presidents Book of Secrets, endnotes 2             67267         0.37
#   Presidents Book of Secrets, index                  43066         0.33
#   Presidents Book of Secrets, endnotes 1             59064         0.31
#   Why We Remember, index                             22254         0.29
#   every real chapter / preface measured           15k...111k   0.00-0.05
#
# The link ratio, not the length cap, is what keeps prose out: with anchors
# counted correctly, real prose scores 0.00-0.05 and both mislabelled
# "Reentry" Prologues score 0.00 — three hundred times below the 0.30 line.
# (The old comment credited the length cap for this, on the strength of
# chapters that "scored 0.47-0.58"; those scores were the anchor bug.)
#
# 20,000 chars: 1.5x the largest genuine ToC measured (13,404 — the bilingual
# Active Inference contents page; its English source was ~6,700, and the 6,000
# cap that used to sit here is precisely why it got translated), and less than
# half the smallest non-ToC document in the corpus that clears the ratio bar at
# all (the 43,066-char Presidents index, 0.33). It is also exactly
# NAME_MAX_TEXT, which keeps this rule from ever being more permissive than the
# ordinary filename layer it rides on. Both "Reentry" counterexamples are still
# rejected — on link ratio, which is the test that was doing the work all along.
# Overshooting the cap means "translate it", the harmless direction, so a
# monster ToC costs one wasted page, not a lost chapter.
TOC_MAX_TEXT = 20_000
# 0.30: comfortably under the 0.46 of the tightest real ToC measured (Penguin
# Random House writes an unlinked one-line description under every chapter
# title, which dilutes the ratio — the old flat 0.5 threshold missed it by 0.04
# and the whole contents page got translated), and comfortably over the 0.23 of
# the densest *short* non-ToC page seen (a 187-char "next reads" ad card). With
# anchors measured correctly this is now the load-bearing half of the test: the
# nearest prose above the line is a 43k-char index, held out by TOC_MAX_TEXT.
TOC_MIN_LINK_RATIO = 0.30


# ------------------------------------------------------------------- parsing


@dataclass
class SpineDoc:
    index: int
    idref: str
    href: str = ""
    path: str = ""
    size: int = 0
    text: str = ""
    epub_types: set[str] = field(default_factory=set)
    heading_text: str = ""
    link_text: str = ""
    properties: str = ""

    @property
    def text_len(self) -> int:
        return len(self.text)

    @property
    def link_ratio(self) -> float:
        """Fraction of the visible text that sits inside <a> elements. A table
        of contents is ~1.0; prose is ~0.0 (measured: 0.90-1.00 vs 0.00-0.16)."""
        return len(self.link_text) / max(len(self.text), 1)


@dataclass
class Decision:
    doc: SpineDoc
    excluded: bool
    layer: str
    reason: str


@dataclass
class Report:
    decisions: list[Decision]
    body_index: int | None
    body_layer: str
    body_reason: str

    @property
    def excluded_ids(self) -> list[str]:
        return [d.doc.idref for d in self.decisions if d.excluded]


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S | re.I)
_HEADING_RE = re.compile(r"<h[1-6]\b.*?</h[1-6]\s*>", re.S | re.I)
# `(?![^>]*/>)` skips *self-closing* anchors. `<a id="ch1en1"/>` is how several
# publishers plant endnote back-references, and matching one made the lazy
# `.*?</a>` run on to the next real `</a>` — swallowing whole paragraphs of
# prose as "link text". Measured: it pushed "The President's Book of Secrets"
# chapter 1 (29k chars of narrative, 5 anchors of which 2 self-closing) to a
# link ratio of 0.92, i.e. more ToC-like than a genuine contents page. Since
# link density is the test that keeps prose from being skipped as a ToC, an
# error in this direction is the dangerous one.
_LINK_RE = re.compile(r"<a\b(?![^>]*/>)[^>]*>.*?</a\s*>", re.S | re.I)
_VOID_OR_INLINE = {
    "br", "img", "hr", "a", "span", "link", "meta", "em", "i", "b", "strong",
    "sup", "sub", "small", "wbr", "svg", "image",
}


def _strip(html: str) -> str:
    txt = _TAG_RE.sub(" ", _SCRIPT_RE.sub(" ", html))
    txt = re.sub(r"&[#a-zA-Z0-9]{1,8};", " ", txt)
    return " ".join(txt.split())


def _canon(t: str) -> str:
    """`doc-toc` -> `toc`: fold the DPUB-ARIA spelling onto the epub:type one."""
    return t[len(_DOC_ROLE_PREFIX):] if t.startswith(_DOC_ROLE_PREFIX) else t


def _with_canon(types: set[str]) -> set[str]:
    """The types as written *plus* their canonical spellings, so lookups match
    either way while the audit line can still quote what the file actually says."""
    return types | {_canon(t) for t in types}


def _types_in_tag(tag: str) -> set[str]:
    out: set[str] = set()
    for attr in ("epub:type", "data-type", "role"):
        m = re.search(rf'\b{re.escape(attr)}="([^"]*)"', tag, re.I)
        if m:
            out |= {t.strip().lower() for t in m.group(1).split() if t.strip()}
    return out


def _structural_types(html: str) -> set[str]:
    """epub:type of <body> plus the first few structural elements inside it.

    Scoped on purpose: chapters carry `epub:type="note"` on their footnotes and
    `epub:type="pagebreak"` on every page anchor, and those must not vote.
    """
    types: set[str] = set()
    body = re.search(r"<body\b[^>]*>", html, re.I)
    if body:
        types |= _types_in_tag(body.group(0))
    start = body.end() if body else 0
    seen = 0
    for m in re.finditer(r"<([a-zA-Z][\w:-]*)\b[^>]*>", html[start : start + 6000]):
        if m.group(1).lower() in _VOID_OR_INLINE:
            continue
        types |= _types_in_tag(m.group(0))
        seen += 1
        if seen >= 3:
            break
    return _with_canon(types)


def _opf_path(zf: zipfile.ZipFile) -> str:
    try:
        container = zf.read("META-INF/container.xml").decode("utf-8", "replace")
        m = re.search(r'full-path="([^"]+)"', container)
        if m:
            return m.group(1)
    except KeyError:
        pass
    return next(n for n in zf.namelist() if n.endswith(".opf"))


def _manifest(opf: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"<item\b[^>]*?/?>", opf):
        frag = m.group(0)
        i = re.search(r'\bid="([^"]+)"', frag)
        h = re.search(r'\bhref="([^"]+)"', frag)
        if not (i and h):
            continue
        props = re.search(r'\bproperties="([^"]*)"', frag)
        out[i.group(1)] = {"href": h.group(1), "properties": props.group(1) if props else ""}
    return out


def _guide(opf: str) -> list[tuple[str, str]]:
    """[(type, href), ...] from <guide>, hrefs unfragmented."""
    block = re.search(r"<guide\b.*?</guide\s*>", opf, re.S | re.I)
    if not block:
        return []
    out = []
    for m in re.finditer(r"<reference\b[^>]*?/?>", block.group(0)):
        frag = m.group(0)
        t = re.search(r'\btype="([^"]+)"', frag)
        h = re.search(r'\bhref="([^"]+)"', frag)
        if t and h:
            out.append((t.group(1).strip().lower(), h.group(1).split("#")[0]))
    return out


def _landmarks(nav_html: str) -> list[tuple[str, str]]:
    """[(epub:type, href), ...] from the nav document's landmarks nav."""
    block = re.search(
        r'<nav\b[^>]*epub:type="[^"]*landmarks[^"]*".*?</nav\s*>', nav_html, re.S | re.I
    )
    if not block:
        return []
    out = []
    for m in re.finditer(r"<a\b[^>]*>", block.group(0), re.I):
        frag = m.group(0)
        t = re.search(r'\bepub:type="([^"]+)"', frag)
        h = re.search(r'\bhref="([^"]+)"', frag)
        if t and h:
            for tok in t.group(1).split():
                out.append((tok.strip().lower(), h.group(1).split("#")[0]))
    return out


def _norm(base: str, href: str) -> str:
    href = href.split("#")[0]
    return posixpath.normpath(posixpath.join(base, href)) if base else href


def _name_match(name: str, pattern: str) -> bool:
    """Whole-word match of `pattern` against the space-separated name tokens,
    tolerating a trailing serial number (`titlepage01`, `cover2`)."""
    return bool(
        re.search(rf"(?:^| )(?:{pattern})\d*(?:$| )", name)
    )


def _name_tokens(doc: SpineDoc) -> str:
    stem = posixpath.basename(doc.href).rsplit(".", 1)[0]
    blob = f"{doc.idref} {stem}".lower()
    return re.sub(r"[^a-z0-9]+", " ", blob)


# -------------------------------------------------------------- the analysis


def _read_spine(src: Path) -> tuple[list[SpineDoc], list[tuple[str, str]], list[tuple[str, str]], str]:
    with zipfile.ZipFile(src) as zf:
        opf_name = _opf_path(zf)
        opf = zf.read(opf_name).decode("utf-8", "replace")
        base = posixpath.dirname(opf_name)
        manifest = _manifest(opf)
        guide = [(t, _norm(base, h)) for t, h in _guide(opf)]

        landmarks: list[tuple[str, str]] = []
        nav_id = next(
            (i for i, v in manifest.items() if "nav" in v["properties"].split()), None
        )
        if nav_id:
            nav_path = _norm(base, manifest[nav_id]["href"])
            try:
                landmarks = [
                    (t, _norm(posixpath.dirname(nav_path), h))
                    for t, h in _landmarks(zf.read(nav_path).decode("utf-8", "replace"))
                ]
            except KeyError:
                landmarks = []

        docs: list[SpineDoc] = []
        spine = re.search(r"<spine\b.*?</spine\s*>", opf, re.S | re.I)
        spine_xml = spine.group(0) if spine else opf
        for idx, m in enumerate(re.finditer(r'<itemref\b[^>]*\bidref="([^"]+)"', spine_xml)):
            idref = m.group(1)
            entry = manifest.get(idref, {})
            href = entry.get("href", "")
            doc = SpineDoc(
                index=idx,
                idref=idref,
                href=href,
                path=_norm(base, href) if href else "",
                properties=entry.get("properties", ""),
            )
            if doc.path:
                try:
                    raw = zf.read(doc.path).decode("utf-8", "replace")
                except KeyError:
                    raw = ""
                if raw:
                    doc.size = len(raw)
                    doc.text = _strip(raw)
                    doc.epub_types = _structural_types(raw)
                    doc.heading_text = _strip(" ".join(_HEADING_RE.findall(raw)))
                    doc.link_text = _strip(" ".join(_LINK_RE.findall(raw)))
            docs.append(doc)
    return docs, guide, landmarks, opf_name


TOC_LABEL = "table of contents"


def _reads_like_toc(doc: SpineDoc) -> bool:
    """Does the document actually *look* like a contents page — short, and most
    of its text inside links? See TOC_MAX_TEXT / TOC_MIN_LINK_RATIO for the
    measurements behind the two numbers."""
    return doc.text_len <= TOC_MAX_TEXT and doc.link_ratio >= TOC_MIN_LINK_RATIO


def _reference_apparatus(doc: SpineDoc) -> str | None:
    """Is this an index / glossary / bibliography? Returns why, for the audit
    line. See REFERENCE_APPARATUS_TYPES for what this guards against.

    `_name_match` is whole-word, so `indexing.xhtml` does not trip it; a chapter
    genuinely called "Index of Notation" does, and is then *kept* — the harmless
    direction, which is why the name list does not need to be careful.
    """
    hit = doc.epub_types & REFERENCE_APPARATUS_TYPES
    if hit:
        return f'epub:type="{sorted(hit)[0]}"'
    if _name_match(_name_tokens(doc), REFERENCE_APPARATUS_NAMES):
        return f"id/href matches /{REFERENCE_APPARATUS_NAMES}/"
    return None


def _believable_toc(doc: SpineDoc) -> bool:
    """Reads like a contents page *and* is not reference apparatus."""
    return _reads_like_toc(doc) and _reference_apparatus(doc) is None


def _toc_claim(
    doc: SpineDoc,
    guide_by_path: dict[str, str],
    landmark_by_path: dict[str, str],
) -> tuple[str, str] | None:
    """Return (layer, reason) if some layer says `doc` is the contents page *and*
    the document reads like one.

    Split out from `_classify` because this is the one verdict that ignores
    position: the contents page is skipped wherever it sits in the spine, so the
    caller applies this to every document, not just to the front-matter prefix.
    """
    if doc.epub_types & BODY_EPUB_TYPES or not _believable_toc(doc):
        return None
    return _toc_claim_source(doc, guide_by_path, landmark_by_path)


def _toc_claim_source(
    doc: SpineDoc,
    guide_by_path: dict[str, str],
    landmark_by_path: dict[str, str],
) -> tuple[str, str] | None:
    """Which layer, if any, says `doc` is the contents page — without checking
    whether the claim survives looking at the document. Split out so the veto in
    `_toc_veto_note` can name the claim it is refusing."""
    lm = landmark_by_path.get(doc.path)
    if lm and FRONT_EPUB_TYPES.get(_canon(lm)) == TOC_LABEL:
        return "landmarks", f'landmarks epub:type="{lm}" ({TOC_LABEL})'

    gt = guide_by_path.get(doc.path)
    if gt and FRONT_GUIDE_TYPES.get(gt) == TOC_LABEL:
        return "guide", f'<guide> reference type="{gt}" ({TOC_LABEL})'

    for t in sorted(doc.epub_types):
        if FRONT_EPUB_TYPES.get(_canon(t)) == TOC_LABEL:
            return "epub:type", f'epub:type="{t}" ({TOC_LABEL})'

    if "nav" in doc.properties.split():
        return "manifest", 'manifest properties="nav" (navigation document)'

    name = _name_tokens(doc)
    for pattern, label in NAME_PATTERNS:
        if label == TOC_LABEL and _name_match(name, pattern):
            return (
                "filename",
                f"id/href matches /{pattern}/ ({TOC_LABEL}), {doc.text_len} chars",
            )
    return None


def _toc_veto_note(
    doc: SpineDoc,
    guide_by_path: dict[str, str],
    landmark_by_path: dict[str, str],
) -> str | None:
    """never silent: a document that *would* have been skipped as the contents
    page, and was not because it is an index / glossary / bibliography, says so
    on its own audit line rather than quietly becoming "body"."""
    apparatus = _reference_apparatus(doc)
    if not apparatus or doc.epub_types & BODY_EPUB_TYPES or not _reads_like_toc(doc):
        return None
    src = _toc_claim_source(doc, guide_by_path, landmark_by_path)
    if not src:
        return None
    return f"{src[1]} NOT BELIEVED, reference apparatus: {apparatus}"


def _classify(
    doc: SpineDoc,
    guide_by_path: dict[str, str],
    landmark_by_path: dict[str, str],
) -> tuple[str, str] | None:
    """Return (layer, reason) if `doc` looks like front matter, else None."""
    if doc.epub_types & BODY_EPUB_TYPES:
        return None

    def _toc_ok(label: str) -> bool:
        """A "table of contents" claim has to survive looking at the document."""
        return label != TOC_LABEL or _believable_toc(doc)

    lm = landmark_by_path.get(doc.path)
    lmc = _canon(lm) if lm else None
    if lmc and lmc in FRONT_EPUB_TYPES and _toc_ok(FRONT_EPUB_TYPES[lmc]):
        return "landmarks", f'landmarks epub:type="{lm}" ({FRONT_EPUB_TYPES[lmc]})'

    gt = guide_by_path.get(doc.path)
    if gt and gt in FRONT_GUIDE_TYPES and _toc_ok(FRONT_GUIDE_TYPES[gt]):
        return "guide", f'<guide> reference type="{gt}" ({FRONT_GUIDE_TYPES[gt]})'

    for t in sorted(doc.epub_types):
        tc = _canon(t)
        if tc in FRONT_EPUB_TYPES and _toc_ok(FRONT_EPUB_TYPES[tc]):
            return "epub:type", f'epub:type="{t}" ({FRONT_EPUB_TYPES[tc]})'

    if lm == "bodymatter":
        # Declared body. The weak layers below (filename, size) may not override
        # it — only the specific, semantic signals above may.
        return None

    if "nav" in doc.properties.split():
        return "manifest", 'manifest properties="nav" (navigation document)'

    name = _name_tokens(doc)
    if doc.text_len <= NAME_MAX_TEXT:
        for pattern, label in NAME_PATTERNS:
            if _name_match(name, pattern) and _toc_ok(label):
                return "filename", f"id/href matches /{pattern}/ ({label}), {doc.text_len} chars"

    if doc.text_len <= ABBREV_MAX_TEXT:
        for pattern, label in ABBREV_PATTERNS:
            if _name_match(name, pattern) and _toc_ok(label):
                return (
                    "filename",
                    f"id/href matches short-page name /{pattern}/ ({label}), "
                    f"only {doc.text_len} chars",
                )

    if "frontmatter" in doc.epub_types and doc.text_len <= FRONTMATTER_TINY_MAX_TEXT:
        return (
            "size",
            f'epub:type="frontmatter" and only {doc.text_len} chars (card page)',
        )

    return None


def _is_part_divider(doc: SpineDoc) -> tuple[str, str] | None:
    """A bare "Part I. ..." separator page: a heading and nothing else."""
    if doc.epub_types & {"bodymatter", "chapter"}:
        return None
    name = _name_tokens(doc)
    looks_part = "part" in doc.epub_types or bool(
        re.search(r"(?:^| )(?:part|section|book|division)\s*\d*(?:$| )", name)
    )
    if not looks_part:
        return None
    if doc.text_len > PART_DIVIDER_MAX_TEXT:
        return None
    rest = doc.text_len - len(doc.heading_text)
    if rest > PART_DIVIDER_MAX_BODY_TEXT:
        return None
    typed = "part" in doc.epub_types
    if not typed and not doc.heading_text:
        # Calibre names every split file `partNNNN.html`; without a heading
        # there is nothing to say this one is a divider rather than a stub.
        return None
    label = 'epub:type="part"' if typed else "id/href looks like a part divider"
    return "size", f"{label}, {doc.text_len} chars of which headings {len(doc.heading_text)}"


def analyze(src: Path) -> Report:
    """Classify every spine document of `src`. Never modifies the file."""
    docs, guide, landmarks, _ = _read_spine(src)
    guide_by_path = {}
    for t, p in guide:
        guide_by_path.setdefault(p, t)
    landmark_by_path = {}
    for t, p in landmarks:
        landmark_by_path.setdefault(p, t)
    index_by_path = {d.path: d.index for d in docs if d.path}

    classified: dict[int, tuple[str, str]] = {}
    for d in docs:
        c = _classify(d, guide_by_path, landmark_by_path)
        if c is None:
            part = _is_part_divider(d)
            # A "Part I" divider sitting between the front matter and chapter 1
            # must not end the front-matter prefix, or the body would look like
            # it starts on a page that is a heading and nothing else.
            c = (part[0], "part divider page: " + part[1]) if part else None
        if c:
            classified[d.index] = c

    # --- where does the body start? authoritative signals first -------------
    body_index: int | None = None
    body_layer = ""
    body_reason = ""

    distrusted: list[str] = []

    def _pointer(path: str | None, layer: str, label: str) -> int | None:
        """Accept a 'the body starts here' pointer unless it aims at a document
        that is itself specifically classified front matter.

        Both pointer kinds are routinely wrong in shipped books: O'Reilly's
        HTMLBook aims `<guide type="text">` at the title page, and Penguin
        Random House aims `landmarks epub:type="bodymatter"` at the dedication.
        A pointer into declared front matter is not a boundary — say so loudly
        and fall through to the next layer.
        """
        if not path or path not in index_by_path:
            return None
        idx = index_by_path[path]
        if idx in classified:
            distrusted.append(
                f"{label} -> {posixpath.basename(path)} DISTRUSTED "
                f"(that document is itself front matter: {classified[idx][1]})"
            )
            return None
        nonlocal body_layer, body_reason
        body_layer, body_reason = layer, f"{label} -> {posixpath.basename(path)}"
        return idx

    body_index = _pointer(
        next((p for t, p in landmarks if t == "bodymatter"), None),
        "landmarks",
        'nav landmarks epub:type="bodymatter"',
    )
    if body_index is None:
        body_index = _pointer(
            next((p for t, p in guide if t == "text"), None),
            "guide",
            '<guide> reference type="text"',
        )

    if body_index is None:
        body_index = next((d.index for d in docs if d.index not in classified), len(docs))
        body_layer = "prefix-scan"
        body_reason = "first spine document not classified as front matter"
    if distrusted:
        body_reason = "; ".join(distrusted) + "; " + body_reason

    # --- decide -------------------------------------------------------------
    decisions: list[Decision] = []
    for d in docs:
        layer, reason, excluded = "", "", False
        toc = _toc_claim(d, guide_by_path, landmark_by_path)
        if toc:
            # Position-independent: a contents page is never translated, whether
            # it sits in the front-matter prefix, behind the declared body start
            # (Calibre and Penguin Random House both ship "inline" contents
            # pages there), or anywhere else in the spine.
            layer, reason = toc
            reason = "contents page (never translated): " + reason
            excluded = True
        elif d.index < body_index:
            if d.index in classified:
                layer, reason = classified[d.index]
                excluded = True
            elif body_layer == "landmarks" and d.text_len <= UNCLASSIFIED_MAX_TEXT:
                layer = "landmarks"
                excluded = True
                reason = "before the declared bodymatter start"
            else:
                reason = (
                    "before the body start but unclassified"
                    + (f" and large ({d.text_len} chars)" if d.text_len > UNCLASSIFIED_MAX_TEXT else "")
                    + " -> kept"
                )
        else:
            part = _is_part_divider(d)
            if part:
                layer, reason = part
                reason = "part divider page: " + reason
                excluded = True
            elif d.index == body_index:
                reason = "body starts here"
            else:
                reason = "body"
        if not excluded:
            note = _toc_veto_note(d, guide_by_path, landmark_by_path)
            if note:
                reason = f"{reason} [{note}]"
        decisions.append(Decision(doc=d, excluded=excluded, layer=layer, reason=reason))

    return Report(decisions, body_index, body_layer, body_reason)


def detect(src: Path, keep_ids: list[str] | None = None) -> Report:
    """`analyze` plus an explicit opt-out list (ids the user wants translated)."""
    report = analyze(src)
    keep = set(keep_ids or ())
    for d in report.decisions:
        if d.excluded and d.doc.idref in keep:
            d.excluded = False
            d.layer = "config"
            d.reason = "listed in front_matter_keep_ids -> kept"
    return report


def print_report(report: Report, prefix: str = "  ") -> None:
    docs = report.decisions
    print(f"{prefix}spine documents: {len(docs)}")
    start = report.body_index if report.body_index is not None else len(docs)
    where = docs[start].doc.idref if start < len(docs) else "(none — nothing classified as body)"
    print(f"{prefix}body starts at #{start} {where}  [{report.body_layer or 'n/a'}: {report.body_reason}]")
    for d in docs:
        mark = "SKIP" if d.excluded else "keep"
        layer = f"[{d.layer}]" if d.layer else ""
        print(
            f"{prefix}  #{d.doc.index:<3d} {mark}  {d.doc.idref:<34.34s} "
            f"{posixpath.basename(d.doc.href):<34.34s} {d.doc.text_len:>7d}c  {layer} {d.reason}"
        )
    ids = report.excluded_ids
    print(f"{prefix}front matter to skip ({len(ids)}): {', '.join(ids) or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: python {Path(sys.argv[0]).name} <book.epub> [...]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        book = Path(arg)
        print(f"\n=== {book.name}")
        print_report(detect(book))
