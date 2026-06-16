"""Build a book-level English->Simplified Chinese glossary from an EPUB.

Stage 2 of the pipeline: scan the whole book (spine reading order), extract
proper-noun / acronym candidates, resolve each to its conventional Chinese
rendering with the LLM, and write a reviewable TSV. Feed that TSV into the
translation prompt so the whole book stays terminologically consistent.

Usage:
    uv run python glossary.py input/book.epub
    uv run python glossary.py input/book.epub --no-resolve      # extract only, no API calls
    uv run python glossary.py input/book.epub --min-freq 2 -o output/book.glossary.tsv
"""

import argparse
import json
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from posixpath import join as posixjoin, normpath

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# EPUB reading (spine reading order)
# ---------------------------------------------------------------------------

CONTAINER = "META-INF/container.xml"


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read(CONTAINER))
    rootfile = root.find(".//{*}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise RuntimeError("container.xml has no rootfile full-path")
    return rootfile.get("full-path")


def spine_documents(epub_path: Path) -> list[tuple[str, str]]:
    """Return [(doc_id, text)] for every content document in spine order."""
    with zipfile.ZipFile(epub_path) as zf:
        opf = _opf_path(zf)
        opf_dir = os.path.dirname(opf)
        root = ET.fromstring(zf.read(opf))
        manifest = {
            item.get("id"): item.get("href")
            for item in root.findall(".//{*}manifest/{*}item")
            if item.get("id") and item.get("href")
        }
        order = [
            ref.get("idref")
            for ref in root.findall(".//{*}spine/{*}itemref")
            if ref.get("idref")
        ]
        docs: list[tuple[str, str]] = []
        for doc_id in order:
            href = manifest.get(doc_id)
            if not href:
                continue
            path = normpath(posixjoin(opf_dir, href)) if opf_dir else href
            try:
                raw = zf.read(path).decode("utf-8", "replace")
            except KeyError:
                continue
            docs.append((doc_id, html_to_text(raw)))
    return docs


class _Stripper(HTMLParser):
    BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "h5", "li", "br", "blockquote", "td"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw: str) -> str:
    parser = _Stripper()
    parser.feed(raw)
    return parser.text()


# ---------------------------------------------------------------------------
# Candidate extraction (proper nouns + acronyms, frequency-counted)
# ---------------------------------------------------------------------------

CONNECTORS = {"of", "the", "and", "for", "de", "von", "van", "del", "da", "&"}
# Capitalized words that are almost always sentence-openers, not names.
COMMON_CAPS = {
    "The", "A", "An", "In", "On", "At", "By", "But", "And", "Or", "So", "Yet",
    "It", "He", "She", "They", "We", "You", "I", "This", "That", "These", "Those",
    "When", "Then", "Now", "Soon", "Still", "Even", "Also", "Yet", "Thus", "Hence",
    "However", "Moreover", "Meanwhile", "Therefore", "Although", "Because", "Since",
    "While", "After", "Before", "During", "Across", "Outside", "Inside", "Through",
    "Adding", "Clearly", "Early", "Endless", "Such", "More", "Most", "Many", "Few",
    "Of", "As", "If", "Mr", "Mrs", "Ms", "Dr",
    "II", "III", "IV", "Chapter", "CHAPTER", "Part",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-']*")
SENT_SPLIT = re.compile(r"(?<=[.!?”\"])\s+|\n+")
ACRONYM_RE = re.compile(r"\b([A-Z]{2,}(?:-?[A-Z0-9]+)*)\b")
# Punctuation that ends a proper-noun run even with no whitespace gap.
BREAK_CHARS = set(",;:!?()[]{}\"“”—–…/«»")


def _sentences(text: str):
    for chunk in SENT_SPLIT.split(text):
        chunk = chunk.strip()
        if chunk:
            yield chunk


def _is_cap(word: str) -> bool:
    return bool(word) and word[0].isupper()


def _clean(tok: str) -> str:
    """Drop a possessive suffix and stray edge punctuation (keep internal '-')."""
    tok = re.sub(r"[’']s$", "", tok)
    return tok.strip("’'.")


def _broken(gap: str) -> bool:
    return any(c in BREAK_CHARS for c in gap)


def extract_candidates(text: str) -> dict[str, dict]:
    """Map normalized term -> {freq, context}. Proper-noun phrases + acronyms."""
    sentences = list(_sentences(text))

    # Pass 1: words seen capitalized in a NON-sentence-initial slot are "trusted"
    # names; this drops "However", "Adding", etc. that only ever open a sentence.
    trusted: set[str] = set()
    for sent in sentences:
        for i, m in enumerate(TOKEN_RE.finditer(sent)):
            tok = m.group()
            if i > 0 and _is_cap(tok) and tok not in COMMON_CAPS:
                trusted.add(_clean(tok))

    freq: Counter = Counter()
    context: dict[str, str] = {}

    def record(term: str, sent: str):
        norm = term.strip(" .,'’-")
        if len(norm) < 2:
            return
        freq[norm] += 1
        context.setdefault(norm, sent[:200].strip())

    def is_name_token(tok: str, first: bool) -> bool:
        return _is_cap(tok) and ((not first and tok not in COMMON_CAPS) or _clean(tok) in trusted)

    for sent in sentences:
        matches = list(TOKEN_RE.finditer(sent))
        n = len(matches)
        i = 0
        while i < n:
            if is_name_token(matches[i].group(), i == 0):
                run = [_clean(matches[i].group())]
                last_end = matches[i].end()
                j = i + 1
                while j < n:
                    if _broken(sent[last_end : matches[j].start()]):
                        break
                    nxt = matches[j].group()
                    if nxt.lower() in CONNECTORS and j + 1 < n:
                        gap2 = sent[matches[j].end() : matches[j + 1].start()]
                        if _is_cap(matches[j + 1].group()) and not _broken(gap2):
                            run.append(nxt.lower())
                            last_end = matches[j].end()
                            j += 1
                            continue
                        break
                    if _is_cap(nxt) and (nxt not in COMMON_CAPS or _clean(nxt) in trusted):
                        run.append(_clean(nxt))
                        last_end = matches[j].end()
                        j += 1
                        continue
                    break
                record(" ".join(run), sent)
                i = j
            else:
                i += 1

        # Acronyms (ENIAC, TSMC, DRAM, EUV...).
        for m in ACRONYM_RE.findall(sent):
            if m not in COMMON_CAPS and len(m) >= 2:
                record(m, sent)

    return {t: {"freq": freq[t], "context": context.get(t, "")} for t in freq}


# ---------------------------------------------------------------------------
# LLM resolution (provider-aware, mirrors main.py env handling)
# ---------------------------------------------------------------------------


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _messages_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _completions_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def llm_chat(system: str, user: str, *, max_tokens: int = 4096, timeout: float = 300.0,
             max_retries: int = 5) -> str:
    """One non-streaming chat turn against the OpenAI- or Anthropic-style proxy.

    Retries transient failures (gateway timeouts 502/503/504/524, rate limit 429,
    and connection/read errors) with exponential backoff, so a single network
    hiccup during a long glossary run doesn't kill the whole process.
    """
    key = _env("EPUB_TRANSLATOR_API_KEY")
    base = _env("EPUB_TRANSLATOR_BASE_URL")
    model = _env("EPUB_TRANSLATOR_MODEL")
    provider = os.getenv("EPUB_TRANSLATOR_PROVIDER", "").lower()
    use_claude = provider == "anthropic" or (not provider and model.startswith("claude-"))

    if use_claude:
        url = _messages_url(base)
        headers = {
            "x-api-key": key,
            "Authorization": f"Bearer {key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
    else:
        url = _completions_url(base)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    RETRYABLE = {429, 500, 502, 503, 504, 524}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                if resp.status_code in RETRYABLE and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"  LLM API {resp.status_code}, retry {attempt + 1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if use_claude:
                return "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text")
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  network error ({type(e).__name__}), retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"LLM API failed after {max_retries} retries: {last_err}")


RESOLVE_SYSTEM = (
    "You are a bilingual terminology expert for English->Simplified Chinese "
    "translation of technical nonfiction (semiconductors, computing, military, "
    "business history). For each English term, give the CONVENTIONAL, established "
    "Simplified Chinese rendering as used in Chinese Wikipedia and 全国科学技术名词审定委员会 "
    "(术语在线), not a literal morpheme-by-morpheme calque.\n"
    "Rules:\n"
    "- People: use the standard Chinese name. For Chinese/Japanese/Korean figures use their "
    "actual native name (e.g. Morris Chang->张忠谋, Akio Morita->盛田昭夫).\n"
    "- Companies/orgs/products/places: use the established Chinese name "
    "(e.g. Fairchild->仙童半导体, B-29 Superfortress->超级空中堡垒).\n"
    "- Prefer the reader-familiar standard term over jargon "
    "(heat-seeking missile->红外制导导弹, not 热寻的导弹).\n"
    "- If several renderings are acceptable, put your pick in zh and the rest in alt, set confidence=low.\n"
    "- If the candidate is NOT a real term/name (generic word, extraction noise), "
    'set zh="" and confidence="skip".\n'
    "Use the provided context sentence to disambiguate.\n"
    'Return STRICT JSON only: an array of {"en","zh","confidence","alt","note"}. '
    'confidence is one of "high","low","skip". alt and note may be "".'
)


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def resolve_terms(terms: list[tuple[str, str]], batch_size: int = 20, timeout: float = 300.0) -> dict[str, dict]:
    """terms: [(en, context)] -> {en: {zh, confidence, alt, note}}."""
    resolved: dict[str, dict] = {}
    for start in range(0, len(terms), batch_size):
        batch = terms[start : start + batch_size]
        payload = [{"en": en, "context": ctx} for en, ctx in batch]
        user = (
            "Resolve these terms to Simplified Chinese. Return one JSON object per term, "
            "same order:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
        )
        try:
            raw = llm_chat(RESOLVE_SYSTEM, user, timeout=timeout)
            for row in _parse_json_array(raw):
                en = (row.get("en") or "").strip()
                if en:
                    resolved[en] = {
                        "zh": (row.get("zh") or "").strip(),
                        "confidence": (row.get("confidence") or "").strip().lower(),
                        "alt": (row.get("alt") or "").strip(),
                        "note": (row.get("note") or "").strip(),
                    }
        except Exception as e:
            # One bad batch shouldn't discard the whole run; skip it and continue.
            print(f"  WARNING: batch {start}-{start + len(batch)} failed ({e}); skipping")
        print(f"  resolved {min(start + batch_size, len(terms))}/{len(terms)}")
    return resolved


# ---------------------------------------------------------------------------
# Reconciliation (dedup + cross-term consistency)
# ---------------------------------------------------------------------------

RECONCILE_SYSTEM = (
    "You are a bilingual terminology editor. Each input GROUP contains related English "
    "terms (the same person/place/entity in different forms) whose Simplified Chinese "
    "renderings are currently INCONSISTENT. For every group, return one CONSISTENT set: "
    "a full name's Chinese must contain the surname's Chinese, every form must agree, and "
    "use the conventional established rendering (Chinese Wikipedia / common usage; e.g. "
    "Intel co-founder Andy Grove -> 安迪·格鲁夫, so Grove -> 格鲁夫). "
    'Return STRICT JSON only: an array of {"en","zh"} covering EVERY term in every group.'
)


def _is_variant(a: str, b: str) -> bool:
    """Morphological kin: prefix relation or shared token (American~America)."""
    la, lb = a.lower(), b.lower()
    if la == lb:
        return False
    if la.startswith(lb) or lb.startswith(la):
        return True
    return bool(set(la.split()) & set(lb.split()))


def _contig_sub(short: list[str], long: list[str]) -> bool:
    """True if `short` is a contiguous sub-sequence of `long` (and shorter)."""
    if len(short) >= len(long):
        return False
    return any(long[s : s + len(short)] == short for s in range(len(long) - len(short) + 1))


def _drop_coordinations(entries: list[dict]) -> list[dict]:
    """Drop 'X and Y' rows when X and Y are each their own entry."""
    ens = {e["en"].lower() for e in entries}
    out = []
    for e in entries:
        parts = re.split(r"\s+and\s+", e["en"])
        if len(parts) >= 2 and all(p.strip().lower() in ens for p in parts):
            continue
        out.append(e)
    return out


def _dedup_casefold(entries: list[dict]) -> list[dict]:
    """Merge surface forms that differ only by case; sum freq, keep natural case."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["en"].lower(), []).append(e)
    out = []
    for g in groups.values():
        g.sort(key=lambda e: (-e["freq"], e["en"].isupper()))
        canon = dict(g[0])
        canon["freq"] = sum(x["freq"] for x in g)
        best = sorted(g, key=lambda e: (0 if e.get("confidence") == "high" else 1, -e["freq"]))[0]
        canon["zh"] = best.get("zh", "")
        out.append(canon)
    return out


def _collapse_same_zh(entries: list[dict]) -> list[dict]:
    """Fold morphological variants that share an identical Chinese rendering."""
    by_zh: dict[str, list[dict]] = {}
    out = []
    for e in entries:
        if e.get("zh"):
            by_zh.setdefault(e["zh"], []).append(e)
        else:
            out.append(e)
    for g in by_zh.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        g.sort(key=lambda e: -e["freq"])
        used = [False] * len(g)
        for i in range(len(g)):
            if used[i]:
                continue
            cluster = [g[i]]
            used[i] = True
            for j in range(i + 1, len(g)):
                if not used[j] and any(_is_variant(m["en"], g[j]["en"]) for m in cluster):
                    cluster.append(g[j])
                    used[j] = True
            head = dict(cluster[0])
            if len(cluster) > 1:
                head["freq"] = sum(m["freq"] for m in cluster)
                extra = "也作: " + ", ".join(m["en"] for m in cluster[1:])
                head["note"] = f'{head.get("note", "")}; {extra}'.strip("; ")
            out.append(head)
    return out


def _reconcile_names(entries: list[dict], allow_llm: bool) -> list[dict]:
    """Find component/full-name groups whose Chinese disagree; fix or flag them."""
    toks = [[t.lower() for t in e["en"].split()] for e in entries]
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(entries)):
        for j in range(len(entries)):
            if i != j and _contig_sub(toks[i], toks[j]):
                parent[find(i)] = find(j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(entries)):
        clusters.setdefault(find(i), []).append(i)

    conflict_groups = []
    for idxs in clusters.values():
        if len(idxs) < 2:
            continue
        bad = any(
            i != j
            and _contig_sub(toks[i], toks[j])
            and entries[i]["zh"]
            and entries[j]["zh"]
            and entries[i]["zh"] not in entries[j]["zh"]
            for i in idxs
            for j in idxs
        )
        if bad:
            conflict_groups.append(idxs)

    if not conflict_groups:
        return entries

    if not allow_llm:
        for idxs in conflict_groups:
            forms = "/".join(sorted({entries[i]["en"] for i in idxs}))
            for i in idxs:
                entries[i]["confidence"] = "low"
                entries[i]["note"] = f'{entries[i].get("note", "")}; 不一致待统一: {forms}'.strip("; ")
        return entries

    payload = [
        [{"en": entries[i]["en"], "zh": entries[i]["zh"], "context": entries[i]["context"][:120]} for i in idxs]
        for idxs in conflict_groups
    ]
    user = (
        "These groups of related terms have inconsistent Chinese. Return a consistent "
        '{"en","zh"} for EVERY term:\n' + json.dumps(payload, ensure_ascii=False)
    )
    try:
        fixed = {
            (r.get("en") or "").strip(): (r.get("zh") or "").strip()
            for r in _parse_json_array(llm_chat(RECONCILE_SYSTEM, user))
        }
    except Exception as err:  # noqa: BLE001 - reconciliation is best-effort
        print(f"  reconcile call failed, leaving as-is: {err}")
        return entries
    for idxs in conflict_groups:
        for i in idxs:
            if fixed.get(entries[i]["en"]):
                entries[i]["zh"] = fixed[entries[i]["en"]]
                entries[i]["note"] = f'{entries[i].get("note", "")}; 已统一'.strip("; ")
    print(f"  reconciled {len(conflict_groups)} inconsistent name group(s)")
    return entries


def reconcile(entries: list[dict], allow_llm: bool) -> list[dict]:
    entries = _drop_coordinations(entries)
    entries = _dedup_casefold(entries)
    entries = _collapse_same_zh(entries)
    entries = _reconcile_names(entries, allow_llm)
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a book-level EN->ZH glossary TSV from an EPUB.")
    p.add_argument("source", help="Path to the source .epub file")
    p.add_argument("-o", "--output", help="TSV path. Default: output/<stem>.glossary.tsv")
    p.add_argument("--min-freq", type=int, default=1, help="Drop candidates below this frequency")
    p.add_argument("--max-terms", type=int, default=400, help="Cap candidates sent for resolution")
    p.add_argument("--batch-size", type=int, default=20, help="Terms per LLM resolution call")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (seconds)")
    p.add_argument("--no-resolve", action="store_true", help="Extract candidates only; no API calls")
    p.add_argument("--no-reconcile", action="store_true", help="Skip the dedup/consistency pass")
    p.add_argument("--keep-skipped", action="store_true", help="Keep entries the model marks as noise")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source EPUB not found: {source}")

    docs = spine_documents(source)
    full_text = "\n".join(text for _, text in docs)
    print(f"source: {source}")
    print(f"spine documents: {len(docs)} | characters: {len(full_text):,}")

    candidates = extract_candidates(full_text)
    candidates = {t: v for t, v in candidates.items() if v["freq"] >= args.min_freq}
    ordered = sorted(candidates.items(), key=lambda kv: (-kv[1]["freq"], kv[0].lower()))
    if len(ordered) > args.max_terms:
        print(f"candidates: {len(ordered)} -> capping to top {args.max_terms} by frequency")
        ordered = ordered[: args.max_terms]
    else:
        print(f"candidates: {len(ordered)}")

    target = Path(args.output) if args.output else Path("output") / f"{source.stem}.glossary.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        {"en": en, "zh": "", "freq": v["freq"], "confidence": "", "alt": "", "note": "", "context": v["context"]}
        for en, v in ordered
    ]

    if not args.no_resolve:
        print("resolving terms via LLM...")
        resolved = resolve_terms(
            [(e["en"], e["context"]) for e in entries],
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
        for e in entries:
            r = resolved.get(e["en"], {})
            e["zh"], e["confidence"] = r.get("zh", ""), r.get("confidence", "")
            e["alt"], e["note"] = r.get("alt", ""), r.get("note", "")
        if not args.keep_skipped:
            entries = [e for e in entries if e["confidence"] != "skip"]

    if not args.no_reconcile:
        before = len(entries)
        entries = reconcile(entries, allow_llm=not args.no_resolve)
        print(f"reconcile: {before} -> {len(entries)} terms")

    entries.sort(key=lambda e: (-e["freq"], e["en"].lower()))

    with target.open("w", encoding="utf-8") as f:
        f.write("english\tchinese\tfreq\tconfidence\talternatives\tnote\tcontext\n")
        for e in entries:
            ctx = e["context"].replace("\t", " ").replace("\n", " ")
            f.write(f'{e["en"]}\t{e["zh"]}\t{e["freq"]}\t{e["confidence"]}\t{e.get("alt","")}\t{e.get("note","")}\t{ctx}\n')

    print(f"glossary: {target} ({len(entries)} terms)")


if __name__ == "__main__":
    main()
