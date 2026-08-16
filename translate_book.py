"""One-command bilingual book translator driven by a YAML config.

Pipeline:
  1. (optional) build/load a book-level glossary  -> glossary.py
  2. detect the book's front matter (cover / title page / copyright /
     dedication / contents / preface / part dividers) -> front_matter.py
  3. (optional) make a spine-trimmed copy of the EPUB so excluded
     documents (front matter, endnotes, index, ...) are skipped
  3. translate with a live tqdm progress bar

Usage:
    uv run python translate_book.py                 # reads translate_book.yaml
    uv run python translate_book.py my_book.yaml
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from epub_translator import SubmitKind, language, translate
from tqdm import tqdm

import front_matter as front_matter_mod
import glossary as glossary_mod
import mask_code as mask_code_mod
import mask_math as mask_math_mod
from main import (
    DEFAULT_USER_PROMPT,
    LANGUAGES,
    SUBMIT_KINDS,
    build_llm,
    compose_user_prompt,
    drop_unsupported_proxy_env,
)

DEFAULTS = {
    "language": "zh",
    "submit": "append-block",
    "concurrency": 16,
    "max_group_tokens": 2600,
    "glossary": {"enabled": True, "path": "", "auto_generate": True, "min_freq": 2},
    "mask_math": True,
    "mask_code": True,
    "skip_front_matter": True,
    "front_matter_keep_ids": [],
    "exclude_spine_ids": [],
    "cache_path": "",
    "user_prompt": "",
}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = {**DEFAULTS, **cfg}
    merged["glossary"] = {**DEFAULTS["glossary"], **(cfg.get("glossary") or {})}
    return merged


def trim_spine(source: Path, exclude_ids: list[str], dest: Path) -> Path:
    """Write a copy of `source` with `exclude_ids` removed from the OPF spine.

    The excluded documents stay in the archive (so internal links don't break)
    but leave the reading order, so translate() never visits them.
    """
    with zipfile.ZipFile(source) as zin:
        opf_name = next(n for n in zin.namelist() if n.endswith(".opf"))
        opf = zin.read(opf_name).decode("utf-8")

    exclude = set(exclude_ids)

    def drop(m: re.Match) -> str:
        return "" if m.group("id") in exclude else m.group(0)

    new_opf = re.sub(r'<itemref\b[^>]*\bidref="(?P<id>[^"]+)"[^>]*/>', drop, opf)
    removed = [i for i in exclude if f'idref="{i}"' in opf]

    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
        if "mimetype" in zin.namelist():
            zout.writestr("mimetype", zin.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for item in zin.infolist():
            if item.filename == "mimetype":
                continue
            data = new_opf.encode("utf-8") if item.filename == opf_name else zin.read(item.filename)
            zout.writestr(item, data)

    print(f"  excluded from spine: {removed or 'none'}")
    return dest


def _spine_itemrefs(opf: str) -> list[tuple[str, str]]:
    """Return [(idref, itemref_xml), ...] in spine order."""
    out = []
    for m in re.finditer(r'<itemref\b[^>]*/>', opf):
        frag = m.group(0)
        id_m = re.search(r'\bidref="([^"]+)"', frag)
        if id_m:
            out.append((id_m.group(1), frag))
    return out


def restore_spine(target: Path, source: Path, exclude_ids: list[str]) -> list[str]:
    """Re-insert `exclude_ids` into `target`'s OPF spine at their original spots.

    `trim_spine` drops excluded documents from the spine only so `translate()`
    skips them; excluding from translation must not hide them from the reader.
    The excluded files already survive into `target` (they stay in the archive),
    but without a spine itemref they are unreachable. Here we read the original
    reading order from `source` and splice each excluded item back — verbatim, so
    a `linear="no"` flag is preserved — after its original predecessor. The
    result is a book whose spine matches the source, with excluded docs present
    but untranslated.
    """
    exclude = set(exclude_ids)
    with zipfile.ZipFile(source) as z:
        src_opf_name = next(n for n in z.namelist() if n.endswith(".opf"))
        original = _spine_itemrefs(z.read(src_opf_name).decode("utf-8"))
    frags = {i: f for i, f in original if i in exclude}
    if not frags:
        return []

    with zipfile.ZipFile(target) as z:
        opf_name = next(n for n in z.namelist() if n.endswith(".opf"))
        opf = z.read(opf_name).decode("utf-8")
    present = {i for i, _ in _spine_itemrefs(opf)}

    restored: list[str] = []
    anchor: str | None = None  # idref the next item should be inserted after
    for idref, _ in original:
        if idref not in exclude:
            anchor = idref
            continue
        if idref in present:  # already in target spine; don't duplicate
            anchor = idref
            continue
        frag = frags[idref]
        if anchor is None:
            opf = re.sub(r'(<spine\b[^>]*>)', lambda m: m.group(1) + frag, opf, count=1)
        else:
            pat = re.compile(r'(<itemref\b[^>]*\bidref="' + re.escape(anchor) + r'"[^>]*/>)')
            opf = pat.sub(lambda m: m.group(1) + frag, opf, count=1)
        restored.append(idref)
        anchor = idref

    if not restored:
        return []

    tmp = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(target) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        if "mimetype" in zin.namelist():
            zout.writestr("mimetype", zin.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for item in zin.infolist():
            if item.filename == "mimetype":
                continue
            data = opf.encode("utf-8") if item.filename == opf_name else zin.read(item.filename)
            zout.writestr(item, data)
    tmp.replace(target)
    return restored


def ensure_glossary(cfg: dict, source: Path) -> Path | None:
    g = cfg["glossary"]
    if not g.get("enabled"):
        return None
    path = Path(g["path"]) if g.get("path") else Path("output") / f"{source.stem}.glossary.tsv"
    if path.exists():
        print(f"glossary: using existing {path}")
        return path
    if not g.get("auto_generate"):
        print(f"glossary: {path} missing and auto_generate=false -> skipping glossary")
        return None
    print(f"glossary: generating from book (min_freq={g['min_freq']})...")
    docs = glossary_mod.spine_documents(source)
    full_text = "\n".join(text for _, text in docs)
    candidates = glossary_mod.extract_candidates(full_text)
    ordered = sorted(
        ((t, v) for t, v in candidates.items() if v["freq"] >= g["min_freq"]),
        key=lambda kv: (-kv[1]["freq"], kv[0].lower()),
    )
    entries = [
        {"en": en, "zh": "", "freq": v["freq"], "confidence": "", "alt": "", "note": "", "context": v["context"]}
        for en, v in ordered
    ]
    print(f"  candidates: {len(entries)}; resolving via LLM...")
    resolved = glossary_mod.resolve_terms([(e["en"], e["context"]) for e in entries])
    for e in entries:
        r = resolved.get(e["en"], {})
        e["zh"], e["confidence"] = r.get("zh", ""), r.get("confidence", "")
        e["alt"], e["note"] = r.get("alt", ""), r.get("note", "")
    entries = [e for e in entries if e["confidence"] != "skip"]
    entries = glossary_mod.reconcile(entries, allow_llm=True)
    entries.sort(key=lambda e: (-e["freq"], e["en"].lower()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("english\tchinese\tfreq\tconfidence\talternatives\tnote\tcontext\n")
        for e in entries:
            ctx = e["context"].replace("\t", " ").replace("\n", " ")
            f.write(f'{e["en"]}\t{e["zh"]}\t{e["freq"]}\t{e["confidence"]}\t{e.get("alt","")}\t{e.get("note","")}\t{ctx}\n')
    print(f"  glossary written: {path} ({len(entries)} terms)")
    return path


def main() -> None:
    load_dotenv()
    drop_unsupported_proxy_env()

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("translate_book.yaml")
    cfg = load_config(config_path)

    source = Path(cfg["source"])
    if not source.exists():
        raise FileNotFoundError(f"Source EPUB not found: {source}")

    lang = cfg["language"]
    output = Path(cfg["output"]) if cfg.get("output") else Path("output") / f"{source.stem}.{lang}-bilingual.epub"
    output.parent.mkdir(parents=True, exist_ok=True)

    cache_path = cfg["cache_path"] or f".cache/{source.stem}"

    print(f"config:      {config_path}")
    print(f"source:      {source}")
    print(f"output:      {output}")
    print(f"language:    {lang}   submit: {cfg['submit']}   concurrency: {cfg['concurrency']}")
    print(f"cache:       {cache_path}  (resumable)")

    # 1. glossary
    glossary_path = ensure_glossary(cfg, source)
    base_prompt = cfg["user_prompt"] or DEFAULT_USER_PROMPT
    user_prompt = compose_user_prompt(base_prompt, glossary_path)

    # 1b. FRONT MATTER: books start with pages nobody wants translated — cover,
    #     praise, title page, copyright, dedication, contents, preface and the
    #     bare "Part I" divider. Detecting them automatically (default on) means
    #     no per-book hand-maintained list; the detector prints its verdict and
    #     the layer behind it for every spine document so a misjudgement is
    #     visible before a single token is spent. `front_matter_keep_ids` forces
    #     individual documents back into translation; `skip_front_matter: false`
    #     turns the whole thing off and restores the hand-listed behaviour.
    exclude_ids = list(cfg["exclude_spine_ids"])
    if cfg.get("skip_front_matter", True):
        print("front matter: auto-detecting (skip_front_matter: true)")
        report = front_matter_mod.detect(source, cfg.get("front_matter_keep_ids") or [])
        front_matter_mod.print_report(report)
        exclude_ids += [i for i in report.excluded_ids if i not in exclude_ids]

    # 2. spine trim (excluded docs)
    translate_source = source
    if exclude_ids:
        trimmed = Path(cache_path).with_suffix(".body.epub")
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        translate_source = trim_spine(source, exclude_ids, trimmed)

    # 2b. MASK math, then MASK code. Both replace content the LLM must never see
    #     with inert placeholders, and both put the ORIGINAL bytes back after
    #     translation (step 4).
    #       math  - epub_translator LaTeX-ifies <math> via mathml2latex: inline
    #               math comes back as unrenderable <m:math> and display math /
    #               matrices leak as literal `$$...$$` text.
    #       code  - "please don't translate code" in the prompt is not a
    #               guarantee; a single chapter here has 1366 inline <code>
    #               elements, and one slip renames an identifier in the Chinese
    #               text or collapses the inline markup.
    #     ORDER: mask math first, code second; restore in the REVERSE order
    #     (LIFO, see step 4). The two placeholder alphabets are mutually inert
    #     (a MATHPLACEHOLDER sentinel contains no <pre>/<code> markup, a code
    #     placeholder contains no <math>), so masking could run either way
    #     round; but whichever masker runs LAST captures the other's sentinels
    #     inside its own mapping (a <math> inside a <pre> is already a math
    #     sentinel when the <pre> is captured), and those sentinels only return
    #     to the document when that mapping is restored — so the last masker
    #     must be the first restorer.
    #     Set `mask_math: false` / `mask_code: false` in the config to skip.
    math_mapping: dict[int, str] = {}
    code_mapping: dict[int, str] = {}
    if cfg.get("mask_math", True):
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        masked = Path(cache_path) / f"{source.stem}.masked.epub"
        math_mapping, n_masked = mask_math_mod.mask_epub(translate_source, masked)
        print(f"math: masked {n_masked} <math> element(s) before translation")
        if n_masked:
            translate_source = masked
        else:
            math_mapping = {}
    if cfg.get("mask_code", True):
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        masked = Path(cache_path) / f"{source.stem}.masked-code.epub"
        code_mapping, n_masked = mask_code_mod.mask_epub(translate_source, masked)
        print(f"code: masked {n_masked} <pre>/<code> element(s) before translation")
        if n_masked:
            translate_source = masked
        else:
            code_mapping = {}

    # translate into a temp file when anything has to be restored into `output`
    translate_target = output
    if math_mapping or code_mapping:
        translate_target = Path(cache_path) / f"{source.stem}.translated.epub"

    # 3. translate with a live progress bar
    import os
    os.environ["EPUB_TRANSLATOR_CACHE_PATH"] = cache_path

    bar = tqdm(total=100, desc="translating", unit="%", bar_format="{l_bar}{bar}| {n:.1f}/100 [{elapsed}<{remaining}]")
    last = 0.0

    def on_progress(progress: float) -> None:
        nonlocal last
        pct = progress * 100
        bar.update(max(0.0, pct - last))
        last = pct

    translate(
        source_path=translate_source,
        target_path=translate_target,
        target_language=LANGUAGES[lang],
        submit=SUBMIT_KINDS[cfg["submit"]],
        user_prompt=user_prompt,
        max_group_tokens=cfg["max_group_tokens"],
        concurrency=cfg["concurrency"],
        llm=build_llm(),
        on_progress=on_progress,
    )
    bar.update(max(0.0, 100 - last))
    bar.close()

    # 4. RESTORE, in the reverse order of masking (code first, then math — see
    #    step 2b). Every placeholder is put back at EVERY occurrence: append-block
    #    keeps the source block and appends its translation, so an inline
    #    placeholder shows up at least twice and restoring only the first would
    #    lose the code/formula from the translated half of the book.
    stages = []
    if code_mapping:
        stages.append(("code", mask_code_mod.restore_epub, code_mapping))
    if math_mapping:
        stages.append(("math", mask_math_mod.restore_epub, math_mapping))

    current = translate_target
    for i, (name, restore_epub, mapping) in enumerate(stages):
        is_last = i == len(stages) - 1
        dest = output if is_last else Path(cache_path) / f"{source.stem}.restored-{name}.epub"
        restored = restore_epub(current, dest, mapping)
        print(f"{name}: restored {restored} occurrence(s) of {len(mapping)} element(s)")
        # Every masked element must come back at least once; fewer occurrences
        # than elements means the translator dropped a placeholder and that
        # formula/listing is missing from the book.
        if restored < len(mapping):
            print(f"  WARNING: {len(mapping) - restored} {name} placeholder(s) never came back")
        current = dest

    # 5. RESTORE spine: excluded docs were dropped from the spine only to keep
    #    them out of translation; put them back so they stay in the reading order
    #    (present but untranslated), not orphaned in the archive.
    if exclude_ids:
        restored_ids = restore_spine(output, source, exclude_ids)
        print(f"spine: restored excluded docs into output: {restored_ids or 'none'}")

    print(f"\n✅ converted: {output}")


if __name__ == "__main__":
    main()
