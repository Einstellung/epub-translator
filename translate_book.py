"""One-command bilingual book translator driven by a YAML config.

Pipeline:
  1. (optional) build/load a book-level glossary  -> glossary.py
  2. (optional) make a spine-trimmed copy of the EPUB so excluded
     documents (endnotes, index, ...) are skipped
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

import glossary as glossary_mod
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

    # 2. spine trim (excluded docs)
    translate_source = source
    if cfg["exclude_spine_ids"]:
        trimmed = Path(cache_path).with_suffix(".body.epub")
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        translate_source = trim_spine(source, cfg["exclude_spine_ids"], trimmed)

    # 2b. MASK math: replace every <math> element with an inert sentinel token so
    #     the translator never sees (and never mangles/flattens) the MathML. The
    #     original formulas are restored verbatim after translation (see step 4).
    #     Without this, epub_translator LaTeX-ifies math via mathml2latex: inline
    #     math comes back as unrenderable <m:math> and display math/matrices leak
    #     as literal `$$...$$` text. Set `mask_math: false` in the config to skip.
    math_mapping: dict[int, str] = {}
    translate_target = output
    if cfg.get("mask_math", True):
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        masked = Path(cache_path) / f"{source.stem}.masked.epub"
        math_mapping, n_masked = mask_math_mod.mask_epub(translate_source, masked)
        print(f"math: masked {n_masked} <math> element(s) before translation")
        if n_masked:
            translate_source = masked
            # translate into a temp file, then restore math into `output`
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

    # 4. RESTORE math: put the original MathML back in place of every sentinel
    #    token (each appears >=1x; append-block duplicates it, so restore ALL).
    if math_mapping:
        restored = mask_math_mod.restore_epub(translate_target, output, math_mapping)
        print(f"math: restored {restored} MathML occurrence(s) into {output}")

    print(f"\n✅ converted: {output}")


if __name__ == "__main__":
    main()
