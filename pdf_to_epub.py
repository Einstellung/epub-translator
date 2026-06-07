"""Convert a PDF into a clean, math-correct EPUB ready for translation.

Pipeline (every step here was validated against real pages, and every
workaround encodes a bug we actually hit):

  1. pdf_craft.transform_markdown  -> Markdown + extracted image assets
     (DeepSeek-OCR; it recognises prose, matrices and inline math well —
      code blocks are its weak spot, expect to hand-check those.)
  2. fix LaTeX over-escaping        -> pdf_craft doubles every command
     backslash inside math spans (\\\\cos, \\\\begin). We restore \\\\<letter>
     to \\<letter> while preserving real \\\\ matrix row-breaks.
  3. rewrite image paths to absolute -> pdf_craft's relative
     markdown_assets_path produces references that don't line up with where
     the files actually land, so pandoc can't embed them. Absolute paths fix it.
  4. pandoc --mathml                 -> turns the (now valid) LaTeX into
     MathML. pandoc's matrix handling is correct where pdf_craft's own
     MathML/SVG/CLIPPING renderers dropped or flattened matrices.
  5. strip <annotation> elements     -> pandoc embeds a raw-LaTeX annotation
     beside each MathML formula. Readers that don't fully support
     <semantics> print that annotation as body text, so every formula shows
     up twice. We remove them.
  6. repackage                       -> rebuild the EPUB zip with mimetype
     stored first and uncompressed, as the spec requires.

Usage:
    uv run python pdf_to_epub.py input/book.pdf
    uv run python pdf_to_epub.py input/book.pdf -o output/book.epub --ocr-size base
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from pdf_craft import transform_markdown

# DeepSeek-OCR resolution tiers (see doc_page_extractor/model.py):
#   tiny=512  small=640  base=1024  large=1280  gundam=1024+640 crop
# base is the sweet spot for normally-typeset books on a 12GB GPU
# (peak ~9GB, no crop). gundam (the pdf_craft default) crops into tiles and
# is better for dense/small-font/scanned pages but costs more VRAM and time.
DEFAULT_OCR_SIZE = "base"

# pdf_craft caches OCR models here; reused across runs so we download once.
MODELS_CACHE = "models"


def _de_escape_latex(text: str) -> str:
    """Undo pdf_craft's double-backslash over-escaping inside math spans.

    pdf_craft emits math as \\[ ... \\] and \\( ... \\) but doubles every
    command backslash inside (\\\\cos -> should be \\cos). A genuine matrix
    row-break is also \\\\, but it is followed by whitespace or a brace, not a
    letter, so de-escaping only \\\\<letter> leaves row-breaks intact.
    """

    def fix_span(m: re.Match) -> str:
        return re.sub(r"\\\\([A-Za-z])", r"\\\1", m.group(0))

    text = re.sub(r"\\\[.*?\\\]", fix_span, text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", fix_span, text, flags=re.S)
    return text


def _absolutize_images(text: str, md_dir: Path) -> str:
    """Rewrite ![](rel/path) image links to absolute paths.

    pdf_craft's relative markdown_assets_path does not line up with where the
    asset files actually land (a nested pdf_test/pdf_test/assets/ layer), so
    pandoc silently fails to embed them. We resolve each link against the
    locations pdf_craft might have used and rewrite to an absolute path that
    exists, so pandoc can always find and embed the image.
    """

    def resolve(m: re.Match) -> str:
        rel = m.group(1)
        if rel.startswith(("http://", "https://", "/")):
            return m.group(0)
        name = Path(rel).name
        candidates = [
            md_dir / rel,
            md_dir / Path(rel).name,
            md_dir / "assets" / name,
            md_dir / md_dir.name / "assets" / name,  # nested-dir bug
            md_dir / "_tmp" / "assets" / name,
        ]
        for c in candidates:
            if c.is_file():
                return f"![]({c.resolve()})"
        # fall back to a recursive search under the markdown dir
        hits = list(md_dir.rglob(name))
        if hits:
            return f"![]({hits[0].resolve()})"
        print(f"  warning: image not found, leaving as-is: {rel}", file=sys.stderr)
        return m.group(0)

    return re.sub(r"!\[\]\(([^)]*)\)", resolve, text)


def _strip_annotations(epub_dir: Path) -> int:
    """Remove <annotation> LaTeX fallbacks from every xhtml file.

    pandoc wraps each MathML formula in <semantics> with an
    <annotation encoding="application/x-tex"> holding the raw LaTeX. Readers
    that don't fully support <semantics> render that annotation as body text,
    so every formula appears twice. Removing it leaves only the MathML.
    """
    removed = 0
    for xhtml in epub_dir.rglob("*.xhtml"):
        html = xhtml.read_text(encoding="utf-8")
        n = html.count("<annotation")
        if not n:
            continue
        html = re.sub(r"<annotation\b[^>]*>.*?</annotation>", "", html, flags=re.S)
        xhtml.write_text(html, encoding="utf-8")
        removed += n
    return removed


def _repackage_epub(epub_dir: Path, epub_path: Path) -> None:
    """Rebuild the EPUB zip with mimetype stored first and uncompressed."""
    if epub_path.exists():
        epub_path.unlink()
    with zipfile.ZipFile(epub_path, "w") as zf:
        # mimetype must be the first entry and stored (not deflated)
        zf.write(epub_dir / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in sorted(epub_dir.rglob("*")):
            if path.is_dir() or path.name == "mimetype":
                continue
            arc = path.relative_to(epub_dir).as_posix()
            zf.write(path, arc, compress_type=zipfile.ZIP_DEFLATED)


def _check_pandoc() -> None:
    if shutil.which("pandoc") is None:
        sys.exit(
            "error: pandoc is required but not found on PATH.\n"
            "Install it (e.g. `sudo apt install pandoc`) and retry."
        )


def convert(
    pdf_path: Path,
    epub_path: Path,
    ocr_size: str = DEFAULT_OCR_SIZE,
    work_dir: Path | None = None,
    keep_work: bool = False,
    title: str | None = None,
) -> Path:
    """Run the full PDF -> clean EPUB pipeline. Returns the EPUB path."""
    _check_pandoc()
    pdf_path = pdf_path.resolve()
    epub_path = epub_path.resolve()
    epub_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = (work_dir or epub_path.parent / f".{epub_path.stem}.work").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    md_path = work_dir / "book.md"
    assets_path = work_dir / "assets"
    analysing_path = work_dir / "_ocr"

    # 1. OCR -> Markdown
    print(f"[1/6] OCR {pdf_path.name} -> markdown (ocr_size={ocr_size}) ...", flush=True)
    t0 = time.time()
    transform_markdown(
        pdf_path=str(pdf_path),
        markdown_path=str(md_path),
        markdown_assets_path=str(assets_path),
        analysing_path=str(analysing_path),
        models_cache_path=MODELS_CACHE,
        ocr_size=ocr_size,
        includes_footnotes=True,
    )
    print(f"      OCR done in {time.time() - t0:.1f}s", flush=True)

    # 2. fix LaTeX over-escaping  3. absolutize image paths
    print("[2/6] fixing LaTeX escaping", flush=True)
    text = md_path.read_text(encoding="utf-8")
    text = _de_escape_latex(text)
    print("[3/6] resolving image paths", flush=True)
    text = _absolutize_images(text, md_path.parent)
    md_path.write_text(text, encoding="utf-8")

    # 4. pandoc -> EPUB with MathML
    print("[4/6] pandoc -> EPUB (MathML)", flush=True)
    raw_epub = work_dir / "raw.epub"
    cmd = [
        "pandoc", str(md_path), "-o", str(raw_epub),
        "--mathml", "-f", "markdown+tex_math_single_backslash",
        "--metadata", f"title={title or pdf_path.stem}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"pandoc failed:\n{result.stderr}")

    # 5. strip annotations  6. repackage
    print("[5/6] stripping LaTeX annotation duplicates", flush=True)
    epub_extract = work_dir / "epub"
    if epub_extract.exists():
        shutil.rmtree(epub_extract)
    epub_extract.mkdir()
    with zipfile.ZipFile(raw_epub) as zf:
        zf.extractall(epub_extract)
    removed = _strip_annotations(epub_extract)
    print(f"      removed {removed} annotation(s)", flush=True)

    print("[6/6] repackaging EPUB", flush=True)
    _repackage_epub(epub_extract, epub_path)

    if not keep_work:
        shutil.rmtree(work_dir)

    print(f"\ndone -> {epub_path}", flush=True)
    return epub_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert a PDF to a clean, math-correct EPUB for translation."
    )
    p.add_argument("pdf", type=Path, help="source PDF path")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output EPUB path (default: output/<pdf-stem>.epub)",
    )
    p.add_argument(
        "--ocr-size", default=DEFAULT_OCR_SIZE,
        choices=["tiny", "small", "base", "large", "gundam"],
        help="DeepSeek-OCR resolution tier (default: base)",
    )
    p.add_argument(
        "--keep-work", action="store_true",
        help="keep the intermediate work directory for debugging",
    )
    args = p.parse_args(argv)

    if not args.pdf.is_file():
        sys.exit(f"error: PDF not found: {args.pdf}")
    epub_path = args.output or Path("output") / f"{args.pdf.stem}.epub"
    convert(
        pdf_path=args.pdf,
        epub_path=epub_path,
        ocr_size=args.ocr_size,
        keep_work=args.keep_work,
    )


if __name__ == "__main__":
    main()


