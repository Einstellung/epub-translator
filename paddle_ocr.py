"""Run PaddleOCR-VL over a PDF and write one Markdown file plus its image assets.

This file lives in the repo but runs in its OWN virtualenv (`.venv-paddle`):
`paddlex` pins `pyyaml==6.0.2` while this project needs `pyyaml>=6.0.3`, so the
two dependency trees cannot share a venv. `pdf_to_epub.py --engine paddle`
therefore invokes this script as a subprocess with that interpreter.

Output contract (the same shape `pdf_craft.transform_markdown` produces, so the
rest of the pipeline is unchanged):

  * one Markdown file at `markdown_path`, pages concatenated in order;
  * images written under `assets_path`, referenced from the Markdown by paths
    that `pdf_to_epub._absolutize_images` can resolve.

Usage:
    .venv-paddle/bin/python paddle_ocr.py book.pdf out.md out_assets/
"""

import argparse
import sys
import time
from pathlib import Path


def _page_markdown(res) -> tuple[str, dict]:
    """Return (markdown text, {relative path: PIL image}) for one page result."""
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return md.get("markdown_texts", ""), md.get("markdown_images", {}) or {}
    if isinstance(md, str):
        return md, {}
    raise RuntimeError(f"unexpected markdown payload on {type(res).__name__}: {type(md)}")


def run(
    pdf_path: Path,
    markdown_path: Path,
    assets_path: Path,
    device: str = "gpu:0",
    pipeline_version: str = "v1.6",
) -> None:
    from paddleocr import PaddleOCRVL

    t0 = time.time()
    pipeline = PaddleOCRVL(pipeline_version=pipeline_version, device=device)
    print(f"      model loaded in {time.time() - t0:.1f}s", flush=True)

    results = list(pipeline.predict(str(pdf_path)))

    # Cross-page restructuring stitches a table (or paragraph) split by a page
    # break back together; it is a no-op for single-page input.
    restructure = getattr(pipeline, "restructure_pages", None)
    if restructure is not None and len(results) > 1:
        try:
            results = list(restructure(results, merge_tables=True))
        except Exception as exc:  # pragma: no cover - depends on pipeline version
            print(f"      warning: restructure_pages failed ({exc}); using raw pages",
                  file=sys.stderr, flush=True)

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    saved = 0
    for res in results:
        text, images = _page_markdown(res)
        texts.append(text)
        for rel, image in images.items():
            target = assets_path / Path(rel).name
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            saved += 1

    join = getattr(pipeline, "concatenate_markdown_pages", None)
    if join is not None:
        try:
            merged = join(texts)
        except Exception:  # pragma: no cover - depends on pipeline version
            merged = "\n\n".join(texts)
    else:
        merged = "\n\n".join(texts)
    if isinstance(merged, dict):  # some versions return the markdown dict again
        merged = merged.get("markdown_texts", "")

    markdown_path.write_text(merged, encoding="utf-8")
    print(f"      {len(results)} page(s), {saved} image(s) -> {markdown_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PaddleOCR-VL: PDF -> Markdown + assets")
    p.add_argument("pdf", type=Path)
    p.add_argument("markdown", type=Path)
    p.add_argument("assets", type=Path)
    p.add_argument("--device", default="gpu:0")
    p.add_argument("--pipeline-version", default="v1.6")
    args = p.parse_args(argv)
    run(args.pdf, args.markdown, args.assets, args.device, args.pipeline_version)


if __name__ == "__main__":
    main()
