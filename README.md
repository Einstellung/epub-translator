## EPUB bilingual converter

This project wraps `epub-translator` with `uv` and environment-based OpenAI-compatible API settings.

### Setup

```bash
cp .env.example .env
```

Edit `.env`:

```bash
EPUB_TRANSLATOR_API_KEY=your transit/proxy key
EPUB_TRANSLATOR_BASE_URL=https://your-proxy.example.com/v1
EPUB_TRANSLATOR_MODEL=the-model-name-supported-by-your-proxy
```

### Convert

Put EPUB files in `input/`, then run:

```bash
uv run python main.py input/book.epub
```

The default output is:

```text
output/book.zh-bilingual.epub
```

Useful options:

```bash
uv run python main.py input/book.epub --concurrency 1
uv run python main.py input/book.epub --submit append-text
uv run python main.py input/book.epub --submit replace
uv run python main.py input/book.epub --language ja
uv run python main.py input/book.epub -o output/custom.epub
```

`append-block` is the default bilingual mode. It keeps the original text and adds translated blocks after it.

## One-command translation (YAML + progress bar)

For translating a whole book, use the YAML-driven runner instead of long CLI flags.
It builds a book-level glossary, skips chapters you don't want (endnotes, index),
shows a live progress bar, and is resumable.

```bash
uv run python translate_book.py                 # reads translate_book.yaml
uv run python translate_book.py my_book.yaml    # or a custom config
```

Edit `translate_book.yaml` to point at your book:

```yaml
source: "input/My Book.epub"
concurrency: 16            # 16 is fast and not rate-limited in practice
glossary:
  enabled: true
  auto_generate: true     # extract + resolve a glossary from the book on first run
  min_freq: 2
exclude_spine_ids:        # documents to skip (their spine ids)
  - endnotes
  - index
```

Every term/name in the glossary is rendered consistently across the whole book.
The run caches per book under `.cache/<book>`, so an interrupted run resumes where
it left off. Generate or inspect a glossary on its own with:

```bash
uv run python glossary.py "input/My Book.epub" --min-freq 2
```

## PDF to EPUB (math-aware)

Have a PDF instead of an EPUB? Convert it first, then feed the EPUB into the
translator above. This uses `pdf_craft` + DeepSeek-OCR for layout/text/formula
recognition and `pandoc` for a clean, math-correct EPUB.

```bash
uv run python pdf_to_epub.py input/book.pdf                 # -> output/book.epub
uv run python pdf_to_epub.py input/book.pdf -o output/my.epub --ocr-size base
```

Requirements: `pandoc` on PATH (`sudo apt install pandoc`) and an NVIDIA GPU.
The DeepSeek-OCR model (~6.3 GB) downloads once into `models/` and is reused.

What the pipeline does, and the gotchas it handles automatically:

1. **OCR -> Markdown** via DeepSeek-OCR. Prose, matrices and inline math come
   out well; **code blocks are the weak spot** — structure and identifiers get
   mangled, so hand-check any code after conversion.
2. **Fixes LaTeX over-escaping** — pdf_craft doubles every command backslash
   inside math (`\\cos` -> `\cos`) while preserving real `\\` matrix row-breaks.
3. **Resolves image paths** — pdf_craft's relative asset paths don't line up
   with where files land, so pandoc can't embed them; we rewrite to absolute.
4. **pandoc `--mathml`** — converts the now-valid LaTeX to MathML. pandoc's
   matrix handling is correct where pdf_craft's own renderers flattened or
   dropped matrices.
5. **Strips `<annotation>` duplicates** — pandoc embeds a raw-LaTeX annotation
   next to each formula; some readers print it as body text, doubling every
   formula. We remove them.
6. **Repackages** the EPUB with `mimetype` stored first, per spec.

OCR resolution tiers (`--ocr-size`): `tiny`/`small`/`base`/`large` send the
whole page as one image (512/640/1024/1280 px); `gundam` crops dense pages into
tiles. **`base`** is the default — best quality/VRAM trade-off for normal books
(peak ~9 GB on a 12 GB card). Use `gundam` for dense, small-font or scanned
pages; it recognises more but costs more VRAM and time.

