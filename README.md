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

`translate_book.yaml` in the repo root is a **template** — don't edit it directly.
Keep one config file per book under `configs/` and load the one you want:

```bash
cp translate_book.yaml configs/my-book.yaml      # one-time per book
# edit configs/my-book.yaml to point at your EPUB
uv run python translate_book.py configs/my-book.yaml
```

Running `translate_book.py` with no argument falls back to the root template.
A book config points at the source EPUB and tunes glossary/exclusions/style:

```yaml
source: "output/My Book.epub"
concurrency: 16            # 16 is fast and not rate-limited in practice
glossary:
  enabled: true
  auto_generate: true     # extract + resolve a glossary from the book on first run
  min_freq: 2
exclude_spine_ids: []     # spine ids to skip (e.g. endnotes, index); [] = none
user_prompt: |            # appended to the system prompt — domain/style hints
  这是一本机器人学技术书，术语统一、公式与代码原文保留。
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
uv run python pdf_to_epub.py input/paper.pdf --split-references   # 论文：拆出参考文献
```

**论文场景加 `--split-references`**：把 References/Bibliography 段拆成独立的
`references.xhtml`，登记进 OPF 的 manifest 和 spine（idref 为 `references`）。学术 PDF 转出的
EPUB 是单文件（整篇一个 `ch001.xhtml`），拆开后书目才有独立 spine id，翻译器就能用
`exclude_spine_ids` 把它排除在翻译外、又保留在成品里（`translate_book.yaml` 默认已排除
`references`）。带体量自检：万一误判标题会保留原样、不动正文。标题识别 References /
Bibliography / 参考文献（忽略大小写与前导编号）。旧的 `--strip-references` 作为别名保留。

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

### Translating a math EPUB: math is masked automatically

`epub_translator` does **not** pass MathML through cleanly. Internally
(`translation/xml_interrupter.py`) it converts every `<math>` element to LaTeX
via `mathml2latex` (which spams `Unknown Tag appeared!! ... semantics`, because
it can't parse the MathML `<semantics>` wrapper) and hands the LaTeX to the LLM.
The result: inline math comes back as a prefixed `<m:math>` form most readers
won't render, and **display math / matrices leak into the output as literal
`$$...$$` LaTeX text** (sometimes an empty `\begin{array}` — the structure is
lost entirely).

`translate_book.py` fixes this end-to-end with a **mask → translate → restore**
wrapper (`mask_math.py`): before translation every `<math>` element is replaced
with an inert `MATHPLACEHOLDERnnnnX` sentinel the translator/LLM pass through
verbatim; afterwards the original, default-namespaced MathML is substituted
back into every occurrence (append-block duplicates each block, so all copies
are restored). Math never reaches the LLM, so it survives as renderable
`<math xmlns="…MathML">` with subscripts/superscripts/matrices intact.

```bash
uv run python pdf_to_epub.py input/book.pdf -o output/book.epub
uv run python translate_book.py configs/book.yaml     # masking is on by default
```

This is automatic — no post-processing step needed. Set `mask_math: false` in
the YAML config to opt out. `fix_epub_math.py` remains as a fallback repair for
EPUBs produced by the older `main.py` path (it only re-namespaces surviving
`<m:math>` and cannot recover display math that already leaked to LaTeX; the
mask wrapper prevents the leak in the first place).


