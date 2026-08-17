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

Then patch the dependencies (see [Local dependency patches](#local-dependency-patches) —
this is required, and required again after every `uv sync`):

```bash
uv run python apply_patches.py
```

### Local dependency patches

`epub-translator` ships two bugs that are fatal for a book-length run, fixed here in
`patches/*.patch` rather than upstream:

* **`<|endoftext|>` kills the process.** A book *about* language models quotes
  `<|endoftext|>`, `<|fim_prefix|>` and friends as ordinary prose. `tiktoken` refuses to
  encode those and raises `ValueError: Encountered text corresponding to disallowed
  special token`, which takes down the whole run — measured at 60% through *Hands-On
  Large Language Models*, with no way to resume past it. The patches pass
  `disallowed_special=()` at all five call sites (`xml_translator/score.py` ×3,
  `hill_climbing.py`, `validation.py`), so the markers count as the plain text they are.
  Fixing only `score.py` just moves the crash to `hill_climbing.py`.
* **429 / 500 / 529 were not retried.** `EPUB_TRANSLATOR_RETRY_TIMES` only ever applied
  to 502/503/504, so a rate limit ended the run. The patch adds 429, 500 and 529 (and
  keeps the 520–527 Cloudflare range). 402 "insufficient balance" and the 4xx credential
  errors still fail on the first response — retrying those only delays the error you
  have to act on.

`uv sync` restores the pristine wheel and silently drops both fixes, so run this after
every sync:

```bash
uv run python apply_patches.py           # idempotent; safe to run any time
uv run python apply_patches.py --check   # exit 1 if anything is unpatched
```

It refuses to run outside the project virtualenv, and if a patch no longer fits (because
the dependency was upgraded) it says so and exits non-zero instead of leaving a
half-patched environment. Regenerate the patch against the new version before starting a
book.

Do not hand-edit files under `.venv/` instead: `uv` installs packages by *hardlinking*
out of `~/.cache/uv/archive-v0/`, so an in-place edit also rewrites the shared uv cache
and every other project on the machine inherits it. `apply_patches.py` always writes a
new file and `os.replace`s it, which breaks the hardlink and leaves the cache clean.

### DeepSeek

DeepSeek speaks plain OpenAI chat-completions, so there is no DeepSeek provider — keep
`EPUB_TRANSLATOR_PROVIDER=openai` and change three lines in `.env`:

```bash
EPUB_TRANSLATOR_PROVIDER=openai
EPUB_TRANSLATOR_API_KEY=sk-your-deepseek-key
EPUB_TRANSLATOR_BASE_URL=https://api.deepseek.com/v1
EPUB_TRANSLATOR_MODEL=deepseek-v4-flash
EPUB_TRANSLATOR_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
```

* **The `/v1` in `BASE_URL` is not optional.** The two stages build the URL differently:
  the translation stage hands the value to the openai SDK, which appends
  `/chat/completions` verbatim, while `glossary.py` adds a `/v1` itself when the base
  lacks one. Drop the `/v1` and the two stages talk to two different paths, one of which
  404s.
* **`deepseek-v4-flash`** is the sensible default; `deepseek-v4-pro` is the stronger,
  pricier one. `deepseek-chat` and `deepseek-reasoner` are retired.
* **Thinking mode is on by default**, and it is pure overhead here. Measured on one real
  translation request: default = 291 reasoning tokens / 321 completion tokens / 3.4s,
  versus 0 / 27 / 1.1s with thinking disabled — same translation quality for this job.
  The reasoning text never reaches the translation either way (the library streams and
  reads only `delta.content`), so it is cost and latency for nothing.
  `EPUB_TRANSLATOR_EXTRA_BODY` turns it off.
* **Concurrency 16** is fine against `api.deepseek.com` directly; drop to 4–8 if 429s
  start showing up.

**Quoting `EPUB_TRANSLATOR_EXTRA_BODY` in `.env`** — python-dotenv, as tested:

| form | result |
|---|---|
| `EXTRA_BODY='{"thinking": {"type": "disabled"}}'` | works — **use this** |
| `EXTRA_BODY={"thinking":{"type":"disabled"}}` | works (bare is fine) |
| either of the above with a trailing `# comment` | works |
| `EXTRA_BODY="{"thinking": {"type": "disabled"}}"` | **breaks** — dotenv reports "could not parse statement", drops the variable, and the run proceeds with thinking still on |

Count the braces: the value needs two closing ones. A truncated value fails at startup
with a message naming the variable and quoting the value back, before any request is
sent — it never wastes a run.

`EPUB_TRANSLATOR_EXTRA_BODY` is a general escape hatch, not a DeepSeek feature: whatever
JSON object you put there is merged into every chat-completions request body, in both
the glossary and the translation stage. Unset or empty, the request body is exactly what
it was before. Malformed JSON fails at startup with the offending value quoted back at
you, not a traceback. The `claude-code` and `anthropic` backends do not build an
OpenAI-style body and print a warning if it is set.

### Alternative engine: the local Claude Code CLI

Instead of an HTTP API you can drive the `claude` CLI that is already installed and
logged in on this machine. No API key, no base URL:

```bash
EPUB_TRANSLATOR_PROVIDER=claude-code
EPUB_TRANSLATOR_CLAUDE_CODE_MODEL=sonnet   # or opus / haiku / a full model name
```

Every translation request becomes one `claude -p` subprocess, started in an empty
temporary directory with `--safe-mode --tools "" --max-turns 1`, so the agent has no
tools, no MCP servers, no skills and no CLAUDE.md. It is *not* a plain translation
model, though, and it is worth knowing why before you point it at a book:

* **The harness still injects its own preamble.** Even with `--safe-mode` and
  `--system-prompt-file`, roughly 160 tokens go in ahead of your system prompt —
  measured: a one-character system prompt plus a one-character user turn still reports
  163 input tokens, and asking the model to echo them back returns the product identity,
  today's date and **the logged-in user's e-mail address**. Treat every request as
  attributable, not anonymous.
* **It keeps its chat reflexes.** Handed a bare short input such as `Part II` or
  `Preface`, it answers "I notice you haven't included the source text to translate…"
  about a third of the time. `claude_code_llm.py` therefore hardens the translation
  system prompt and labels the user turn, and validates every response before returning
  it, so a chat-style non-answer is retried instead of being written into `.cache`
  forever.

Requests go through the same on-disk cache, so a crashed run resumes. Ctrl-C is safe:
SIGINT/SIGTERM kill every live `claude` process group and remove its temp directory
before exiting. Measured on this machine: ~4s per short request, and 8 concurrent
requests also finish in ~4s total; each process is ~290MB resident, so cap them with
`EPUB_TRANSLATOR_CLAUDE_CODE_MAX_CONCURRENCY` on a small machine. See `.env.example`
for the timeout/retry knobs.

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
skip_front_matter: true   # auto-skip cover/title/copyright/dedication/toc/preface/part pages
exclude_spine_ids: []     # extra spine ids to skip (e.g. endnotes, index); [] = none
user_prompt: |            # appended to the system prompt — domain/style hints
  这是一本机器人学技术书，术语统一、公式与代码原文保留。
```

### 前置页面默认不翻译

翻译从正文第一章开始。封面、赞誉/题献页、书名页、版权页、目录、前言，以及只有一行标题的
Part 分隔页，`front_matter.py` 会自动识别并跳过——它们照旧原样留在成品的阅读顺序里，只是不译。

识别分层，从权威到启发式：nav `landmarks` → OPF `<guide>` → 文档自己的 `epub:type` →
spine id/文件名 → 体量。"正文从哪开始"的指针（`landmarks bodymatter`、`guide type="text"`）
只在它没指向一个本身就是前置页面的文档时才采信——实测 O'Reilly 把 `type="text"` 指到书名页、
企鹅兰登把 `bodymatter` 指到题献页，照单全收就会漏排一半前置页。安全上宁可漏排不可错排：
泛泛的 `epub:type="frontmatter"` 单独不足以排除（Reentry 的 7000 字 Prologue 就挂着这个标），
文件名匹配对长文档不生效，前置页面只能是 spine 的连续前缀。

每本书跑之前都会把 9 行/26 行的判定表打印出来（每个文档：排还是留、依据哪一层），误判一眼可见。
误判时用 `front_matter_keep_ids: [该文档的 spine id]` 强制翻译它，或 `skip_front_matter: false`
整个关掉。单独审计一本书：

```bash
uv run python front_matter.py "input/My Book.epub"
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

### Code is never translated

Same trick, applied to source code (`mask_code.py`, on by default). Telling the
LLM "don't translate code" is not a guarantee: one chapter of *Hands-On Large
Language Models* carries 1,366 inline `<code>` elements, and a single slip
renames an identifier inside the Chinese text or collapses the inline markup
when the paragraph is re-assembled. So the model never sees the code:

* every **`<pre>` listing** (including the `<code>` spans nested in it) is
  swapped for an EMPTY `<pre data-codemask="N"></pre>`. Empty means the
  translator never visits it, which is what keeps append-block from emitting
  every listing twice — the block passes through untouched and is restored
  exactly once, in place;
* every **inline `<code>`** becomes a `CODEPLACEHOLDERnnnnnX` sentinel that
  stays inside the sentence, so the prose around it is still translated with
  the code pinned at its original position.

Afterwards the ORIGINAL bytes are put back at every occurrence — entities,
whitespace and syntax-highlight spans included — so the code in the output is
byte-for-byte the code in the source. Round-tripping the reference book without
an LLM (844 elements: 55 `<pre>` blocks + 789 inline `<code>`) reproduces the
source EPUB byte-identically.

Set `mask_code: false` in the YAML config to opt out. The two maskers stack:
math is masked first and code second, and they are restored in the reverse
order (code, then math), because the second masker can capture the first one's
sentinels inside its mapping (a `<math>` inside a `<pre>`).


