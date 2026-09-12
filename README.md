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

`epub-translator` ships two bugs that are fatal for a book-length run, plus one
behaviour this project does not want; all three are fixed in `patches/*.patch` rather
than upstream:

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
* **The reader's outline came back bilingual.** Every entry in the sidebar TOC turned
  into source-plus-translation crammed onto one line — `Cover 封面`, `Chapter One:
  Before the Book 第一章：书成之前` — which is unreadable at sidebar width and is not
  what a bilingual build is for. Suspiciously, `toc.ncx` of the same book stayed clean
  English: `epub/toc.py` is a translation channel of its own (`read_toc()` → LLM →
  `write_toc()`), separate from the spine walk, and `_find_toc_path()` picks *exactly
  one* file per book — the nav document for EPUB3, the NCX for EPUB2. So an EPUB3 book
  gets its nav translated and its NCX left alone, and an EPUB2 book the reverse.
  This one cannot be fixed in our own code: the nav document lives only in the OPF
  manifest as `properties="nav"` and is not in the spine, so `front_matter.py` and
  `exclude_spine_ids` — which both walk the spine — never see it, and the library
  exposes no switch. The patch makes `read_toc()` return an empty TOC list, which closes
  the channel at its source: `translate()` then gives the TOC no progress weight, emits
  no TOC task (no tokens spent on it) and never calls `write_toc`. Nav and NCX are
  copied through byte for byte — still in the manifest, still `properties="nav"`, every
  link still resolving, just not translated. Verified end to end on both branches: an
  EPUB3 slice (nav + NCX) and an EPUB2 slice (NCX only), each with the two TOC files
  SHA-identical to the source, zero dangling TOC links, and the chapter text translated
  as usual.

`uv sync` restores the pristine wheel and silently drops all three, so run this after
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
uv run python polish_cjk.py output/my-book.zh-bilingual.epub out.epub   # see below
```

**For a Chinese target, `translate_book.py` is not the last step.** Publisher
stylesheets set a Latin-only `font-family` (Verdana, Arial), so every Han
character falls back to whatever font the reading app happens to pick — the
translation renders lighter and larger than the surrounding English, at a line
height meant for Latin. Run `polish_cjk.py` afterwards; see
[CJK typography pass](#cjk-typography-pass-polish_cjkpy).

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

识别分层，从权威到启发式：nav `landmarks` → OPF `<guide>` → 文档自己的 `epub:type`
（`epub:type="toc"` 和 DPUB-ARIA 的 `role="doc-toc"` 一视同仁）→ spine id/文件名 → 体量。
"正文从哪开始"的指针（`landmarks bodymatter`、`guide type="text"`）只在它没指向一个本身就是
前置页面的文档时才采信——实测 O'Reilly 把 `type="text"` 指到书名页、企鹅兰登把 `bodymatter`
指到题献页，照单全收就会漏排一半前置页。安全上宁可漏排不可错排：泛泛的
`epub:type="frontmatter"` 单独不足以排除（Reentry 的 7000 字 Prologue 就挂着这个标），
文件名匹配对长文档不生效，前置页面只能是 spine 的连续前缀。

**目录页一律不翻译**，且不受"连续前缀"的限制：无论它排在 spine 的哪个位置（Calibre 和企鹅兰登
都会把 inline 目录塞在正文起点之后），只要判定为目录就跳过。判定不看别人怎么标，只看文档本身
像不像目录——**短**（≤ 6000 字符）**且链接密集**（链接文字占比 ≥ 0.30）。两个条件缺一不可：
Reentry 的 OPF 把 `<reference type="toc">` 指到了 7000 字的叙事 Prologue，O'Reilly 的正文章节
因为交叉引用多，链接占比高达 0.47–0.58，只靠其中一个条件就会把正文当目录扔掉。

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
translator above. Layout, text and formula recognition run through PaddleOCR-VL
1.6 by default, with DeepSeek-OCR available as `--engine deepseek`; `pandoc`
turns the result into a clean, math-correct EPUB. The default engine needs an
inference server running first — see
[OCR engines](#ocr-engines---engine-paddle-default-and---engine-deepseek).

```bash
# start the PaddleOCR-VL server once, leave it running (~7 GB of VRAM)
VIRTUAL_ENV=.venv-paddle .venv-paddle/bin/paddleocr genai_server \
    --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --port 8118 \
    --backend_config vllm_12gb.yaml

uv run python pdf_to_epub.py input/book.pdf                 # -> output/book.epub
uv run python pdf_to_epub.py input/paper.pdf --split-references   # 论文：拆出参考文献
uv run python pdf_to_epub.py input/book.pdf --engine deepseek --ocr-size base
```

**论文场景加 `--split-references`**：把 References/Bibliography 段拆成独立的
`references.xhtml`，登记进 OPF 的 manifest 和 spine（idref 为 `references`）。学术 PDF 转出的
EPUB 是单文件（整篇一个 `ch001.xhtml`），拆开后书目才有独立 spine id，翻译器就能用
`exclude_spine_ids` 把它排除在翻译外、又保留在成品里（`translate_book.yaml` 默认已排除
`references`）。带体量自检：万一误判标题会保留原样、不动正文。标题识别 References /
Bibliography / 参考文献（忽略大小写与前导编号）。旧的 `--strip-references` 作为别名保留。

Requirements: `pandoc` on PATH (`sudo apt install pandoc`) and an NVIDIA GPU.
PaddleOCR-VL's models (~2 GB) download once into `~/.paddlex/official_models`;
DeepSeek-OCR (~6.3 GB) downloads once into `models/`. Both are reused.

What the pipeline does, and the gotchas it handles automatically:

1. **OCR -> Markdown**, PaddleOCR-VL by default. Prose, matrices and inline
   math come out well on both engines; on DeepSeek-OCR **code blocks are the
   weak spot** — structure and identifiers get mangled, so hand-check any code
   after conversion.
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

DeepSeek-OCR resolution tiers (`--ocr-size`, ignored by the default engine):
`tiny`/`small`/`base`/`large` send the whole page as one image
(512/640/1024/1280 px); `gundam` crops dense pages into tiles. **`base`** is the
default — best quality/VRAM trade-off for normal books (peak ~8.6 GB on a 12 GB
card). Use `gundam` for dense, small-font or scanned pages; it recognises more
but costs more VRAM and time.

### OCR engines: `--engine paddle` (default) and `--engine deepseek`

`--engine paddle` runs **PaddleOCR-VL 1.6** (released 2026-05-28, ~1.0 B
parameters, BF16) against a vLLM inference server you start yourself. It is the
default because on the same pages it is 6-8x faster than DeepSeek-OCR and its
markdown is at least as good.

Start the server first and leave it running; it holds **~7.1 GB of VRAM** for
as long as it is up, so nothing else should be on the card:

```bash
VIRTUAL_ENV=.venv-paddle .venv-paddle/bin/paddleocr genai_server \
    --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --port 8118 \
    --backend_config vllm_12gb.yaml
```

`pdf_to_epub.py` checks `/health` before it starts and exits with this command
if nothing answers, rather than falling back to in-process recognition — that
path took over 300 s for a single page here (GPU idle, one CPU core pinned),
slow enough to look like a hang. Point `--server-url` elsewhere to use a server
on another host or port, or pass `--engine deepseek` to work without one.

**Measured on an RTX 3060 12 GB**, same page sets, same machine, one engine at a
time:

| page set | PaddleOCR-VL 1.6 (vLLM) | DeepSeek-OCR (`base`) |
| --- | --- | --- |
| 4 scanned pages (`04-nii-blackboard`) | 15 s — 3.8 s/page, peak 10339 MiB | 97 s — 24.2 s/page, peak 8602 MiB |
| 3 two-column pages (`01-leases`) | 14 s — 4.7 s/page, peak 9937 MiB | 100 s — 33.3 s/page, peak 8602 MiB |
| 3 two-column pages with figures (`07-contract-net`) | 13 s — 4.3 s/page, peak 9941 MiB | 105 s — 35.0 s/page, peak 8606 MiB |
| 1 page, 3x3 matrix product (`airobotics_p34`) | 13 s, peak 9041 MiB | 42 s, peak 8602 MiB |

The PaddleOCR-VL peaks include the resident server, and every run pays ~3 s to
load the layout model into the client process. So the engine is faster but
hungrier: 10.3 GB of a 12 GB card at peak against DeepSeek-OCR's 8.6 GB.

Quality, both engines' markdown read against the PDFs:

* **Two-column reading order** — both follow the columns correctly. Paddle also
  rejoins words the typesetter hyphenated across a line or column break ("so-"
  + "lution"); DeepSeek leaves them broken and inserts a stray period
  (`at the most "op- . portune" time`, `problem- . solving`). Paddle's own slip
  is the mirror image: in `01-leases` it emitted the tail of one split word as
  its own block (`tum.`) and wrote "data." where the text said "datum."
* **Heading levels** — paddle is consistent: numbered sections at `##`,
  subsections at `###`, sub-subsections at `####`. DeepSeek gives sibling
  sections different depths in the same paper (`07-contract-net` has I. and
  III. at `###` but II. and IV. at `##`). Both duplicate one figure caption as
  a heading on the magazine-layout page.
* **LaTeX formulas** — both reproduce the 3x3 rotation-matrix product in full,
  and both reach the EPUB as 5 `<mtable>` / 15 `<mtr>` / 33 `<mtd>` with 0
  leftover `<annotation>` and MathML in the default namespace. Paddle emits
  plain `$…$` / `$$…$$` so step 2's de-escaping is skipped, and it kept the
  Figure 1-5 caption that DeepSeek dropped.
* **Scanned-page errors** — over a 200-word sample of `04-nii-blackboard`,
  paddle made 1 substitution ("opportunities" for "opportune"), about 0.5%.
  DeepSeek made 6: four hyphenations left broken, a footnote digit glued to the
  sentence ("1976.1"), and a stray apostrophe. About 3%.

Upstream's own numbers point the same way: 96.33% overall on OmniDocBench v1.6
at release, and OmniDocBench v1.5 puts PaddleOCR-VL-1.5 at 0.075 overall edit
distance against DeepSeek-OCR-2 at 0.100 (text 0.048, formula 0.198, table
0.096, reading order 0.057). The DeepSeek-OCR we use through `pdf_craft` is the
earlier v1.

Installing the engine (it **cannot** share this project's venv — `paddlex` pins
`pyyaml==6.0.2` and we need `pyyaml>=6.0.3`, so `uv` cannot resolve the two
together; `pdf_to_epub.py` drives it as a subprocess, overridable with
`PADDLE_PYTHON`):

```bash
uv venv --python 3.12 .venv-paddle
VIRTUAL_ENV=.venv-paddle uv pip install paddlepaddle-gpu==3.3.1 \
    --index https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --index-strategy unsafe-best-match
VIRTUAL_ENV=.venv-paddle uv pip install 'paddleocr[doc-parser]'

# the vLLM serving backend
VIRTUAL_ENV=.venv-paddle uv pip install einops 'torch==2.8.0' \
    'transformers<5.0.0' uvloop 'vllm==0.10.2' xformers
VIRTUAL_ENV=.venv-paddle uv pip install --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
```

The `cu130` index carries only nightly builds, so we pin the `cu126` stable
wheel; it runs fine against a CUDA 13.0 driver.

Three things about that backend install are not in the docs:

* `paddleocr install_genai_server_deps vllm` is the documented command and it
  shells out to `python -m pip`, which a `uv venv` does not have. Either
  `uv pip install pip` first or install the specs directly as above — they are
  exactly what `paddlex.utils.deps.get_genai_dep_specs("vllm-server")` returns.
* `flash-attn` is not optional even though vLLM 0.10.2 ships its own attention
  kernels: paddlex's `is_genai_engine_plugin_available` requires it on a CUDA
  box, and without it the `genai_server` subcommand does not even register (it
  is hidden from `paddleocr --help` either way, because argparse only lists
  subcommands declared with a help string). Install the prebuilt wheel matching
  torch 2.8 / cp312 / cxx11abiTRUE; building it from the sdist takes hours.
* paddlex starts vLLM with `max_num_batched_tokens=131072`, whose encoder cache
  alone leaves KV cache at **-0.36 GiB** on a 12 GB card and kills the engine
  with "No available memory for the cache blocks". `vllm_12gb.yaml` in this repo
  cuts it to 16384 and sets `gpu-memory-utilization: 0.62`, which leaves a
  280k-token KV cache and ~4 GB for the layout model in the client process.

Models (~2 GB) download to `~/.paddlex/official_models` on first use.

`paddle_ocr.py` is the subprocess entry point. It emits the same shape as
`pdf_craft.transform_markdown` — one markdown file plus an assets directory —
so steps 3 to 6 of the pipeline, `--split-references` included, are shared by
both engines. Step 3 resolves both syntaxes: pdf_craft writes `![](path)`,
PaddleOCR-VL writes `<img src="imgs/…">` inside a centring `<div>`.

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

Set `mask_code: false` in the YAML config to opt out.

### Tables are never translated

Third masker, same shape (`mask_table.py`, on by default). `table`, `tr`, `td`
and `th` are **not** in the translator's inline-tag set, so every single cell
counts as its own block — and in `append-block` mode a translated copy is
appended after every block. Inside a table that means a translated `<td>` after
each original `<td>`: a 3-column `<thead>` comes back with **6 cells**, the
layout collapses in the reader, and the columns that hold model names, token
counts and years are "translated" for nothing. Even where a cell does hold
prose, interleaving two languages cell by cell across a grid is unreadable.

So a table is kept entirely in the source language: every `<table>` element is
swapped for an EMPTY `<table data-tablemask="N"></table>` (text-free for the
same reason as the `<pre>` placeholder — a sentinel with text would be emitted
twice and duplicate the table), and the original markup is put back verbatim
afterwards. The **caption** normally sits outside the element
(`<h6>Table 3-1. …</h6>`), so it is translated like any other block and the
reader still gets a Chinese description of what the table shows.

Set `mask_table: false` in the YAML config to opt out.

### How the three maskers stack

Masking runs **math → code → table**; restoring runs the reverse,
**table → code → math**. The placeholder alphabets are mutually inert, so the
masking order is free, but whichever masker runs LAST captures the earlier
ones' placeholders inside its own mapping (a `<math>` inside a `<pre>` is
already a math sentinel when the `<pre>` is captured; a `<pre>` inside a table
cell is already a code placeholder when the `<table>` is captured), and those
placeholders only return to the document when that mapping is restored — so the
last masker must be the first restorer (LIFO).

The plumbing all three share — content-file discovery, EPUB repackaging and the
depth-counting element scanner — lives in `mask_common.py`. Round-tripping the
reference book through all three maskers without an LLM (844 `<pre>`/`<code>`
elements + 4 `<table>`s) reproduces every source XHTML document byte-identically.



## CJK typography pass (`polish_cjk.py`)

Retail EPUBs set `font-family` for **Latin** text — Manning's reasoning-model
book, for instance, has `#sbo-rt-content div{font-family:Verdana}` governing
every body paragraph. Verdana has no Chinese glyphs, so once the translation is
appended the reader falls back **per character** to whatever CJK font the device
happens to pick: the weight no longer matches (looks washed out), Han characters
fill the whole em box while the Latin letters around them do not (looks
oversized), and the mixed-script baseline wobbles. On top of that the source CSS
usually sets no `line-height` for body text, and the translator's segment
mapping copies the English spacing into the Chinese output, leaving stray
spaces at the start/end of translated blocks and inside inline tags
(`<em> 《书名》 </em>`).

`polish_cjk.py` is a standalone post-processing pass over a finished bilingual
EPUB that fixes all three:

1. **Tags the translated blocks.** Every block-level element whose *own-level*
   text is CJK-dominant (weighted `2·CJK / (2·CJK + latin_letters) ≥ 0.15`, at
   least one CJK character) gets `class="zh-translation"` and
   `lang`/`xml:lang="zh-CN"`. Descendant block elements are opaque when that
   text is measured, so container `<div>`s are never tagged but a translated
   paragraph that wraps an inline `<pre>` still is.
2. **Injects CSS.** A `<style id="zh-polish">` is appended to each document's
   `<head>` with a cross-platform Chinese font stack (Latin faces first so Latin
   words inside a Chinese paragraph keep Latin glyphs), `line-height`,
   `text-align: justify` with `text-justify: inter-ideograph`, and a monospace
   reset for `code`/`pre` inside translated blocks. It sets **no colour**, so
   the source book's `@media (prefers-color-scheme: dark)` rules keep governing
   the translation exactly as they govern the original. Re-running replaces the
   block in place instead of stacking a second copy.
3. **Cleans stray whitespace**, and only inside tagged blocks: runs at the start
   or end of a block, runs between two CJK characters, and runs next to CJK
   punctuation are deleted; a run with CJK on one side and Latin on the other
   collapses to a single space (`使用 pip` keeps its space).
4. **Embeds a Chinese font subset** (on by default). A font stack alone still
   leaves rendering up to whatever the device happens to have installed, so the
   pass subsets a CJK face down to the characters this book actually uses,
   writes it into the EPUB, registers it in the OPF manifest, adds an
   `@font-face` to the injected CSS, and splices the family into the **CJK
   position** of the stack — Latin faces stay ahead of it, so English words and
   digits inside a translated paragraph keep their Latin glyphs.

Nothing is re-serialised: the parser only produces source offsets, and the edits
are point replacements on the raw text, so untagged content is byte-identical by
construction. Repackaging rewrites the source zip entry by entry, preserving
order, timestamps and attributes, with `mimetype` first and stored — the same
invariant `pdf_to_epub._repackage_epub` maintains — which also makes the output
deterministic.

### The embedded font

The subset is built with fontTools at run time from
`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` **font number 2**
(Noto Sans CJK **SC** — index 0 is the JP face, whose glyph forms are wrong for
Chinese body text). Noto is OFL-licensed, so embedding and redistributing a
subset is fine. Output flavour is WOFF, not WOFF2: no brotli module is needed,
and EPUB3 readers support WOFF more widely than WOFF2 — not worth a dependency
to save ~30 KB.

Two coverage tiers:

| `--font-coverage` | characters | font bytes | EPUB delta |
|---|---|---|---|
| `book` (default) | only what the translation uses (1198 for the reasoning-model book) | 194 KiB | +218 KB |
| `gb2312` | that ∪ all of GB2312 (7445, generated from Python's `gb2312` codec, no hard-coded table) | 1.24 MiB | +1.30 MB |

Pick `gb2312` when the reader is likely to annotate inside the book: their own
notes then render in the same face instead of falling back.

The font file follows the book's own convention — it lands in whichever
directory already holds fonts (`OEBPS/Misc/` next to `JetBrains.woff2` here,
`OEBPS/fonts/`, or the OPF directory itself), falling back to `Misc/`. Because
the `<style>` is inlined per document rather than written to a shared CSS file,
the `@font-face` `src` is resolved **relative to each document**, which matters
in books whose documents sit at different depths (`OEBPS/` and `OEBPS/xhtml/`).
Idempotency is per-artefact: one font entry, one `@font-face` per document, one
manifest item, and re-running reproduces the file byte for byte. Two details
that determinism depends on — `recalcTimestamp=False` when loading the source
face (otherwise every save stamps the current time), and stripping
`<style>`/`<script>` before collecting the character set (the injected stack
itself contains `微软雅黑`, which would otherwise grow the subset on the second
run).

If the font source is missing or fontTools cannot load it, the pass **fails
loudly** with instructions (`apt install fonts-noto-cjk`, `--font-file`,
`--no-embed-font`) and writes nothing — degrading silently to "no font embedded"
would leave you believing the book carries one.

```bash
uv run python polish_cjk.py output/book.epub output/book.polished.epub
uv run python polish_cjk.py output/book.epub --stats     # count only, no write
uv run python polish_cjk.py in.epub out.epub --no-embed-font        # skip the font
uv run python polish_cjk.py in.epub out.epub --font-coverage gb2312
# knobs: --line-height --font-family --mono-family --text-align
#        --letter-spacing --threshold --min-cjk
#        --embed-font/--no-embed-font --font-coverage --font-file --font-number
```

Measured on `output/从零构建推理模型_中英对照.epub` (26 documents): 1946 blocks
tagged, 4114 whitespace fixes; CJK `<p>`s starting or ending with whitespace
went 1132/1160 → 0/0, spaces inside inline tags 180/192 → 0/0. Meanwhile the
source book's 473 `<pre>` and 1703 `<code>` elements are still present verbatim,
all 3489 `pre`/`code` fragments in the book are byte-identical before and after,
all 180 images and every other non-XHTML entry (including `toc.ncx` and the
26-entry spine) are unchanged, the 25 607 nodes outside translated blocks are
byte-identical, and running the tool on its own output reproduces the file
byte for byte. The embedded `book`-tier subset covers all 1198 CJK characters
in the book with zero misses, and its `@font-face` `src` resolves to the font
from all 26 documents.

The pass is not book-specific: it has also been run over
`The Presidents Book of Secrets` (1191 blocks, 2383 characters, font in
`OEBPS/Misc/`), `The Experience Machine` (764 blocks, 2158 characters, font in
the publisher's own `OEBPS/fonts/`, documents split across two directories, and
that book's two pre-existing `@font-face` rules left untouched),
`Hands-On大语言模型` (772 blocks, 1069 characters, font at the OPF root, all 55
`<pre>` and 2645 `<code>` byte-identical) and `具身智能` (378 blocks, 879
characters, `EPUB/` root instead of `OEBPS/`) — every book byte-identical
outside its XHTML and OPF, with the same coverage and idempotency results.
