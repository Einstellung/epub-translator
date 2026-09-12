"""Convert a PDF into a clean, math-correct EPUB ready for translation.

OCR is PaddleOCR-VL 1.6 driven through a local vLLM inference server. The
script starts that server itself unless you point `--server-url` at one you
already run, and kills what it started before it exits.

Pipeline (every step here was validated against real pages, and every
workaround encodes a bug we actually hit):

  1. OCR -> Markdown + extracted image assets. PaddleOCR-VL 1.6 runs against
     the vLLM server at ~4 s per page, and it rejoins words hyphenated across
     line and column breaks.
  2. repair the OCR's markdown       -> three faults that cost whole pages:
     a display-math delimiter the engine never closes swallows the prose after
     it; `$ x $` is not math to pandoc, which then drops the LaTeX inside it;
     and an algorithm listing arrives as loose lines that markdown runs
     together into one paragraph. See _normalise_markdown.
  3. escape stray angle brackets   -> a BNF grammar printed as <message> =>
     <header> reaches pandoc as raw HTML and lands in the EPUB as an unclosed
     <header> element, i.e. invalid XHTML. A bare tag the document never closes
     becomes text; a void tag is self-closed, since raw <br> is not XHTML either.
  4. rewrite image paths to absolute -> the engine's relative asset paths do
     not line up with where the files actually land, so pandoc can't embed
     them. Absolute paths fix it.
  5. pandoc --mathml                 -> turns the LaTeX PaddleOCR-VL emits into
     MathML. pandoc's matrix handling is correct where the OCR engines' own
     MathML/SVG renderers dropped or flattened matrices.
  6. strip <annotation> elements     -> pandoc embeds a raw-LaTeX annotation
     beside each MathML formula. Readers that don't fully support <semantics>
     print that annotation as body text, so every formula shows up twice. We
     remove them.
  7. validate every document         -> the translator parses with xml.etree
     and dies on the first malformed file, mid-book. We parse each document
     here instead, and refuse to ship an EPUB that won't parse.
  8. repackage                       -> rebuild the EPUB zip with mimetype
     stored first and uncompressed, as the spec requires.

Usage:
    uv run python pdf_to_epub.py input/book.pdf
    uv run python pdf_to_epub.py input/paper.pdf --split-references
    uv run python pdf_to_epub.py input/book.pdf --server-url http://host:8118/v1
"""

import argparse
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

HERE = Path(__file__).parent
PADDLE_RUNNER = HERE / "paddle_ocr.py"
PADDLE_MODEL = "PaddleOCR-VL-1.6-0.9B"
PADDLE_SERVER_PORT = 8118
PADDLE_SERVER_URL = f"http://localhost:{PADDLE_SERVER_PORT}/v1"
VLLM_CONFIG = HERE / "vllm_12gb.yaml"

# How long to wait for the server to answer /health. Loading the weights and
# capturing CUDA graphs takes ~60 s on an RTX 3060; the ceiling is generous
# because a first run also downloads the model.
SERVER_READY_TIMEOUT = 600

# VRAM the OCR *client* needs beside the server: PaddleOCR-VL's layout detector
# plus paddle's allocator and the page bitmaps. Measured peak on this repo's
# test pages is ~2.6 GB (scanned pages are the worst case), so we hold back
# 3 GB. Do not lower this to the ~1.5 GB the layout weights suggest: leaving
# only 2.7 GB for the client is exactly the OOM we hit at
# gpu-memory-utilization 0.62 with a 1.3 GB desktop on the card.
CLIENT_RESERVE_MIB = 3072

# Below this the server has no room left for a KV cache worth having, so we
# stop rather than start something that will die during profiling.
MIN_GPU_UTILISATION = 0.35

# HTML element names, for telling markup apart from text that looks like it.
# Being on this list is not enough on its own: a BNF grammar happily prints
# <header> and <text>, which are elements. See _escape_stray_tags.
_HTML_ELEMENTS = frozenset(
    """a abbr address area article aside audio b base bdi bdo blockquote body br
    button canvas caption cite code col colgroup data datalist dd del details
    dfn dialog div dl dt em embed fieldset figcaption figure footer form h1 h2
    h3 h4 h5 h6 head header hgroup hr html i iframe img input ins kbd label
    legend li link main map mark menu meta meter nav noscript object ol optgroup
    option output p param picture pre progress q rp rt ruby s samp script search
    section select slot small source span strong style sub summary sup table
    tbody td template textarea tfoot th thead time title tr track u ul var video
    wbr
    math mi mn mo mrow ms mspace msqrt mroot mfrac msub msup msubsup munder
    mover munderover mmultiscripts mtable mtr mtd mtext merror mpadded mphantom
    mstyle menclose semantics annotation
    svg g path circle ellipse line polyline polygon rect text tspan defs use
    """.split()
)

# Elements that never have a closing tag, so a lone <br> is still markup.
_VOID_ELEMENTS = frozenset(
    """area base br col embed hr img input link meta param source track wbr
    """.split()
)


def _gpu_memory_mib() -> tuple[int, int]:
    """Return (free, total) VRAM in MiB for GPU 0, via nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(
            "error: nvidia-smi failed, so the server's VRAM budget cannot be "
            f"computed:\n{out.stderr.strip()}"
        )
    free, total = (int(v.strip()) for v in out.stdout.strip().split(",")[:2])
    return free, total


def _gpu_utilisation_budget() -> float:
    """Pick vLLM's gpu-memory-utilization from what is free right now.

    vLLM reads the flag as a fraction of *total* VRAM and refuses to start
    unless that much is free (see v1/worker/gpu_worker.py), so a value fixed in
    a config file is wrong as soon as the desktop's own usage moves — the 0.62
    we used to ship dies with a browser open. We leave CLIENT_RESERVE_MIB for
    the OCR client and round the rest down to a multiple of 0.05.
    """
    free, total = _gpu_memory_mib()
    usable = free - CLIENT_RESERVE_MIB
    budget = math.floor(usable / total * 20) / 20 if usable > 0 else 0.0
    if budget < MIN_GPU_UTILISATION:
        sys.exit(
            "error: not enough free VRAM to run the OCR server.\n"
            f"  free {free} MiB of {total} MiB, minus {CLIENT_RESERVE_MIB} MiB "
            f"reserved for the OCR client, leaves a budget of {budget:.2f} "
            f"(floor {MIN_GPU_UTILISATION:.2f}).\n"
            "Close what else is on the card, or point --server-url at a server "
            "on another machine."
        )
    return budget


def _server_healthy(server_url: str, timeout: float = 3.0) -> bool:
    """True if a PaddleOCR-VL genai server answers /health at that endpoint."""
    import urllib.error
    import urllib.request

    health = server_url.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _backend_config(work_dir: Path, utilisation: float) -> Path:
    """Write this run's vLLM backend config: the checked-in knobs from
    vllm_12gb.yaml plus the memory budget computed from the card's state."""
    base = VLLM_CONFIG.read_text(encoding="utf-8") if VLLM_CONFIG.is_file() else ""
    path = work_dir / "vllm_backend.yaml"
    path.write_text(
        f"{base.rstrip()}\ngpu-memory-utilization: {utilisation:.2f}\n",
        encoding="utf-8",
    )
    return path


class PaddleServer:
    """The vLLM inference server, started by us and killed by us.

    It runs in its own session so the whole tree (vLLM forks workers) can be
    signalled at once, and `stop` runs from a finally block and from the
    SIGINT/SIGTERM handlers — a server left behind holds most of the card.
    """

    def __init__(self, work_dir: Path, utilisation: float) -> None:
        self.log_path = work_dir / "vllm-server.log"
        self.utilisation = utilisation
        config = _backend_config(work_dir, utilisation)
        cmd = [
            "paddleocr", "genai_server",
            "--model_name", PADDLE_MODEL,
            "--backend", "vllm",
            "--port", str(PADDLE_SERVER_PORT),
            "--backend_config", str(config),
        ]
        if shutil.which("paddleocr") is None:
            cmd[:1] = [sys.executable, "-m", "paddleocr"]
        self.log = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        print(
            f"      started vLLM server pid {self.proc.pid}, "
            f"gpu-memory-utilization {utilisation:.2f}, log {self.log_path}",
            flush=True,
        )

    def wait_ready(self, server_url: str, timeout: int = SERVER_READY_TIMEOUT) -> None:
        started = time.time()
        while time.time() - started < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                sys.exit(
                    f"error: the vLLM server exited with code {code} before it "
                    f"was ready. Last lines of {self.log_path}:\n" + self._log_tail()
                )
            if _server_healthy(server_url):
                print(f"      server ready after {time.time() - started:.0f}s",
                      flush=True)
                return
            time.sleep(2)
        self.stop()
        sys.exit(
            f"error: the vLLM server did not answer {server_url} within "
            f"{timeout}s. Last lines of {self.log_path}:\n" + self._log_tail()
        )

    def _log_tail(self, lines: int = 25) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "  (log unreadable)"
        return "\n".join("  " + line for line in text.splitlines()[-lines:])

    def stop(self) -> None:
        """Kill the server's process group. Safe to call more than once."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        print(f"      stopping vLLM server pid {proc.pid}", flush=True)
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
            if proc.poll() is not None:
                break
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
        try:
            self.log.close()
        except OSError:
            pass


def _run_paddle(
    pdf_path: Path,
    md_path: Path,
    assets_path: Path,
    server_url: str,
) -> None:
    """OCR the PDF with PaddleOCR-VL, writing markdown + assets.

    A subprocess, not an import: paddle's GPU allocator goes away with it, and a
    hard crash in the OCR client comes back as an exit code we can report
    instead of taking the server down with it.
    """
    cmd = [
        sys.executable, str(PADDLE_RUNNER),
        str(pdf_path), str(md_path), str(assets_path),
        "--vl-backend", "vllm-server", "--server-url", server_url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"PaddleOCR-VL failed (exit {result.returncode})")


# --- markdown repair, between OCR and pandoc -------------------------------
#
# Each of the three faults below was measured on arXiv 2608.25512, a 92-page A4
# paper and the first long document this pipeline ran end to end.

_DISPLAY_OPEN = re.compile(r"(?<!\\)\\\[")
_DISPLAY_CLOSE = re.compile(r"(?<!\\)\\\]")

# One inline-math pair, padding optional. It has to match the tight pairs too:
# the pairs are consumed left to right, and a pattern that skipped `$x$` would
# then read the gap after it, `$ yields a pair $`, as the next formula.
_INLINE_MATH = re.compile(r"(?<![$\\])\$[ \t]*([^$\n]*?)[ \t]*\$(?!\$)")

# One line of pseudocode. Keywords are matched case-sensitively and lowercase:
# an English sentence starts with a capital, so `for` and `if` here cost almost
# nothing in false positives, while `For` and `If` would be expensive.
_PSEUDOCODE_STEP = re.compile(
    r"""^[ \t]*(?:
        \d{1,3}[.:)]?[ \t]                      # numbered step: "12 async ..."
      | \|                                      # the engine's indent marker
      | (?:async[ \t]+)?function\b | (?:sub)?procedure\b | method\b
      | (?:Input|Output|Require|Ensure|Data|Result)[ \t]*:
      | for\b | foreach\b | while\b | repeat\b | until\b | do\b
      | if\b | else\b | elif\b | then\b | switch\b | case\b
      | return\b | break\b | continue\b | yield\b
      | end\b | let\b | assert\b | await\b
    )""",
    re.VERBOSE,
)
# `Algorithm 7 Isolation realm reassignment`: a caption, not a sentence about one.
_ALGORITHM_CAPTION = re.compile(r"^Algorithm[ \t]+\d+[ \t]+[A-Z][^.]{0,60}$")
# A short line that is prose: a capitalised start and a full stop at the end.
_SENTENCE = re.compile(r"^[A-Z][^|]*\.$")
_PSEUDOCODE_MIN_LINES = 4
_PSEUDOCODE_MAX_WIDTH = 100
_PSEUDOCODE_MIN_RATIO = 0.8


def _markdown_blocks(text: str):
    """Yield (start, stop) line ranges of the paragraphs pandoc will see.

    A paragraph is a run of non-blank lines; fenced code, headings and blank
    lines are not paragraphs and are never yielded, so a caller may rewrite any
    range it gets without touching code or structure.
    """
    lines = text.split("\n")
    fence: str | None = None
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            i += 1
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            i += 1
            continue
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        j = i + 1
        while (j < len(lines) and lines[j].strip()
               and not lines[j].lstrip().startswith(("#", "```", "~~~"))):
            j += 1
        yield i, j
        i = j


def _close_display_math(text: str) -> tuple[str, int, int]:
    r"""Close a display-math delimiter the engine opened and never closed.

    pandoc's math parser runs from `\[` to the next `\]` wherever that is, so one
    dropped closer swallows everything up to the next formula: on 2608.25512 a
    single unterminated `\[` (the OCR of a commutative diagram, whose \begin{array}
    came out unbalanced) absorbed 6725 characters — prose, a figure and 54 inline
    formulas — into one unrenderable span. Closing it at the end of its own
    paragraph keeps the damage to the one formula. An odd number of `$$` in a
    paragraph is the same fault in the other delimiter.

    Paragraph-scoped, not line-scoped, because a display formula may legitimately
    run over several lines. Returns (text, `\]` added, `$$` added).
    """
    lines = text.split("\n")
    brackets = dollars = 0
    for start, stop in _markdown_blocks(text):
        block = "\n".join(lines[start:stop])
        missing = len(_DISPLAY_OPEN.findall(block)) - len(_DISPLAY_CLOSE.findall(block))
        suffix = ""
        if missing > 0:
            suffix += "\\]" * missing
            brackets += missing
        if block.count("$$") % 2:
            suffix += "$$"
            dollars += 1
        if suffix:
            lines[stop - 1] = lines[stop - 1].rstrip() + suffix
    return "\n".join(lines), brackets, dollars


def _tighten_inline_math(text: str) -> tuple[str, int]:
    r"""Rewrite `$ x $` as `$x$`, which is the only form pandoc reads as math.

    pandoc's tex_math_dollars requires no space after the opening `$` and none
    before the closing one. PaddleOCR-VL pads both (1218 pairs on 2608.25512),
    and the cost is not just unrendered math: `markdown+raw_tex` then reads the
    `\Gamma` inside as raw inline TeX and drops it from the HTML, so `$ \Gamma_n $`
    reaches the reader as `$ _n $`. Tightening the delimiters took this paper
    from 764 to 1974 `<math>` elements.

    Fenced code and `$$` display blocks are left alone. Returns (text, pairs).
    """
    fixed = 0

    def tighten(match: re.Match) -> str:
        nonlocal fixed
        inner = match.group(1).strip()
        tight = f"${inner}$"
        if not inner or match.group(0) == tight:
            return match.group(0)   # nothing inside, or already tight
        fixed += 1
        return tight

    out: list[str] = []
    fence: str | None = None
    display = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is not None:
            out.append(line)
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            out.append(line)
            continue
        # A line with an odd number of `$$` opens (or closes) a display block;
        # leave every line of one untouched, delimiters included.
        odd = line.count("$$") % 2 == 1
        if display or odd:
            out.append(line)
            display = display != odd
            continue
        out.append(_INLINE_MATH.sub(tighten, line))
    return "".join(out), fixed


def _is_pseudocode_line(line: str) -> bool:
    """A short line that reads as one step of an algorithm, not as prose."""
    if len(line) > _PSEUDOCODE_MAX_WIDTH:
        return False
    stripped = line.lstrip()
    if stripped.startswith(("#", "```", "~~~", "<", ">", "![", "*", "-")):
        return False
    if stripped.startswith("|") and line.rstrip().endswith("|"):
        return False  # a markdown table row
    return True


def _fence_captioned_algorithms(text: str) -> tuple[str, int]:
    """Fence the lines that follow an `Algorithm N Title` caption.

    The caption is the strong signal, so the lines after it need only be short
    and non-prose; blank lines between them are dropped, which rejoins a listing
    the engine split across a page break. Collection stops at the first long
    line, which is how the surrounding prose ends up outside the fence — the
    paper's own paragraphs are one line each and far longer than a step.
    """
    lines = text.split("\n")
    out: list[str] = []
    fence: str | None = None
    fenced = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if fence is not None:
            out.append(line)
            if stripped.startswith(fence):
                fence = None
            i += 1
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1
        if not _ALGORITHM_CAPTION.match(line):
            continue
        body: list[tuple[int, str]] = []
        j = i
        while j < len(lines):
            if not lines[j].strip():
                j += 1
                continue
            if not _is_pseudocode_line(lines[j]):
                break
            body.append((j, lines[j]))
            j += 1
        # Only a trailing line that is plainly a sentence is dropped: a step is
        # as likely to end in `fiber.parent.fiber` as in a keyword.
        while body and _SENTENCE.match(body[-1][1]):
            body.pop()
        steps = sum(1 for _, b in body if _PSEUDOCODE_STEP.match(b))
        if len(body) < 3 or steps * 2 < len(body):
            continue
        out.extend(["", "```", *(b for _, b in body), "```"])
        fenced += 1
        i = body[-1][0] + 1
    return "\n".join(out), fenced


def _fence_pseudocode_paragraphs(text: str) -> tuple[str, int]:
    """Fence a paragraph that is a listing whose caption the engine dropped.

    With no caption to lean on the test has to be narrow, because a false
    positive sets prose in monospace: at least four consecutive short lines, four
    fifths of which read as a step.
    """
    lines = text.split("\n")
    fenced = 0
    # Rewrite from the end so the earlier ranges stay valid as fences go in.
    for start, stop in reversed(list(_markdown_blocks(text))):
        block = lines[start:stop]
        if len(block) < _PSEUDOCODE_MIN_LINES:
            continue
        if not all(_is_pseudocode_line(line) for line in block):
            continue
        steps = sum(1 for line in block if _PSEUDOCODE_STEP.match(line))
        if steps < _PSEUDOCODE_MIN_LINES or steps < _PSEUDOCODE_MIN_RATIO * len(block):
            continue
        lines[start:stop] = ["```", *block, "```"]
        fenced += 1
    return "\n".join(lines), fenced


def _fence_pseudocode(text: str) -> tuple[str, int]:
    """Fence the algorithm listings the engine emits as loose lines.

    PaddleOCR-VL puts each line of an algorithm on its own markdown line with no
    fence, so markdown joins the lot into one run-on paragraph: an 18-line
    algorithm becomes a sentence, the line structure is lost, and `mask_code` in
    the translator has nothing to protect (it only sees marked-up code).

    Two passes, because half the listings on 2608.25512 kept their caption and
    half did not. Returns (text, blocks fenced).
    """
    text, captioned = _fence_captioned_algorithms(text)
    text, loose = _fence_pseudocode_paragraphs(text)
    return text, captioned + loose


def _normalise_markdown(text: str) -> str:
    """Repair the OCR's markdown so pandoc reads what the page said."""
    text, brackets, dollars = _close_display_math(text)
    print(f"      closed {brackets} unterminated \\[ and {dollars} unterminated $$",
          flush=True)
    text, pairs = _tighten_inline_math(text)
    print(f"      tightened {pairs} space-padded inline-math pair(s)", flush=True)
    text, blocks = _fence_pseudocode(text)
    print(f"      fenced {blocks} pseudocode block(s)", flush=True)
    return text


def _escape_stray_tags(text: str) -> tuple[str, int, int]:
    """Escape bare <word> sequences that are text, not markup, outside code.

    The Contract Net paper prints its message grammar as `<message> => <header>
    <addressee> <text> ...`. pandoc reads those as raw HTML, so the EPUB gets an
    unclosed <header> and stops being well-formed XHTML; the translator then
    fails on that file. PaddleOCR-VL escapes such names itself on some pages and
    not others, so we normalise.

    A name being an HTML element is not enough to keep it — half a BNF grammar
    is element names. A bare tag counts as markup only if it is a void element
    (a lone <br> is real) or if the document pairs it: an opening tag somewhere
    and a matching `</name>`. PaddleOCR-VL closes what it opens and writes its
    <div>/<img> with attributes, which this leaves alone; a non-terminal has no
    closing tag anywhere, so it becomes literal text.

    A void tag is kept but self-closed, because pandoc passes raw HTML through
    verbatim and `<br>` on its own is not well-formed XHTML either.

    Code is left alone (a backslash inside a code span would be printed), and so
    is anything the engine already escaped. Returns (text, escaped, self-closed).
    """
    opened = {m.group(1).lower()
              for m in re.finditer(r"<([A-Za-z][A-Za-z0-9._:-]*)(?=[\s/>])", text)}
    closed = {m.group(1).lower()
              for m in re.finditer(r"</([A-Za-z][A-Za-z0-9._:-]*)\s*>", text)}

    pattern = re.compile(r"(?<!\\)<(/?)([A-Za-z][A-Za-z0-9._:-]*)>")
    escaped = closed_up = 0

    def sub(m: re.Match) -> str:
        nonlocal escaped, closed_up
        slash, name = m.group(1), m.group(2)
        key = name.lower()
        if key in _HTML_ELEMENTS:
            if key in _VOID_ELEMENTS:
                if slash:
                    return m.group(0)
                closed_up += 1
                return f"<{name}/>"
            if key in opened and key in closed:
                return m.group(0)
        escaped += 1
        return f"\\<{slash}{name}\\>"

    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is not None:
            out.append(line)
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            out.append(line)
            continue
        # Split on inline code spans and rewrite only the parts outside them.
        parts = re.split(r"(`+[^`]*`+)", line)
        out.append("".join(
            part if i % 2 else pattern.sub(sub, part)
            for i, part in enumerate(parts)
        ))
    return "".join(out), escaped, closed_up


def _resolve_asset(rel: str, md_dir: Path) -> Path | None:
    """Find the file the OCR engine's relative image reference points at.

    PaddleOCR-VL writes imgs/<name> into the markdown while paddle_ocr.py saves
    the image flat under the assets directory, so the reference never resolves
    as written. The candidate list covers that and the older layouts.
    """
    name = Path(rel).name
    candidates = [
        md_dir / rel,
        md_dir / name,
        md_dir / "assets" / name,
        md_dir / md_dir.name / "assets" / name,
        md_dir / "_tmp" / "assets" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    hits = list(md_dir.rglob(name))  # last resort: search the work dir
    return hits[0].resolve() if hits else None


def _absolutize_images(text: str, md_dir: Path) -> str:
    """Rewrite image references to absolute paths, so pandoc can embed them.

    Covers both syntaxes the engine emits: ![](rel/path) for a plain figure, and
    an <img src="rel/path"> inside a centring <div> for a scaled one. An
    unresolvable reference is left alone and reported, never dropped.
    """

    def warn(rel: str) -> None:
        print(f"  warning: image not found, leaving as-is: {rel}", file=sys.stderr)

    def resolve_md(m: re.Match) -> str:
        rel = m.group(1)
        if rel.startswith(("http://", "https://", "/", "data:")):
            return m.group(0)
        found = _resolve_asset(rel, md_dir)
        if found is None:
            warn(rel)
            return m.group(0)
        return f"![]({found})"

    def resolve_html(m: re.Match) -> str:
        rel = m.group(2)
        if rel.startswith(("http://", "https://", "/", "data:")):
            return m.group(0)
        found = _resolve_asset(rel, md_dir)
        if found is None:
            warn(rel)
            return m.group(0)
        return f"{m.group(1)}{found}{m.group(3)}"

    text = re.sub(r"!\[\]\(([^)]*)\)", resolve_md, text)
    return re.sub(
        r'(<img\b[^>]*?\bsrc=")([^"]*)(")', resolve_html, text, flags=re.I
    )


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


def _validate_xml(epub_dir: Path) -> tuple[int, int]:
    """Refuse to ship an EPUB whose documents the translator cannot parse.

    translate_book.py parses with xml.etree, which is stricter than most EPUB
    readers and gives up on the first malformed document — as a run that died at
    0% after pandoc emitted a valueless attribute showed. So every XHTML, OPF
    and NCX is parsed here: with lxml first (the clearer error message), then
    with xml.etree, the parser that actually has to cope. A file only xml.etree
    rejects is rewritten from the lxml tree; one that neither can read aborts
    the conversion, naming the file.
    """
    checked = rewritten = 0
    for path in sorted(
        p for pattern in ("*.xhtml", "*.opf", "*.ncx")
        for p in epub_dir.rglob(pattern)
    ):
        checked += 1
        data = path.read_bytes()
        try:
            tree = etree.fromstring(data)
        except etree.XMLSyntaxError as err:
            sys.exit(
                f"error: {path.relative_to(epub_dir)} is not well-formed XML "
                f"and would break the translator:\n  {err}"
            )
        try:
            ElementTree.fromstring(data)
            continue
        except ElementTree.ParseError as err:
            print(f"      rewriting {path.relative_to(epub_dir)} ({err})", flush=True)
        path.write_bytes(
            etree.tostring(tree, encoding="utf-8", xml_declaration=True)
        )
        rewritten += 1
        try:
            ElementTree.fromstring(path.read_bytes())
        except ElementTree.ParseError as err:
            sys.exit(
                f"error: {path.relative_to(epub_dir)} still does not parse after "
                f"rewriting:\n  {err}"
            )
    return checked, rewritten


# Section titles that mark the bibliography. Matched case-insensitively after
# stripping any leading numbering ("VIII. References" / "8 References").
_REFERENCE_TITLES = {"references", "reference", "bibliography", "参考文献"}
_XHTML_NS = "http://www.w3.org/1999/xhtml"


def _tail_from_heading(ref_h, body) -> list:
    """Collect, in document order, the heading and everything after it.

    Walks ref_h -> body: the heading and its following siblings, then each
    ancestor's following siblings up to (excluding) body. This is exactly the
    tail the old strip path deleted; here we return it so it can be moved into a
    separate document instead of dropped.
    """
    tail = [ref_h, *ref_h.itersiblings()]
    node = ref_h.getparent()
    while node is not None and node is not body:
        tail.extend(node.itersiblings())
        node = node.getparent()
    return tail


def _new_references_doc(src_head, nodes: list) -> etree._ElementTree:
    """Build a standalone references.xhtml around the moved `nodes`.

    Minimal head: a title plus the source document's stylesheet <link>s (so the
    split-off page keeps the same styling). Appending `nodes` moves them out of
    the source tree, which is what the caller wants.
    """
    nsmap = {None: _XHTML_NS}
    root = etree.Element(f"{{{_XHTML_NS}}}html", nsmap=nsmap)
    head = etree.SubElement(root, f"{{{_XHTML_NS}}}head")
    title = etree.SubElement(head, f"{{{_XHTML_NS}}}title")
    title.text = "References"
    if src_head is not None:
        for link in src_head.findall(f"{{{_XHTML_NS}}}link"):
            head.append(deepcopy(link))
    doc_body = etree.SubElement(root, f"{{{_XHTML_NS}}}body")
    for node in nodes:
        doc_body.append(node)
    return etree.ElementTree(root)


def _register_in_opf(epub_dir: Path, ref_path: Path, idref: str) -> None:
    """Add references.xhtml to the OPF manifest and append it to the spine."""
    opf_path = next(iter(sorted(epub_dir.rglob("*.opf"))), None)
    if opf_path is None:
        print("  warning: no OPF found; references.xhtml not registered", file=sys.stderr)
        return
    tree = etree.parse(str(opf_path))
    root = tree.getroot()
    ns = etree.QName(root).namespace

    def q(tag: str) -> str:
        return f"{{{ns}}}{tag}" if ns else tag

    manifest = root.find(q("manifest"))
    spine = root.find(q("spine"))
    if manifest is None or spine is None:
        print("  warning: OPF has no manifest/spine; references.xhtml not registered", file=sys.stderr)
        return

    href = ref_path.relative_to(opf_path.parent).as_posix()
    item = etree.SubElement(manifest, q("item"))
    item.set("id", idref)
    item.set("href", href)
    item.set("media-type", "application/xhtml+xml")
    itemref = etree.SubElement(spine, q("itemref"))
    itemref.set("idref", idref)
    tree.write(str(opf_path), encoding="utf-8", xml_declaration=True)


def _split_references(epub_dir: Path) -> str:
    """Move the References/Bibliography section into its own spine document.

    pandoc emits the whole paper as one flat xhtml, so there is no separate
    spine item the translator's `exclude_spine_ids` can reach. To keep the
    bibliography in the final book but out of translation, we split it into a
    `references.xhtml` document and register it in the OPF; the translator then
    excludes it by idref (see translate_book.yaml) while it stays in the output.

    Strategy: find the References heading, then move it and everything after it
    in document order (walking up the ancestor chain, so the cut stays clean even
    when the whole body is wrapped in a single <section>, which is what pandoc
    produces) into the new document. A safety check leaves everything in place if
    the cut would gut the source document, so a mis-detected heading can't empty it.
    """
    for xhtml in sorted(epub_dir.rglob("*.xhtml")):
        tree = etree.parse(str(xhtml))
        root = tree.getroot()
        body = root.find(f"{{{_XHTML_NS}}}body")
        if body is None:
            continue

        ref_h = None
        for level in range(1, 7):
            for h in body.iter(f"{{{_XHTML_NS}}}h{level}"):
                text = "".join(h.itertext()).strip().lower()
                text = re.sub(r"^[ivxlcdm0-9]+[.\s]+", "", text)  # drop "viii. "/"8 "
                if text in _REFERENCE_TITLES:
                    ref_h = h
                    break
            if ref_h is not None:
                break
        if ref_h is None:
            continue

        tail = _tail_from_heading(ref_h, body)
        before = len(etree.tostring(body))
        tail_bytes = sum(len(etree.tostring(n)) for n in tail)
        after = before - tail_bytes
        if after < 2000 or after < before * 0.15:
            print(
                f"  warning: split-references would gut {xhtml.name} "
                f"({before} -> ~{after} bytes); leaving references in place.",
                file=sys.stderr,
            )
            return "over-cut guarded; references kept"

        # Move the tail into a new document (appending detaches it from source).
        src_head = root.find(f"{{{_XHTML_NS}}}head")
        ref_tree = _new_references_doc(src_head, tail)
        ref_path = xhtml.parent / "references.xhtml"
        ref_tree.write(str(ref_path), encoding="utf-8", xml_declaration=True)
        tree.write(str(xhtml), encoding="utf-8", xml_declaration=True)
        _register_in_opf(epub_dir, ref_path, "references")
        return f"split from {xhtml.name} into {ref_path.name} ({before} -> {after} bytes)"

    return "no References heading found; nothing split"


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


def _ocr(
    pdf_path: Path,
    md_path: Path,
    assets_path: Path,
    work_dir: Path,
    server_url: str | None,
) -> None:
    """Step 1: OCR the PDF, starting and stopping our own server if needed."""
    if server_url:
        if not _server_healthy(server_url):
            sys.exit(
                f"error: no PaddleOCR-VL inference server at {server_url}.\n"
                "Drop --server-url to have this script start one itself."
            )
        print(f"      using the server already running at {server_url}", flush=True)
        _run_paddle(pdf_path, md_path, assets_path, server_url)
        return

    if _server_healthy(PADDLE_SERVER_URL):
        sys.exit(
            f"error: something is already listening at {PADDLE_SERVER_URL}.\n"
            f"Pass --server-url {PADDLE_SERVER_URL} to use it, or stop it first."
        )
    server = PaddleServer(work_dir, _gpu_utilisation_budget())

    def bail(signum, _frame):  # noqa: ANN001 - signal handler signature
        server.stop()
        sys.exit(128 + signum)

    installed = [(sig, signal.signal(sig, bail))
                 for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)]
    try:
        server.wait_ready(PADDLE_SERVER_URL)
        _run_paddle(pdf_path, md_path, assets_path, PADDLE_SERVER_URL)
    finally:
        for sig, handler in installed:
            signal.signal(sig, handler)
        server.stop()


def convert(
    pdf_path: Path,
    epub_path: Path,
    server_url: str | None = None,
    work_dir: Path | None = None,
    keep_work: bool = False,
    title: str | None = None,
    split_references: bool = False,
) -> Path:
    """Run the full PDF -> clean EPUB pipeline. Returns the EPUB path.

    `server_url` names an inference server that is already running; without one
    we start a server for this conversion and stop it before returning.
    """
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

    # 1. OCR -> Markdown
    t0 = time.time()
    print(f"[1/8] OCR {pdf_path.name} -> markdown (PaddleOCR-VL) ...", flush=True)
    _ocr(pdf_path, md_path, assets_path, work_dir, server_url)
    print(f"      OCR done in {time.time() - t0:.1f}s", flush=True)

    # 2. repair the markdown  3. escape stray angle brackets  4. image paths
    text = md_path.read_text(encoding="utf-8")
    print("[2/8] repairing the OCR's markdown", flush=True)
    text = _normalise_markdown(text)
    text, escaped, closed_up = _escape_stray_tags(text)
    print(f"[3/8] escaped {escaped} <tag> sequence(s) that are text, "
          f"self-closed {closed_up} void tag(s)", flush=True)
    print("[4/8] resolving image paths", flush=True)
    text = _absolutize_images(text, md_path.parent)
    md_path.write_text(text, encoding="utf-8")

    # 4. pandoc -> EPUB with MathML
    print("[5/8] pandoc -> EPUB (MathML)", flush=True)
    raw_epub = work_dir / "raw.epub"
    cmd = [
        "pandoc", str(md_path), "-o", str(raw_epub),
        "--mathml", "-f", "markdown+tex_math_single_backslash",
        "--metadata", f"title={title or pdf_path.stem}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"pandoc failed:\n{result.stderr}")

    # 5. strip annotations  6. validate  7. repackage
    print("[6/8] stripping LaTeX annotation duplicates", flush=True)
    epub_extract = work_dir / "epub"
    if epub_extract.exists():
        shutil.rmtree(epub_extract)
    epub_extract.mkdir()
    with zipfile.ZipFile(raw_epub) as zf:
        zf.extractall(epub_extract)
    removed = _strip_annotations(epub_extract)
    print(f"      removed {removed} annotation(s)", flush=True)

    if split_references:
        print("[6b/8] splitting off References section", flush=True)
        print(f"      {_split_references(epub_extract)}", flush=True)

    print("[7/8] validating XML", flush=True)
    checked, rewritten = _validate_xml(epub_extract)
    print(f"      {checked} document(s) parse, {rewritten} rewritten", flush=True)

    print("[8/8] repackaging EPUB", flush=True)
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
        "--server-url", default=None,
        help="endpoint of a PaddleOCR-VL inference server that is already "
             f"running, e.g. {PADDLE_SERVER_URL}. Omit it and this script "
             f"starts a server on port {PADDLE_SERVER_PORT} for the conversion "
             "and stops it afterwards.",
    )
    p.add_argument(
        "--keep-work", action="store_true",
        help="keep the intermediate work directory for debugging",
    )
    p.add_argument(
        "--split-references", "--strip-references", dest="split_references",
        action="store_true",
        help="move the References/Bibliography section into its own spine "
             "document (references.xhtml) so the translator can exclude it from "
             "translation while it stays in the book; recommended for papers, "
             "safe no-op if none is found. --strip-references is a kept alias.",
    )
    args = p.parse_args(argv)

    if not args.pdf.is_file():
        sys.exit(f"error: PDF not found: {args.pdf}")
    epub_path = args.output or Path("output") / f"{args.pdf.stem}.epub"
    convert(
        pdf_path=args.pdf,
        epub_path=epub_path,
        server_url=args.server_url,
        keep_work=args.keep_work,
        split_references=args.split_references,
    )


if __name__ == "__main__":
    main()
