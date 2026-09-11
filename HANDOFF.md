# PaddleOCR-VL evaluation — where this stopped

## Done

* Research: PaddleOCR-VL **1.6**, released 2026-05-28, ~1.0 B params, BF16,
  ~2 GB VRAM for the weights. `paddleocr` 3.7.0 + `paddlepaddle-gpu` 3.3.1
  (cu126 stable; the cu130 index has nightlies only) installed into a separate
  `.venv-paddle`, because `paddlex` pins `pyyaml==6.0.2` against our
  `pyyaml>=6.0.3` — `uv add` fails to resolve, so the two trees cannot share a
  venv. Markdown output, LaTeX formulas, HTML tables, layout reading order.
  Upstream benchmarks and the install commands are written up in the README.
* `--engine {deepseek,paddle}` wired into `pdf_to_epub.py`; `paddle_ocr.py` is
  the subprocess entry point. Default stays **deepseek**.
* DeepSeek-OCR baseline measured (`--ocr-size base`, RTX 3060 12 GB), and the
  `--engine deepseek` path verified unchanged: `airobotics_p34.pdf` still
  produces 5 `<mtable>` / 15 `<mtr>` / 33 `<mtd>`, 0 leftover `<annotation>`,
  MathML default namespace intact.
* `lxml>=6.1.1` and `torchvision>=0.27.0` added to `pyproject.toml`.
  torchvision was the silent gap: a fresh `uv sync` gave
  `ImportError: ... requires ... torchvision` from inside DeepSeek-OCR's
  remote-code config, so the deepseek engine did not run at all until it was
  added. (It pulled torch 2.12 -> 2.14 in this worktree.)

## Not done — PaddleOCR-VL quality was never measured

Serverless PaddleOCR-VL is too slow to benchmark here. One page
(`airobotics_p34.pdf`) did not finish in 300 s; three two-column pages were
killed after 41 minutes. Throughout, the GPU sat at 0–19% with ~8.4 GB reserved
by Paddle's allocator while one CPU core stayed at 91%. Layout detection does
use the GPU, so it is the VL recognition step. Upstream "strongly recommends"
a serving backend instead of in-process inference.

## Next step to finish the evaluation

1. `VIRTUAL_ENV=.venv-paddle uv pip install 'paddlex[genai-vllm-server]'`
   (was still downloading when this session ended; it was killed, so re-run).
2. Start the server, then pass `vl_rec_backend="vllm-server"` and
   `vl_rec_server_url` through `paddle_ocr.py` (add flags for both).
3. Re-run the four page sets and fill in the README table's missing column:
   per-page seconds, peak VRAM, two-column order, heading levels, LaTeX
   matrices, scanned-page error rate over a 200-word sample.
4. Only if quality is at least equal should `DEFAULT_ENGINE` flip to `paddle`.

Test inputs used: first 4 pages of `04-nii-blackboard.pdf`, first 3 of
`01-leases.pdf`, first 3 of `07-contract-net.pdf` (all under
`~/Documents/legion-readings/originals/`), plus `pdf_test/airobotics_p34.pdf`.
