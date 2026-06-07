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
