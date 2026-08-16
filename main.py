import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from epub_translator import LLM, SubmitKind, language, translate

from claude_code_llm import DEFAULT_MODEL as DEFAULT_CLAUDE_CODE_MODEL
from claude_code_llm import ClaudeCodeLLM
from claude_llm import ClaudeLLM


SUBMIT_KINDS = {
    "append-block": SubmitKind.APPEND_BLOCK,
    "append-text": SubmitKind.APPEND_TEXT,
    "replace": SubmitKind.REPLACE,
}

LANGUAGES = {
    "zh": language.CHINESE,
    "en": language.ENGLISH,
    "ja": language.JAPANESE,
    "ko": language.KOREAN,
    "fr": language.FRENCH,
    "de": language.GERMAN,
    "es": language.SPANISH,
}

DEFAULT_USER_PROMPT = """
Translate the book into natural, accurate Simplified Chinese for readers of technical nonfiction and popular science.
Prioritize fluent Chinese expression over word-for-word literal translation while preserving every factual detail.
For established technical expressions in military, computing, semiconductor, engineering, and science contexts, translate the underlying concept with the conventional Chinese expression instead of translating the English words morpheme by morpheme.
Actively avoid stiff calques, machine-translation phrasing, and unnatural literal renderings when a normal Chinese technical or nonfiction expression is available.
Keep the author's argument structure, emphasis, chronology, and paragraph-level meaning intact.
Preserve names, organizations, places, product names, numbers, dates, and historical references accurately.
Do not summarize, omit, embellish, explain, or add translator notes.
""".strip()


def load_glossary(path: Path, max_terms: int = 300) -> str:
    """Read a glossary TSV (from glossary.py) into a prompt-ready term list.

    Emits the highest-frequency terms as `English → 中文` lines so the model
    renders names/terms consistently across the whole book.
    """
    rows: list[tuple[str, str, int]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en = (row.get("english") or "").strip()
            zh = (row.get("chinese") or "").strip()
            if not en or not zh:
                continue
            try:
                freq = int(row.get("freq") or 0)
            except ValueError:
                freq = 0
            rows.append((en, zh, freq))
    if not rows:
        return ""
    rows.sort(key=lambda r: (-r[2], r[0].lower()))
    lines = [f"- {en} → {zh}" for en, zh, _ in rows[:max_terms]]
    return (
        "Use this glossary for the following terms. Whenever a term on the left "
        "appears, render it with the exact Chinese on the right, consistently "
        "throughout. The glossary overrides your own preferred rendering for these "
        "terms (but never invent or insert a term that is not in the source text).\n"
        + "\n".join(lines)
    )


def compose_user_prompt(base_prompt: str, glossary_path: Path | None) -> str:
    if not glossary_path:
        return base_prompt
    block = load_glossary(glossary_path)
    if not block:
        return base_prompt
    return f"{base_prompt}\n\n{block}"


def drop_unsupported_proxy_env() -> None:
    """httpx does not handle socks:// proxy URLs unless optional SOCKS support is installed."""
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.getenv(name)
        if value and value.lower().startswith("socks://"):
            os.environ.pop(name, None)


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


# Spellings of EPUB_TRANSLATOR_PROVIDER that route through the local Claude Code CLI.
# Kept in sync with glossary.CLAUDE_CODE_PROVIDERS so both entry points agree.
CLAUDE_CODE_PROVIDERS = {"claude-code", "claude_code", "claudecode"}


def build_llm() -> LLM:
    provider = os.getenv("EPUB_TRANSLATOR_PROVIDER", "").strip().lower()

    if provider in CLAUDE_CODE_PROVIDERS:
        # Local headless Claude Code CLI. Deliberately reads no API key and no base URL:
        # the `claude` binary brings its own credentials, so a .env holding nothing but
        # EPUB_TRANSLATOR_PROVIDER=claude-code must be enough to start.
        return ClaudeCodeLLM(
            # `or` not a getenv default: an empty EPUB_TRANSLATOR_CLAUDE_CODE_MODEL= line
            # in .env would otherwise send `--model ""` -> API Error 400 on every request.
            model=os.getenv("EPUB_TRANSLATOR_CLAUDE_CODE_MODEL") or DEFAULT_CLAUDE_CODE_MODEL,
            token_encoding=os.getenv("EPUB_TRANSLATOR_TOKEN_ENCODING", "o200k_base"),
            timeout=env_float("EPUB_TRANSLATOR_CLAUDE_CODE_TIMEOUT", 600.0),
            retry_times=env_int("EPUB_TRANSLATOR_CLAUDE_CODE_RETRY_TIMES", 5),
            retry_interval_seconds=env_float("EPUB_TRANSLATOR_CLAUDE_CODE_RETRY_INTERVAL", 6.0),
            cache_path=os.getenv("EPUB_TRANSLATOR_CACHE_PATH", ".cache/translation"),
            max_concurrency=env_int("EPUB_TRANSLATOR_CLAUDE_CODE_MAX_CONCURRENCY", 0) or None,
            claude_bin=os.getenv("EPUB_TRANSLATOR_CLAUDE_CODE_BIN") or None,
        )

    model = env("EPUB_TRANSLATOR_MODEL")
    if provider == "anthropic" or model.startswith("claude-"):
        return ClaudeLLM(
            key=env("EPUB_TRANSLATOR_API_KEY"),
            url=env("EPUB_TRANSLATOR_BASE_URL"),
            model=model,
            token_encoding=os.getenv("EPUB_TRANSLATOR_TOKEN_ENCODING", "o200k_base"),
            timeout=float(os.getenv("EPUB_TRANSLATOR_TIMEOUT", "120")),
            retry_times=int(os.getenv("EPUB_TRANSLATOR_RETRY_TIMES", "5")),
            retry_interval_seconds=float(os.getenv("EPUB_TRANSLATOR_RETRY_INTERVAL", "6")),
            cache_path=os.getenv("EPUB_TRANSLATOR_CACHE_PATH", ".cache/translation"),
        )

    return LLM(
        key=env("EPUB_TRANSLATOR_API_KEY"),
        url=env("EPUB_TRANSLATOR_BASE_URL"),
        model=model,
        token_encoding=os.getenv("EPUB_TRANSLATOR_TOKEN_ENCODING", "o200k_base"),
        timeout=float(os.getenv("EPUB_TRANSLATOR_TIMEOUT", "120")),
        retry_times=int(os.getenv("EPUB_TRANSLATOR_RETRY_TIMES", "5")),
        retry_interval_seconds=float(os.getenv("EPUB_TRANSLATOR_RETRY_INTERVAL", "6")),
        cache_path=os.getenv("EPUB_TRANSLATOR_CACHE_PATH", ".cache/translation"),
        log_dir_path=os.getenv("EPUB_TRANSLATOR_LOG_DIR", ".cache/logs"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an EPUB into a bilingual EPUB through an OpenAI-compatible API."
    )
    parser.add_argument("source", help="Path to the source .epub file")
    parser.add_argument(
        "-o",
        "--output",
        help="Target .epub path. Defaults to output/<source-stem>.zh-bilingual.epub",
    )
    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGES),
        default=os.getenv("EPUB_TRANSLATOR_TARGET_LANGUAGE", "zh"),
        help="Target language code",
    )
    parser.add_argument(
        "--submit",
        choices=sorted(SUBMIT_KINDS),
        default=os.getenv("EPUB_TRANSLATOR_SUBMIT", "append-block"),
        help="append-block keeps original text and adds translated blocks",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("EPUB_TRANSLATOR_CONCURRENCY", "16")),
        help="Parallel translation requests. 16 is a good default; lower it only for rate-limited proxy APIs.",
    )
    parser.add_argument(
        "--max-group-tokens",
        type=int,
        default=int(os.getenv("EPUB_TRANSLATOR_MAX_GROUP_TOKENS", "2600")),
        help="Maximum tokens grouped into a single translation request.",
    )
    parser.add_argument(
        "--user-prompt",
        default=os.getenv("EPUB_TRANSLATOR_USER_PROMPT", DEFAULT_USER_PROMPT),
        help="Optional extra instruction for translation style.",
    )
    parser.add_argument(
        "--glossary",
        default=os.getenv("EPUB_TRANSLATOR_GLOSSARY", ""),
        help="Path to a glossary TSV (from glossary.py) to enforce consistent terms.",
    )
    return parser.parse_args()


def default_output_path(source: Path, target_language: str) -> Path:
    return Path("output") / f"{source.stem}.{target_language}-bilingual.epub"


def main() -> None:
    load_dotenv()
    drop_unsupported_proxy_env()
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source EPUB not found: {source}")
    if source.suffix.lower() != ".epub":
        raise ValueError(f"Source file must be an .epub file: {source}")

    target = Path(args.output) if args.output else default_output_path(source, args.language)
    target.parent.mkdir(parents=True, exist_ok=True)

    glossary_path = Path(args.glossary) if args.glossary else None
    if glossary_path and not glossary_path.exists():
        raise FileNotFoundError(f"Glossary not found: {glossary_path}")
    user_prompt = compose_user_prompt(args.user_prompt, glossary_path)

    print(f"source: {source}")
    print(f"target: {target}")
    print(f"mode: {args.submit}")
    print(f"language: {args.language}")
    print(f"concurrency: {args.concurrency}")
    if glossary_path:
        print(f"glossary: {glossary_path}")

    def on_progress(progress: float) -> None:
        print(f"progress: {progress:.1%}")

    translate(
        source_path=source,
        target_path=target,
        target_language=LANGUAGES[args.language],
        submit=SUBMIT_KINDS[args.submit],
        user_prompt=user_prompt,
        max_group_tokens=args.max_group_tokens,
        concurrency=args.concurrency,
        llm=build_llm(),
        on_progress=on_progress,
    )

    print(f"converted: {target}")


if __name__ == "__main__":
    main()
