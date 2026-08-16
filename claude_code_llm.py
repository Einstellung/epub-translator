"""Use a locally installed, headless Claude Code CLI (`claude -p`) as the translation LLM.

This mirrors `claude_llm.py` (which talks HTTP to the Anthropic API), except every
request is a short-lived `claude -p` subprocess instead of an HTTP call. That means no
API key and no base URL: the CLI's own credentials (OAuth login, apiKeyHelper, Bedrock,
...) are used.

The CLI is a coding agent, and it cannot be degraded all the way back into a plain
translation model. `BASE_CLI_ARGS` strips the tools, the MCP servers, the skills, the
plugins, the hooks and every CLAUDE.md (see the measurements there), but the harness
still prepends an identity/environment block of its own, *in front of* whatever
`--system-prompt-file` supplies. Measured on v2.1.233 with a one-character system prompt
and a one-character user turn: 163 input tokens, i.e. ~160 tokens that are neither, and
asking the model to echo them back returns the product identity, the logged-in user's
e-mail address and today's date. So: much closer to a bare LLM than a coding agent, but
never exactly one, and never anonymous. Two consequences are handled here rather than
wished away:

  * the model retains agent reflexes, so a bare, very short user turn ("Preface",
    "Part II") is often answered with "I notice you haven't included the source text
    to translate…" instead of a translation. `_split_messages` therefore frames the
    user turn and `_TRANSLATE_HARDENING` is appended to the translation system prompt;
  * such a chat-style non-answer would otherwise be cached forever, so every response
    is validated before it is allowed out of `run_claude_cli` (see `_validate_translation`).
"""

import atexit
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from importlib.resources import files
from os import PathLike
from pathlib import Path

from epub_translator.llm.context import LLMContext
from epub_translator.llm.increasable import Increasable
from epub_translator.llm.types import Message, MessageRole
from epub_translator.template import create_env
from jinja2 import Environment, Template
from tiktoken import Encoding, get_encoding

DEFAULT_MODEL = "sonnet"

# Flags that turn the Claude Code agent back into a bare LLM. Measured on CLI v2.1.233
# with a 30-token prompt (input+cache_creation tokens reported by --output-format json):
#
#   (no flags)                                       ~21.5k tokens, reads global CLAUDE.md
#   --tools ""                                       ~12.0k tokens, reads global CLAUDE.md
#   --tools "" --strict-mcp-config                    ~1.3k tokens, reads global CLAUDE.md
#   --safe-mode --tools ""                             ~210 tokens, no CLAUDE.md
#
# `--safe-mode` is what actually keeps user/project CLAUDE.md, skills, plugins, hooks and
# MCP servers out of the context (auth, model selection and permissions keep working).
# The rest are belt-and-braces so a future change to safe-mode semantics cannot silently
# start feeding the agent's world back into the translation prompt.
#
# The ~210 tokens that remain are NOT the prompt: ~160 of them are the harness's own
# identity/environment preamble (product identity, today's date, the logged-in user's
# e-mail address), which no flag removes -- measured directly, a 1-char system prompt
# plus a 1-char user turn still reports 163 input tokens. Assume the model always knows
# it is Claude Code, and never send anything through here you would not send to it.
#
# NOTE: `--bare` looks like the obvious choice but must NOT be used: it refuses to read
# the OAuth credentials, so a normal `claude` login fails with "Not logged in".
BASE_CLI_ARGS: tuple[str, ...] = (
    "-p",
    "--output-format",
    "json",
    # No customizations: no CLAUDE.md, no skills, no plugins, no hooks, no MCP.
    "--safe-mode",
    # Empty string = disable every built-in tool.
    "--tools",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    # One assistant turn, no agent loop.
    "--max-turns",
    "1",
    # Do not litter ~/.claude/projects with one transcript per translated group.
    "--no-session-persistence",
    # Documented as ignored when --system-prompt(-file) is used; passed anyway because it
    # is free and would matter if the default system prompt ever came back.
    "--exclude-dynamic-system-prompt-sections",
)

# Env vars that leak in when the translator itself is started from inside a Claude Code
# session; they make the child believe it is a nested agent.
_DROPPED_ENV_NAMES = (
    "CLAUDECODE",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "AI_AGENT",
)
_DROPPED_ENV_PREFIXES = ("CLAUDE_CODE_",)
# ...except these, which are how a headless/CI or Bedrock/Vertex user authenticates.
# Dropping them (the prefix rule above used to) silently breaks `claude setup-token`
# logins and re-routes Bedrock/Vertex users to the first-party API.
_KEPT_CLAUDE_CODE_ENV = frozenset(
    {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
        "CLAUDE_CODE_CLIENT_CERT",
        "CLAUDE_CODE_CLIENT_KEY",
        "CLAUDE_CODE_CLIENT_KEY_PASSPHRASE",
    }
)

# Errors that will never fix themselves; retrying only wastes minutes.
# Mirrors the CLI's own non-retryable classifier (v2.1.233 ships the regex
# `not logged in|please run /login|authentication failed|invalid api key|
#  oauth token (expired|revoked)|credit balance (is )?too low|usage limit reached|...`).
_FATAL_MARKERS = (
    "not logged in",
    "invalid api key",
    "invalid x-api-key",
    "authentication failed",
    "oauth token expired",
    "oauth token revoked",
    "unknown option",
    "unknown command",
    "credit balance is too low",
    "credit balance too low",
    "please run /login",
    # `--model <typo>` -> api_error_status 404, identical on every attempt.
    "it may not exist or you may not have access to it",
)


class ClaudeCliError(RuntimeError):
    """A `claude -p` invocation failed after exhausting its retries."""


class ClaudeCliFatalError(ClaudeCliError):
    """A failure that cannot possibly recover; skip the retry loop entirely.

    String-sniffing (`_is_fatal`) covers the errors the CLI reports in its JSON body;
    this subclass covers the ones we raise ourselves and already know are permanent
    (a non-executable binary, a shutdown in progress, ...).
    """


class ClaudeCliShutdown(ClaudeCliFatalError):
    """Raised instead of starting/retrying a request once SIGINT/SIGTERM arrived."""


# ---------------------------------------------------------------------------
# Live subprocess registry: nothing may outlive this interpreter
# ---------------------------------------------------------------------------
# Every call runs `claude` in its own session (`start_new_session=True`) so a timeout can
# kill the whole node tree. The flip side is that Ctrl-C in the terminal reaches only
# *this* process: the children are in different process groups and never see SIGINT. With
# 16 concurrent requests that used to leave 16 orphaned node processes (~290 MB each)
# reparented to init, still burning quota, plus 16 leaked temp dirs -- and, because the
# worker threads are plain non-daemon ThreadPoolExecutor threads that the interpreter
# joins on exit, the shell did not even come back until every request had timed out.
#
# So: keep a registry of live (process, workdir) pairs, and on SIGINT/SIGTERM/atexit kill
# every process group and remove every temp dir. `_SHUTTING_DOWN` additionally makes the
# in-flight worker threads bail out instead of retrying, so the join is instant.
# Reentrant on purpose: the signal handler runs *in the main thread*, so it can fire
# while that same thread is already inside _register_process/_unregister_process. A plain
# Lock would deadlock the interpreter on Ctrl-C.
_LIVE_LOCK = threading.RLock()
_LIVE_PROCESSES: dict[int, tuple[subprocess.Popen, str]] = {}
_SHUTTING_DOWN = threading.Event()
_PREVIOUS_HANDLERS: dict[int, object] = {}
_HANDLERS_INSTALLED = False


def _register_process(process: subprocess.Popen, workdir: str) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCESSES[process.pid] = (process, workdir)


def _unregister_process(process: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCESSES.pop(process.pid, None)


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except Exception:
            pass


def cleanup_live_processes() -> int:
    """Kill every live `claude` process group and delete its temp dir. Returns the count."""
    with _LIVE_LOCK:
        entries = list(_LIVE_PROCESSES.values())
        _LIVE_PROCESSES.clear()
    for process, workdir in entries:
        _kill_process_group(process)
        shutil.rmtree(workdir, ignore_errors=True)
    return len(entries)


def _signal_handler(signum: int, frame) -> None:
    _SHUTTING_DOWN.set()
    cleanup_live_processes()
    previous = _PREVIOUS_HANDLERS.get(signum)
    if callable(previous):
        # The default SIGINT handler is signal.default_int_handler, which raises
        # KeyboardInterrupt -- exactly the behaviour the rest of the program expects.
        previous(signum, frame)
        return
    if previous == signal.SIG_IGN:
        return
    # Previous handler was SIG_DFL: restore it and re-raise so the exit status is the
    # normal 128+signum instead of a silent 0.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def install_signal_handlers() -> None:
    """Idempotent. MUST be reached from the main thread at least once.

    `signal.signal` is illegal off the main thread, and the first `run_claude_cli` of a
    run usually happens *on a worker thread* (both the translation pool and the glossary
    pool). So a worker-thread call must leave the flag clear, or it permanently poisons
    the one main-thread call that could have installed the handlers -- which is exactly
    how the first version of this fix silently did nothing at all.
    """
    global _HANDLERS_INSTALLED
    with _LIVE_LOCK:
        if _HANDLERS_INSTALLED:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        installed = False
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                _PREVIOUS_HANDLERS[signum] = signal.signal(signum, _signal_handler)
                installed = True
            except (ValueError, OSError):  # unsupported platform
                _PREVIOUS_HANDLERS.pop(signum, None)
        _HANDLERS_INSTALLED = installed


# atexit runs *after* threading shutdown has already joined the pool threads, so it can
# never be the primary mechanism -- it is the last sweep for anything the handlers missed
# (e.g. an uncaught exception unwinding the main thread).
atexit.register(cleanup_live_processes)


# ---------------------------------------------------------------------------
# Global concurrency ceiling
# ---------------------------------------------------------------------------
# Each request is a whole node process (~290 MB resident); 16 at once is ~4.6 GB. The cap
# lives at module level, not on the executor, because `glossary.py` calls `run_claude_cli`
# directly -- with an executor-only semaphore, EPUB_TRANSLATOR_CLAUDE_CODE_MAX_CONCURRENCY
# had no effect at all on the glossary stage.
_LIMIT_LOCK = threading.Lock()
_LIMIT_VALUE: int | None = None
_LIMIT_SEMAPHORE: threading.BoundedSemaphore | None = None
_LIMIT_CONFIGURED = False


def set_max_concurrency(limit: int | None) -> None:
    """Cap simultaneous `claude` processes process-wide. None/<=0 means unlimited.

    Called once at start-up (from `ClaudeCodeExecutor`, or lazily from the env var).
    Changing the limit while requests are in flight is not supported.
    """
    global _LIMIT_VALUE, _LIMIT_SEMAPHORE, _LIMIT_CONFIGURED
    with _LIMIT_LOCK:
        _LIMIT_CONFIGURED = True
        if not limit or limit <= 0:
            _LIMIT_VALUE, _LIMIT_SEMAPHORE = None, None
            return
        if _LIMIT_VALUE == limit and _LIMIT_SEMAPHORE is not None:
            return
        _LIMIT_VALUE = limit
        _LIMIT_SEMAPHORE = threading.BoundedSemaphore(limit)


def _concurrency_gate():
    global _LIMIT_CONFIGURED
    with _LIMIT_LOCK:
        configured, semaphore = _LIMIT_CONFIGURED, _LIMIT_SEMAPHORE
    if not configured:
        raw = (os.getenv("EPUB_TRANSLATOR_CLAUDE_CODE_MAX_CONCURRENCY") or "").strip()
        try:
            set_max_concurrency(int(raw) if raw else None)
        except ValueError:
            set_max_concurrency(None)
        with _LIMIT_LOCK:
            semaphore = _LIMIT_SEMAPHORE
    return semaphore if semaphore is not None else contextlib.nullcontext()


def _claude_binary() -> str:
    return os.getenv("EPUB_TRANSLATOR_CLAUDE_CODE_BIN") or "claude"


def _child_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _DROPPED_ENV_NAMES
        and (key in _KEPT_CLAUDE_CODE_ENV or not key.startswith(_DROPPED_ENV_PREFIXES))
    }
    env["NO_COLOR"] = "1"
    # An auto-update in the middle of a book-long run would be a fun way to lose an hour.
    env.setdefault("DISABLE_AUTOUPDATER", "1")
    env.setdefault("DISABLE_NON_ESSENTIAL_MODEL_CALLS", "1")
    env.setdefault("DISABLE_ERROR_REPORTING", "1")
    # Deliberately NOT set: DISABLE_TELEMETRY=1 measurably *inflates* the prompt
    # (193 -> 558 input tokens for the same request, reproduced 2/2 on v2.1.233).
    return env


def _is_fatal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FATAL_MARKERS)


def _summarize(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _invoke_once(
    system: str | None,
    user: str,
    *,
    model: str,
    timeout: float,
    claude_bin: str,
    extra_args: Sequence[str],
) -> str:
    """One `claude -p` run. Raises ClaudeCliError on any failure. Not retried here."""
    if _SHUTTING_DOWN.is_set():
        raise ClaudeCliShutdown("shutting down; not starting a new claude CLI process")

    # A private empty cwd per call. Without it the CLI would suck the project's
    # CLAUDE.md and directory listing into every single translation request.
    workdir = tempfile.mkdtemp(prefix="epub-claude-code-")
    process: subprocess.Popen | None = None
    try:
        args: list[str] = [claude_bin, *BASE_CLI_ARGS, "--model", model]
        if system:
            # Via a file, never argv: rendered fill.jinja/translate.jinja system prompts
            # run to several KB and argv has both a size limit and quoting traps.
            system_file = Path(workdir) / "system-prompt.txt"
            system_file.write_text(system, encoding="utf-8")
            args += ["--system-prompt-file", str(system_file)]
        args += list(extra_args)

        try:
            process = subprocess.Popen(  # noqa: S603
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                env=_child_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                # Own process group so a timeout can take the node child tree with it.
                start_new_session=True,
            )
        except FileNotFoundError:
            raise  # handled (and made fatal) by run_claude_cli
        except (PermissionError, NotADirectoryError, IsADirectoryError) as err:
            # Permanently broken binary/path: five retries would just waste ~3 minutes.
            raise ClaudeCliFatalError(
                f"cannot execute claude CLI {claude_bin!r}: {err}"
            ) from err
        except OSError as err:
            # ENOMEM is the realistic one here (16 node processes ~= 4.6 GB), and it is
            # transient, so it must land in the backoff loop rather than escape raw and
            # kill an hour-long run from inside a worker thread.
            raise ClaudeCliError(
                f"failed to start claude CLI {claude_bin!r}: {type(err).__name__}: {err}"
            ) from err

        _register_process(process, workdir)
        try:
            stdout, stderr = process.communicate(input=user, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except Exception:
                stdout, stderr = "", ""
            raise ClaudeCliError(
                f"claude CLI timed out after {timeout}s; stderr: {_summarize(stderr)}"
            ) from None

        if _SHUTTING_DOWN.is_set():
            # The process was killed by the signal handler, not by the model finishing.
            raise ClaudeCliShutdown("claude CLI was terminated by shutdown")

        # Parse first, THEN look at the exit code: the CLI reports "Not logged in",
        # "credit balance too low", a bad --model and every other API error as
        # exit-code 1 *plus* a JSON body whose "result" field carries the only
        # human-readable reason. That field sits ~700 bytes into the payload, past
        # the 600-char excerpt, so reporting raw stdout both hides the reason from
        # the user and hides the fatal marker from _is_fatal() -> 6 pointless
        # retries (~3 minutes) per request for a permanently broken setup.
        payload: dict | None = None
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None

        if process.returncode != 0:
            if payload is not None:
                raise ClaudeCliError(
                    f"claude CLI exited with {process.returncode} "
                    f"(subtype={payload.get('subtype')}, "
                    f"api_error_status={payload.get('api_error_status')}): "
                    f"{_summarize(str(payload.get('result')))}; "
                    f"stderr: {_summarize(stderr)}"
                )
            raise ClaudeCliError(
                f"claude CLI exited with {process.returncode}; "
                f"stderr: {_summarize(stderr)}; stdout: {_summarize(stdout)}"
            )

        if payload is None:
            raise ClaudeCliError(
                f"claude CLI returned non-JSON output: {_summarize(stdout)}; "
                f"stderr: {_summarize(stderr)}"
            )

        result = payload.get("result")
        if payload.get("is_error") or payload.get("api_error_status"):
            raise ClaudeCliError(
                f"claude CLI reported an error (subtype={payload.get('subtype')}, "
                f"api_error_status={payload.get('api_error_status')}): {_summarize(str(result))}"
            )
        if not isinstance(result, str) or not result.strip():
            raise ClaudeCliError(
                f"claude CLI returned an empty result; stderr: {_summarize(stderr)}"
            )
        return result
    finally:
        if process is not None:
            _unregister_process(process)
            # Never delete the cwd out from under a live process, and never let one
            # survive this function -- including the narrow window where a signal
            # arrived before the process made it into the registry.
            if process.poll() is None:
                _kill_process_group(process)
        shutil.rmtree(workdir, ignore_errors=True)


def run_claude_cli(
    system: str | None,
    user: str,
    *,
    model: str,
    timeout: float,
    retry_times: int = 5,
    retry_interval_seconds: float = 6.0,
    claude_bin: str | None = None,
    extra_args: Sequence[str] | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Run one headless `claude -p` turn and return the assistant's plain text.

    The prompt goes in over stdin and the system prompt over a temp file, so neither is
    bounded by argv limits or shell quoting. Every call gets its own empty temporary cwd,
    which is what keeps the project's CLAUDE.md / directory contents out of the context,
    and the CLI is stripped down as far as it goes (no tools, no MCP, no skills, one
    turn) -- see the module docstring for what cannot be stripped.

    Thread-safe: no shared mutable state, one subprocess and one temp dir per call.
    Simultaneous processes are capped by `set_max_concurrency` /
    EPUB_TRANSLATOR_CLAUDE_CODE_MAX_CONCURRENCY.

    `validate` gets the assistant's text and returns None to accept it or a reason to
    reject it. A rejected response is retried like any other transient failure, which is
    what keeps a chat-style non-answer ("I notice you haven't included the source
    text…") out of the permanent on-disk cache -- LLMContext commits whatever comes back
    here, and a committed answer is never revisited on a later run.

    Sampling parameters are deliberately unsupported: the CLI exposes no equivalent of
    `temperature`, `top_p` or `max_tokens`, so callers that pass them are ignored rather
    than silently lied to.

    Raises `ClaudeCliError` once `retry_times` retries are exhausted, or immediately for
    errors that cannot recover (not logged in, bad flag, ...). The message carries a
    trimmed stderr/stdout excerpt.
    """
    install_signal_handlers()
    binary = claude_bin or _claude_binary()
    args = tuple(extra_args or ())
    last_error: Exception | None = None

    for attempt in range(retry_times + 1):
        try:
            with _concurrency_gate():
                result = _invoke_once(
                    system=system,
                    user=user,
                    model=model,
                    timeout=timeout,
                    claude_bin=binary,
                    extra_args=args,
                )
            reason = validate(result) if validate is not None else None
            if reason is None:
                return result
            raise ClaudeCliError(f"rejected claude CLI response: {reason}")
        except FileNotFoundError as err:
            raise ClaudeCliFatalError(
                f"claude CLI not found: {binary!r}. Install Claude Code or set "
                f"EPUB_TRANSLATOR_CLAUDE_CODE_BIN to its absolute path."
            ) from err
        except ClaudeCliFatalError:
            raise
        except ClaudeCliError as err:
            last_error = err
            if _is_fatal(str(err)):
                raise
            if attempt >= retry_times:
                break
            if retry_interval_seconds > 0:
                # Capped exponential backoff: usage limits need more than a flat 6s.
                # Interruptible, so Ctrl-C during a backoff does not stall the exit.
                if _SHUTTING_DOWN.wait(min(retry_interval_seconds * (2**attempt), 120.0)):
                    raise ClaudeCliShutdown("shutting down during retry backoff") from err

    raise ClaudeCliError(
        f"claude CLI failed after {retry_times + 1} attempts: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Prompt hardening + response validation
# ---------------------------------------------------------------------------
# Appended to the rendered translate.jinja system prompt only. fill.jinja is left byte
# for byte alone: its user turn is long and highly structured and has never produced a
# chat-style non-answer, and the library already re-prompts on a malformed fill.
#
# Measured on sonnet via `claude -p`, six short real inputs from the target book
# (`Copyright © 2022. All rights reserved.` / `Part II` / `1.` / `MATHPLACEHOLDER0001X`
# / `Preface` / `A Path Towards Autonomous Machine Intelligence`):
#
#   bare user turn, no hardening              12/18 answered "you haven't included the
#                                             source text" instead of translating
#   hardening only, bare user turn             6/12 still did
#   user-turn label only, no hardening         2/6  still did
#   hardening + labelled user turn             0/48 did; every one produced a real
#                                             translation (or the verbatim placeholder)
#
# Both halves are load-bearing; neither alone is enough.
_TRANSLATE_HARDENING = """

--- Runtime note (headless single-turn CLI) ---
The user turn below contains the ENTIRE input for this task. Nothing else will follow,
no files or tools exist, and there is no one to ask. The input is often very short — a
lone heading, one word, a number, a date, or an opaque placeholder token — and that is
still a complete, valid input.
Always produce the output the instructions above call for, for exactly the input you
were given. Never echo the input back, never ask for more text, never apologise, never
explain, never add commentary."""

# The "nothing else" half is not decoration. With the bare label, one-word headings
# ("Preface", "Part II", "Introduction") came back untranslated or with the source
# echoed above the translation in 2/12 runs, and once as a Chinese-language
# "please provide the source text" -- a meta-answer `_validate_translation` cannot see,
# because it does contain Chinese. With this label: 12/12 clean.
_TRANSLATE_USER_PREFIX = (
    "Source text to translate (reply with the translation only, nothing else):\n"
)

_KIND_TRANSLATE = "translate"
_KIND_FILL = "fill"

# Target languages whose output must contain a script the (Latin/English) source cannot
# supply. Latin-script targets (French, German, ...) are deliberately absent: no cheap
# character test distinguishes "translated into German" from "handed back untouched",
# and a rule that cannot tell them apart would reject valid work forever.
_ZH = "一-鿿㐀-䶿豈-﫿"
_REQUIRED_SCRIPT: dict[str, tuple[re.Pattern[str], str]] = {
    "simplified chinese": (re.compile(f"[{_ZH}]"), "Chinese characters"),
    "traditional chinese": (re.compile(f"[{_ZH}]"), "Chinese characters"),
    "japanese": (re.compile(f"[{_ZH}぀-ゟ゠-ヿ]"), "Japanese characters"),
    "korean": (re.compile("[가-힯ᄀ-ᇿ]"), "Hangul"),
    "russian": (re.compile("[Ѐ-ӿ]"), "Cyrillic"),
    "arabic": (re.compile("[؀-ۿ]"), "Arabic script"),
    "hindi": (re.compile("[ऀ-ॿ]"), "Devanagari"),
    "thai": (re.compile("[฀-๿]"), "Thai script"),
}

_ALNUM_RE = re.compile(r"[^0-9a-z一-鿿]+")


def _fingerprint(text: str) -> str:
    """Lower-case, alphanumerics only: 'Part II.' and 'part ii' collapse to 'partii'."""
    return _ALNUM_RE.sub("", text.lower())


def _validate_translation(source: str, response: str, target_language: str) -> str | None:
    """Reject responses that are certainly not a translation of `source`.

    Only failure evidence that is cheap AND unambiguous is used, because a false
    rejection is far worse than a miss: the group would be retried five times and then
    abort the run, forever, for content that was fine. The rule is therefore

        reject  <=>  the target language needs a script the source cannot contain
                     AND the response contains not one character of that script
                     AND the response is not simply the source passed through

    with "passed through" defined generously. The third clause is what protects the
    legitimately script-free groups; every one of these is real, from this repo:

      * `MATHPLACEHOLDER0001X` (mask_math.py sentinels) -- there is a cached
        translation whose entire content is that token and nothing else;
      * a group that is only code, an identifier, a URL, an ISBN or a bare number;
      * bibliography lines a translator is expected to leave in the original.

    In all of those the response is the source (modulo case/whitespace/punctuation), so
    the fingerprint comparison accepts it. A chat-style non-answer or a refusal shares
    almost nothing with the source, and when it does quote it back ("It looks like you
    referenced item \"1.\" but no source text…") it is many times longer, which the
    length guard catches.

    Known and accepted miss: a response that is the source verbatim because the model
    declined to translate it ("Preface" -> "Preface"). Indistinguishable from a
    legitimate pass-through by any cheap rule, and merely leaves one heading in English.
    """
    required = _REQUIRED_SCRIPT.get(target_language.strip().lower())
    if required is None:
        return None
    pattern, script_name = required
    if pattern.search(response):
        return None

    source_fp = _fingerprint(source)
    if not source_fp:
        # Nothing translatable in the source (pure punctuation/symbols/whitespace).
        return None

    response_fp = _fingerprint(response)
    if response_fp == source_fp:
        return None
    if source_fp and response_fp and (source_fp in response_fp or response_fp in source_fp):
        # Tolerate a dropped/added trailing period or a stray quote, but not a short
        # source quoted inside a long apology.
        if len(response_fp) <= len(source_fp) * 2 + 20:
            return None

    return (
        f"no {script_name} in a {target_language} translation, and the response is not "
        f"the source passed through (source={_summarize(source, 120)!r}, "
        f"response={_summarize(response, 200)!r})"
    )


def _validate_fill(response: str) -> str | None:
    """The fill turn must return an <xml> block; anything else is a non-answer.

    Strictly weaker than the library's own check (which needs a complete, parseable
    `<xml>...</xml>`), so it can never reject something the library would have accepted
    -- it just fails it here, where a cheap CLI-level retry fixes it, instead of caching
    the non-answer and burning a fill retry on it.
    """
    if "<xml" in response.lower():
        return None
    return f"no <xml> block in the fill response: {_summarize(response, 200)!r}"


class _RegisteredTemplate:
    """Wraps a jinja Template so we know which prompt produced which system message.

    The library asks for templates by name (`llm.template("translate")`) and then hands
    the rendered text back to us as an anonymous SYSTEM message. Recording the rendered
    string here -- together with the `target_language` it was rendered with -- is what
    lets the executor pick the right user-turn framing and the right response validation
    without sniffing the prompt's wording, which would silently stop working the day the
    library rewords its templates.
    """

    def __init__(self, template: Template, kind: str, registry: dict[str, tuple[str, str]],
                 lock: "threading.Lock") -> None:
        self._template = template
        self._kind = kind
        self._registry = registry
        self._lock = lock

    def render(self, *args, **kwargs) -> str:
        # The rendered text is returned UNCHANGED. It goes straight into the library's
        # Message list, which is what LLMContext sha512s for the on-disk resume cache, so
        # appending the hardening block here would silently invalidate every `.cache`
        # entry ever written. The hardening is applied in
        # `ClaudeCodeExecutor._split_messages`, after the hash has been taken.
        text = self._template.render(*args, **kwargs)
        target_language = str(kwargs.get("target_language", "") or "")
        with self._lock:
            self._registry[text] = (self._kind, target_language)
        return text

    def __getattr__(self, name: str):
        return getattr(self._template, name)


class ClaudeCodeExecutor:
    """LLMExecutor-compatible executor backed by the `claude` CLI."""

    def __init__(
        self,
        model: str,
        timeout: float,
        retry_times: int,
        retry_interval_seconds: float,
        max_concurrency: int | None = None,
        claude_bin: str | None = None,
        prompt_registry: dict[str, tuple[str, str]] | None = None,
        registry_lock: "threading.Lock | None" = None,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._retry_times = retry_times
        self._retry_interval_seconds = retry_interval_seconds
        self._claude_bin = claude_bin
        self._prompt_registry = prompt_registry if prompt_registry is not None else {}
        self._registry_lock = registry_lock or threading.Lock()
        # Process-wide, so `glossary.py`'s direct run_claude_cli() calls obey it too.
        if max_concurrency is not None:
            set_max_concurrency(max_concurrency)
        # Installed here (main thread, at build_llm time) rather than at import time:
        # importing this module must not change signal handling for the HTTP providers.
        install_signal_handlers()

    def request(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        cache_key: str | None,
    ) -> str:
        # max_tokens / temperature / top_p have no CLI equivalent and are ignored.
        # cache_key is handled by LLMContext (the on-disk resume cache), not here.
        del max_tokens, temperature, top_p, cache_key
        system, user, validate = self._split_messages(messages)
        return run_claude_cli(
            system,
            user,
            model=self._model,
            timeout=self._timeout,
            retry_times=self._retry_times,
            retry_interval_seconds=self._retry_interval_seconds,
            claude_bin=self._claude_bin,
            validate=validate,
        )

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, str, Callable[[str], str | None] | None]:
        """Flatten the message list into (system prompt, single user prompt, validator).

        `claude -p` takes exactly one user turn, so a retry conversation
        (user / assistant / user, as the XML fill loop builds) is rendered as a
        role-labelled transcript inside that one turn.

        A *translation* turn additionally gets the hardening block and an explicit user
        label. Handing the CLI a bare short user turn makes it answer "I notice you
        haven't included the source text to translate…" about a third of the time; see
        `_TRANSLATE_HARDENING` for the measurements.

        Both are applied HERE rather than in the template, because LLMContext sha512s the
        library's `Message` list to key the on-disk resume cache: anything that changes a
        Message invalidates every cache entry ever written -- for a book, hours of
        re-translation. Applied here they change only what the CLI is handed, so existing
        `.cache` entries keep hitting (verified).
        """
        system_parts = [m.message for m in messages if m.role == MessageRole.SYSTEM]
        conversation = [m for m in messages if m.role != MessageRole.SYSTEM]
        system = "\n\n".join(part for part in system_parts if part) or None

        kind, target_language = "", ""
        if system is not None:
            with self._registry_lock:
                kind, target_language = self._prompt_registry.get(system, ("", ""))

        if kind == _KIND_TRANSLATE and system is not None:
            system += _TRANSLATE_HARDENING

        if len(conversation) == 1:
            source = conversation[0].message
            if kind == _KIND_TRANSLATE:
                validator = (
                    lambda response: _validate_translation(source, response, target_language)
                )
                return system, _TRANSLATE_USER_PREFIX + source, validator
            if kind == _KIND_FILL:
                return system, source, _validate_fill
            return system, source, None

        parts: list[str] = []
        for message in conversation:
            label = "Assistant" if message.role == MessageRole.ASSISTANT else "User"
            parts.append(f"{label}:\n{message.message}")
        # Multi-turn only ever happens on the fill retry loop.
        validator = _validate_fill if kind == _KIND_FILL else None
        return system, "\n\n".join(parts), validator


class ClaudeCodeLLM:
    """Drop-in replacement for `epub_translator.LLM` that shells out to Claude Code."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token_encoding: str = "o200k_base",
        timeout: float = 300.0,
        top_p: float | tuple[float, float] | None = None,
        temperature: float | tuple[float, float] | None = None,
        retry_times: int = 5,
        retry_interval_seconds: float = 6.0,
        cache_path: PathLike | str | None = None,
        max_concurrency: int | None = None,
        claude_bin: str | None = None,
    ) -> None:
        prompts_path = Path(str(files("epub_translator"))) / "data"
        self._templates: dict[str, _RegisteredTemplate] = {}
        self._encoding: Encoding = get_encoding(token_encoding)
        self._env: Environment = create_env(prompts_path)
        # Kept only so LLMContext has something to increase; the CLI ignores both.
        self._top_p = Increasable(top_p)
        self._temperature = Increasable(temperature)
        self._cache_path = self._ensure_dir_path(cache_path)
        # rendered system prompt -> (template kind, target language)
        self._prompt_registry: dict[str, tuple[str, str]] = {}
        self._registry_lock = threading.Lock()
        self._executor = ClaudeCodeExecutor(
            model=model,
            timeout=timeout,
            retry_times=retry_times,
            retry_interval_seconds=retry_interval_seconds,
            max_concurrency=max_concurrency,
            claude_bin=claude_bin,
            prompt_registry=self._prompt_registry,
            registry_lock=self._registry_lock,
        )

    @property
    def encoding(self) -> Encoding:
        return self._encoding

    def context(self, cache_seed_content: str | None = None) -> LLMContext:
        return LLMContext(
            executor=self._executor,
            cache_path=self._cache_path,
            cache_seed_content=cache_seed_content,
            top_p=self._top_p,
            temperature=self._temperature,
        )

    def template(self, template_name: str) -> Template:
        template = self._templates.get(template_name)
        if template is None:
            template = _RegisteredTemplate(
                self._env.get_template(template_name),
                template_name,
                self._prompt_registry,
                self._registry_lock,
            )
            self._templates[template_name] = template
        return template  # type: ignore[return-value]

    def _ensure_dir_path(self, path: PathLike | str | None) -> Path | None:
        if path is None:
            return None
        dir_path = Path(path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        elif not dir_path.is_dir():
            return None
        return dir_path.resolve()
