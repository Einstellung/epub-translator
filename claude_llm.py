import json
import time
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

import requests
from epub_translator.llm.context import LLMContext
from epub_translator.llm.increasable import Increasable
from epub_translator.llm.types import Message, MessageRole
from epub_translator.template import create_env
from jinja2 import Environment, Template
from tiktoken import Encoding, get_encoding


class ClaudeExecutor:
    def __init__(
        self,
        api_key: str,
        url: str,
        model: str,
        timeout: float | None,
        retry_times: int,
        retry_interval_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._url = self._messages_url(url)
        self._model = model
        self._timeout = timeout
        self._retry_times = retry_times
        self._retry_interval_seconds = retry_interval_seconds

    def request(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        cache_key: str | None,
    ) -> str:
        del cache_key
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or 4096,
            "messages": self._convert_messages(messages),
        }
        system = self._system_prompt(messages)
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        last_error: Exception | None = None
        for attempt in range(self._retry_times + 1):
            try:
                response = requests.post(
                    self._url,
                    headers={
                        "x-api-key": self._api_key,
                        "Authorization": f"Bearer {self._api_key}",
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Claude API error {response.status_code}: {response.text[:1000]}"
                    )
                data = response.json()
                return self._extract_text(data)
            except Exception as err:
                last_error = err
                if attempt >= self._retry_times:
                    break
                if self._retry_interval_seconds > 0:
                    time.sleep(self._retry_interval_seconds)

        if last_error is None:
            raise RuntimeError("Claude request failed with unknown error")
        raise last_error

    def _messages_url(self, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def _system_prompt(self, messages: list[Message]) -> str:
        return "\n\n".join(
            message.message for message in messages if message.role == MessageRole.SYSTEM
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        converted: list[dict[str, str]] = []
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                continue
            role = "assistant" if message.role == MessageRole.ASSISTANT else "user"
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"] += "\n\n" + message.message
            else:
                converted.append({"role": role, "content": message.message})
        return converted or [{"role": "user", "content": ""}]

    def _extract_text(self, data: dict[str, Any]) -> str:
        parts: list[str] = []
        for item in data.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        if parts:
            return "".join(parts)
        raise RuntimeError(f"Claude response did not contain text content: {json.dumps(data)[:1000]}")


class ClaudeLLM:
    def __init__(
        self,
        key: str,
        url: str,
        model: str,
        token_encoding: str,
        timeout: float | None = None,
        top_p: float | tuple[float, float] | None = None,
        temperature: float | tuple[float, float] | None = None,
        retry_times: int = 5,
        retry_interval_seconds: float = 6.0,
        cache_path: PathLike | str | None = None,
    ) -> None:
        prompts_path = Path(str(files("epub_translator"))) / "data"
        self._templates: dict[str, Template] = {}
        self._encoding: Encoding = get_encoding(token_encoding)
        self._env: Environment = create_env(prompts_path)
        self._top_p = Increasable(top_p)
        self._temperature = Increasable(temperature)
        self._cache_path = self._ensure_dir_path(cache_path)
        self._executor = ClaudeExecutor(
            api_key=key,
            url=url,
            model=model,
            timeout=timeout,
            retry_times=retry_times,
            retry_interval_seconds=retry_interval_seconds,
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
            template = self._env.get_template(template_name)
            self._templates[template_name] = template
        return template

    def _ensure_dir_path(self, path: PathLike | str | None) -> Path | None:
        if path is None:
            return None
        dir_path = Path(path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        elif not dir_path.is_dir():
            return None
        return dir_path.resolve()
