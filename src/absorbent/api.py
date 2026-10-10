"""Общий клиент OpenAI и учёт каждой HTTP-попытки, включая повторы SDK."""

from __future__ import annotations

import math
import time
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

import httpx2
import openai

from absorbent.config import PLACEHOLDER_API_KEY, Provider, api_key_from
from absorbent.events import Journal, JournalError, current_journal, request_fields


def _number(value):
    """Сохранить только конечное неотрицательное измерение, не произвольный JSON."""
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is float and math.isfinite(value) and value >= 0:
        return value
    return None


def _usage(response: httpx2.Response) -> dict | None:
    """Извлечь разрешённые числовые поля; отсутствие usage не означает нулевой расход."""
    try:
        data = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return None
    usage = data["usage"]
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    input_details = usage.get("input_token_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    input_details = input_details if isinstance(input_details, dict) else {}
    return {
        "prompt_tokens": _number(usage.get("prompt_tokens")),
        "completion_tokens": _number(usage.get("completion_tokens")),
        "cache_hit": _number(usage.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens"))),
        "cache_miss": _number(usage.get("prompt_cache_miss_tokens")),
        "reasoning_tokens": _number(completion_details.get("reasoning_tokens")),
        "audio_tokens": _number(prompt_details.get("audio_tokens", input_details.get("audio_tokens"))),
        "completion_audio_tokens": _number(completion_details.get("audio_tokens")),
        "audio_seconds": _number(usage.get("seconds")),
        "prompt_audio_seconds": _number(usage.get("prompt_audio_seconds")),
        "completion_audio_seconds": _number(usage.get("completion_audio_seconds")),
        "input_tokens": _number(usage.get("input_tokens")),
        "output_tokens": _number(usage.get("output_tokens")),
        "total_tokens": _number(usage.get("total_tokens")),
    }


@dataclass
class _Attempt:
    journal: Journal
    request_id: str
    started: float
    fields: dict


class JournalHTTPClient(httpx2.Client):
    """Сохранить настройки SDK и не потерять транспортные ошибки между повторами."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("timeout", httpx2.Timeout(**openai.DEFAULT_TIMEOUT.as_dict()))
        limits = openai.DEFAULT_CONNECTION_LIMITS
        kwargs.setdefault("limits", httpx2.Limits(
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry))
        kwargs.setdefault("follow_redirects", True)
        kwargs["event_hooks"] = {"request": [self._request], "response": [self._response]}
        self._pending: ContextVar[_Attempt | None] = ContextVar("absorbent_http_attempt", default=None)
        super().__init__(**kwargs)

    def send(self, request: httpx2.Request, **kwargs) -> httpx2.Response:
        token = self._pending.set(None)
        try:
            return super().send(request, **kwargs)
        except JournalError:
            raise
        except (httpx2.TransportError, httpx2.DecodingError) as exc:
            self._finish(status_code=None, usage=None, error_type=type(exc).__name__, billing="unknown")
            raise
        finally:
            self._pending.reset(token)

    def _request(self, request: httpx2.Request) -> None:
        journal = current_journal()
        if journal is None:
            return
        attempt = _Attempt(journal, str(uuid4()), time.monotonic(), request_fields())
        self._pending.set(attempt)
        journal.write("api_request_start", request_id=attempt.request_id, **attempt.fields)

    def _response(self, response: httpx2.Response) -> None:
        # Хук вызывается до чтения тела; буфер нужен и журналу, и последующему разбору SDK.
        response.read()
        usage = _usage(response)
        self._finish(status_code=response.status_code, usage=usage, error_type=None,
                     billing="reported" if usage is not None else "unknown")

    def _finish(self, **fields) -> None:
        attempt = self._pending.get()
        if attempt is None:
            return
        # Снять pending до записи: ошибка журнала не должна закрывать попытку повторно.
        self._pending.set(None)
        attempt.journal.write("api_request_end", request_id=attempt.request_id, **attempt.fields,
                              elapsed_seconds=time.monotonic() - attempt.started, **fields)


def create_client(provider: Provider) -> openai.OpenAI:
    """Создать прежний клиент без журнала или его инструментированный HTTP-транспорт."""
    api_key = api_key_from(provider)
    kwargs = {}
    if current_journal() is not None:
        kwargs["http_client"] = JournalHTTPClient()
    return openai.OpenAI(base_url=provider.base_url, api_key=api_key or PLACEHOLDER_API_KEY, **kwargs)
