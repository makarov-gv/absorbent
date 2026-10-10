"""Машинный журнал этапов и реальных HTTP-попыток без содержимого лекции."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO
from uuid import uuid4

from openai import OpenAIError


class JournalError(OpenAIError):
    """Ошибка журнала, которую SDK обязан пропустить без повторного запроса."""


_sink: ContextVar[Journal | None] = ContextVar("absorbent_journal", default=None)
_active_stage: ContextVar[StageStatus | None] = ContextVar("absorbent_stage", default=None)
_call: ContextVar[dict | None] = ContextVar("absorbent_api_call", default=None)


class Journal:
    """Дописывать JSONL, сбрасывая каждое событие на диск до продолжения работы."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = str(uuid4())
        self.seq = 0
        self.exit_code = 0
        self._file: TextIO | None = None
        self._failure: JournalError | None = None

    def open(self) -> None:
        """Открыть файл отдельно от начала прогона для ранней диагностики CLI."""
        if self._file is not None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            raise JournalError(f"не удалось открыть журнал: {exc}") from exc

    def __enter__(self) -> Journal:
        from absorbent.version import get_version_info

        self.open()
        self._token = _sink.set(self)
        try:
            self._version = get_version_info()
            emit("run_start", **self._version)
        except BaseException:
            _sink.reset(self._token)
            self._close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            code = self.exit_code if exc_type is None else 1
            emit("run_end", **self._version, status="ok" if code == 0 else "error",
                 exit_code=code)
        finally:
            _sink.reset(self._token)
            self._close()

    def _close(self) -> None:
        try:
            self._file.close()
        except OSError as exc:
            raise JournalError(f"не удалось закрыть журнал: {exc}") from exc
        finally:
            self._file = None

    def write(self, event: str, **fields) -> None:
        """Записать событие; после потери журнала больше не продолжать обработку."""
        if self._failure is not None:
            raise self._failure
        self.seq += 1
        record = dict(fields, schema_version=1, run_id=self.run_id, seq=self.seq,
                      event=event, timestamp_utc=datetime.now(timezone.utc).isoformat())
        try:
            self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._file.flush()
        except OSError as exc:
            self._failure = JournalError(f"не удалось записать журнал: {exc}")
            raise self._failure from exc


def current_journal() -> Journal | None:
    """Активный журнал текущего контекста, если он включён."""
    return _sink.get()


def emit(event: str, **fields) -> None:
    """Записать событие только при явно включённом журнале."""
    journal = current_journal()
    if journal is not None:
        journal.write(event, **fields)


class StageStatus:
    """Итог этапа; кэш и отсутствие входа обозначаются явным пропуском."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "ok"

    def skip(self) -> None:
        self.status = "skipped"


def skip_stage() -> None:
    """Отметить пропуск активного этапа из его реализации."""
    active = _active_stage.get()
    if active is not None:
        active.skip()


@contextmanager
def stage(name: str) -> Iterator[StageStatus]:
    """Закрыть начатый этап и восстановить контекст даже при исключении."""
    status = StageStatus(name)
    token = _active_stage.set(status)
    started = time.monotonic()
    try:
        emit("stage_start", stage=name)
        try:
            yield status
        except BaseException:
            status.status = "error"
            raise
        finally:
            emit("stage_end", stage=name, status=status.status,
                 elapsed_seconds=time.monotonic() - started)
    finally:
        _active_stage.reset(token)


@contextmanager
def api_call(stage_name: str, provider: str, model: str) -> Iterator[None]:
    """Привязать HTTP-попытки SDK к этапу, провайдеру и выбранной модели."""
    token = _call.set({"stage": stage_name, "provider": provider, "model": model})
    try:
        yield
    finally:
        _call.reset(token)


def request_fields() -> dict:
    """Только метаданные вызова, без ключей, заголовков, URL и тела запроса."""
    fields = _call.get()
    if fields is not None:
        return fields
    active = _active_stage.get()
    return {"stage": active.name if active is not None else None, "provider": None, "model": None}
