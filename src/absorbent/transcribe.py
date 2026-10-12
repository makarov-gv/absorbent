"""Распознавание речи: whisper (локально) и voxtral (Mistral API)."""

from __future__ import annotations

import importlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from absorbent.api import create_client
from absorbent.config import ModelSpec, Provider, api_key_from
from absorbent.events import api_call, current_journal, report_progress, skip_stage

# Канонический промежуточный формат — transcript.json:
# [{"start": float, "end": float, "text": str, "speaker": str | None}, ...]


class STTError(Exception):
    """Ошибка этапа распознавания речи."""


def _write_transcript(out_json: Path, segments: list[dict]) -> None:
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(segments, handle, ensure_ascii=False, indent=2)


@contextmanager
def _whisper_progress() -> Iterator[None]:
    """Сообщать журналу обработанные секунды whisper по ходу распознавания.

    Своего колбэка у [whisper.model.Whisper.transcribe()] нет: прогресс ведёт
    внутренний tqdm по кадрам мел-спектрограммы. Пока идёт вызов, подменяем
    ссылку на tqdm в модуле [whisper.transcribe] адаптером, который переводит
    кадры в секунды (один кадр — 10 мс) и передаёт их как stage_progress.
    Без активного журнала ничего не подменяем и не импортируем whisper.
    """
    if current_journal() is None:
        yield
        return
    import tqdm as tqdm_module
    from whisper.audio import HOP_LENGTH, SAMPLE_RATE

    seconds_per_frame = HOP_LENGTH / SAMPLE_RATE
    whisper_transcribe = importlib.import_module("whisper.transcribe")
    state: dict[str, float | None] = {"total": None}

    class _RecordingTqdm(tqdm_module.tqdm):
        """tqdm whisper: кадры мел-спектрограммы превращает в секунды аудио."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._processed_frames = 0
            total = self.total * seconds_per_frame if self.total else None
            state["total"] = total
            report_progress("seconds", 0.0, total, force=True)

        def update(self, n: float = 1) -> None:
            # При disable=True tqdm не увеличивает n, но whisper продолжает обработку.
            self._processed_frames += n
            super().update(n)
            report_progress("seconds", self._processed_frames * seconds_per_frame, state["total"])

    class _TqdmShim:
        tqdm = _RecordingTqdm

    previous = whisper_transcribe.tqdm
    whisper_transcribe.tqdm = _TqdmShim
    try:
        yield
    finally:
        whisper_transcribe.tqdm = previous
    # Финал — только при успешном возврате: при ошибке весь файл не обработан.
    total = state["total"]
    if total is not None:
        report_progress("seconds", total, total, force=True)


def _whisper_transcribe(spec: ModelSpec, input_path: Path, language: str) -> tuple[list[dict], str | None]:
    """Локальная транскрипция через openai-whisper; возвращает (сегменты, язык)."""
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(spec.whisper_model, device=device)
    with _whisper_progress():
        result = model.transcribe(
            str(input_path),
            language=(None if language == "auto" else language),
            beam_size=5,
            verbose=False,
        )
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip(), "speaker": None}
        for seg in result["segments"]
    ]
    detected = result.get("language")
    return segments, detected


def _audio_duration_seconds(audio_path: Path) -> float:
    """Длительность аудиофайла через ffprobe (для одиночного сегмента-фолбэка)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         str(audio_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0.0
    return float(proc.stdout.strip())


def _extract_audio(input_path: Path, audio_path: Path) -> None:
    """Извлечь моно-дорожку 16 кГц для отправки в Voxtral."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(input_path), "-vn", "-ar", "16000", "-ac", "1",
         str(audio_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise STTError(f"ffmpeg не извлёк аудио: {proc.stderr.strip()}")


def _voxtral_transcribe(spec: ModelSpec, provider: Provider, input_path: Path, language: str,
                        audio_path: Path) -> tuple[list[dict], str | None]:
    """Облачная транскрипция через OpenAI-совместимый /v1/audio/transcriptions Mistral.

    Диаризация и тайминги запрашиваются явно (diarize, сегменты); имена полей
    ответа соответствуют живой документации Mistral: segments[].{text,start,end,
    speaker_id}. Ошибки API всплывают как STTError.
    """
    api_key = api_key_from(provider)
    # Заголовок x-api-key: именно так устроен задокументированный curl-вызов
    # Mistral; Bearer от OpenAI SDK дублируется на тот случай, если
    # OpenAI-совместимая прослойка ожидает именно его.
    extra_headers = {"X-Api-Key": api_key} if api_key else {}
    client = create_client(provider)
    _extract_audio(input_path, audio_path)
    # У облачного вызова промежуточного прогресса нет: сообщаем только старт и
    # финал по длительности отправленной дорожки (реальный объём работы).
    total_seconds = _audio_duration_seconds(audio_path) if current_journal() else 0.0
    if total_seconds:
        report_progress("seconds", 0.0, total_seconds, force=True)

    with open(audio_path, "rb") as audio_file, api_call("stt", provider.name, spec.api_model):
        # Mistral-специфичные поля уходят через extra_body, а не именованные
        # аргументы: diarize отсутствует в сигнатуре OpenAI SDK, а типизация
        # timestamp_granularities меняется между версиями openai.
        raw = client.audio.transcriptions.with_raw_response.create(
            model=spec.api_model,
            file=(audio_path.name, audio_file, "audio/wav"),
            response_format="verbose_json",
            extra_headers=extra_headers,
            extra_body={
                "diarize": True,
                "timestamp_granularities": "segment",
                "language": (None if language == "auto" else language),
            },
        )
    data = raw.http_response.json()

    segments = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
            "speaker": seg.get("speaker_id") or seg.get("speaker"),
        })
    if not segments and data.get("text"):
        # Фолбэк без сегментов: один сегмент на всю длительность дорожки.
        segments.append({
            "start": 0.0,
            "end": _audio_duration_seconds(audio_path),
            "text": data["text"].strip(),
            "speaker": None,
        })
    detected = data.get("language") or None
    if total_seconds:
        report_progress("seconds", total_seconds, total_seconds, force=True)
    return segments, detected


def run_stt(spec: ModelSpec, provider: Provider, input_path: Path, language: str, out_json: Path,
            work_dir: Path, force: bool = False) -> str | None:
    """Запустить распознавание речи; вернуть обнаруженный язык ("ru"/"en"/None).

    Пишет transcript.json в out_json, а служебные файлы этапа — в work_dir
    (каталог кэша): language.txt и извлечённое для облачного Voxtral аудио
    audio.wav. В плоском режиме результат (transcript.json) лежит рядом с
    медиафайлом, а кэш — в <каталог>/.absorbent/<имя>/, поэтому пути
    результата и кэша разведены. При наличии кэша и без force возвращает
    язык из кэша, не транскрибируя. Пустой результат (нет аудиодорожки)
    пишется как [] с предупреждением.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    lang_file = work_dir / "language.txt"
    if out_json.exists() and not force:
        skip_stage()
        if lang_file.exists():
            return lang_file.read_text(encoding="utf-8").strip() or None
        return None

    if spec.backend == "whisper":
        segments, detected = _whisper_transcribe(spec, input_path, language)
    elif spec.backend == "voxtral":
        audio_path = work_dir / "audio.wav"
        segments, detected = _voxtral_transcribe(spec, provider, input_path, language, audio_path)
    else:
        raise STTError(f"неизвестный backend распознавания: '{spec.backend}'")

    if not segments:
        print("внимание: пустой транскрипт (нет аудиодорожки или речи)")
    _write_transcript(out_json, segments)
    if detected:
        lang_file.write_text(detected, encoding="utf-8")
    elif lang_file.exists():
        lang_file.unlink()
    return detected
