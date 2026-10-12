"""Тесты распознавания речи: кэш, whisper-бэкенд, voxtral-бэкенд, ошибки."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from absorbent.config import ConfigError, ModelSpec, Provider, load_config
from absorbent.events import Journal, stage
from absorbent.transcribe import STTError, run_stt

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _whisper_spec() -> ModelSpec:
    return ModelSpec(name="whisper-turbo", backend="whisper", whisper_model="turbo", default=True)


def _voxtral_spec() -> ModelSpec:
    return ModelSpec(name="voxtral-transcribe-2", backend="voxtral", provider="mistral",
                     model="voxtral-mini-latest")


def _write_cached(out_json: Path, language: str | None = None) -> None:
    """Заранее записать transcript.json (+ language.txt)."""
    out_json.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "привет", "speaker": None}]),
                        encoding="utf-8")
    if language is not None:
        out_json.with_name("language.txt").write_text(language, encoding="utf-8")


class TestCache:
    def test_returns_cached_language(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")

        def _boom(*a, **k):  # pragma: no cover - вызов не должен случиться.
            raise AssertionError("транскрипция не должна запускаться при кэше")

        monkeypatch.setattr("absorbent.transcribe._whisper_transcribe", _boom)
        monkeypatch.setattr("absorbent.transcribe._voxtral_transcribe", _boom)
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4",
                           "auto", out, tmp_path)
        assert detected == "ru"

    def test_cached_without_language_returns_none(self, tmp_path: Path) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out)
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4",
                           "auto", out, tmp_path)
        assert detected is None

    def test_force_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")
        calls = {"n": 0}

        def _fake_whisper(spec, input_path, language):
            calls["n"] += 1
            return [], None

        monkeypatch.setattr("absorbent.transcribe._whisper_transcribe", _fake_whisper)
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "input.mp4", "auto", out,
                tmp_path, force=True)
        assert calls["n"] == 1


class TestWhisperBackend:
    def test_transcribes_to_json(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["segments"] = [
            {"start": 0.0, "end": 2.5, "text": "  первый сегмент  "},
            {"start": 2.5, "end": 5.0, "text": "второй"},
        ]
        fake_whisper["language"] = "ru"
        input_path = tmp_path / "lecture.mp4"
        input_path.write_bytes(b"media")
        out = tmp_path / "transcript.json"

        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), input_path, "auto", out, tmp_path)
        assert detected == "ru"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [
            {"start": 0.0, "end": 2.5, "text": "первый сегмент", "speaker": None},
            {"start": 2.5, "end": 5.0, "text": "второй", "speaker": None},
        ]
        assert out.with_name("language.txt").read_text(encoding="utf-8") == "ru"
        # Параметры вызова whisper: auto -> language None, beam_size=5.
        call = fake_whisper["transcribe_calls"][0]
        assert call["language"] is None
        assert call["beam_size"] == 5
        assert call["verbose"] is False
        assert fake_whisper["load_calls"][0][0] == "turbo"

    def test_device_cpu_when_no_cuda(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["cuda"] = False
        fake_whisper["language"] = None
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "ru",
                tmp_path / "t.json", tmp_path)
        assert fake_whisper["load_calls"][0][1] == "cpu"

    def test_explicit_language_passed(self, tmp_path: Path, fake_whisper: dict) -> None:
        fake_whisper["language"] = "ru"
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "en",
                tmp_path / "t.json", tmp_path)
        assert fake_whisper["transcribe_calls"][0]["language"] == "en"

    def test_empty_segments_warns_and_writes_empty(self, tmp_path: Path, fake_whisper: dict,
                                                   capsys: pytest.CaptureFixture) -> None:
        fake_whisper["segments"] = []
        fake_whisper["language"] = None
        out = tmp_path / "transcript.json"
        detected = run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4",
                           "auto", out, tmp_path)
        assert detected is None
        assert json.loads(out.read_text(encoding="utf-8")) == []
        assert "пустой транскрипт" in capsys.readouterr().out

    def test_language_file_removed_when_no_detection(self, tmp_path: Path, fake_whisper: dict) -> None:
        out = tmp_path / "transcript.json"
        _write_cached(out, language="ru")
        fake_whisper["language"] = None
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "auto", out,
                tmp_path, force=True)
        assert not out.with_name("language.txt").exists()


class TestWhisperProgress:
    """Живой прогресс whisper: кадры мел-спектрограммы пересчитываются в секунды."""

    def _run(self, tmp_path: Path, fake_whisper: dict, frames: int) -> list[dict]:
        fake_whisper["frames"] = frames
        fake_whisper["segments"] = [{"start": 0.0, "end": 0.5, "text": "слово"}]
        fake_whisper["language"] = "ru"
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("stt"):
                run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4",
                        "auto", tmp_path / "t.json", tmp_path)
        return [row for row in (json.loads(line) for line in
                                path.read_text(encoding="utf-8").splitlines())
                if row["event"] == "stage_progress"]

    def test_start_and_final_seconds(self, tmp_path: Path, fake_whisper: dict) -> None:
        rows = self._run(tmp_path, fake_whisper, frames=30)
        assert rows[0]["stage"] == "stt"
        assert rows[0]["unit"] == "seconds"
        assert rows[0]["completed"] == 0.0
        assert rows[0]["total"] == pytest.approx(0.3)
        assert rows[-1]["completed"] == pytest.approx(0.3)
        assert rows[-1]["total"] == pytest.approx(0.3)

    def test_monotonic_seconds_when_interval_allows(self, tmp_path: Path, fake_whisper: dict,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("absorbent.events._PROGRESS_INTERVAL_SECONDS", 0.0)
        rows = self._run(tmp_path, fake_whisper, frames=5)
        completed = [row["completed"] for row in rows]
        # Старт, по событию на кадр, затем финальное подтверждение полноты.
        assert completed == pytest.approx([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05])
        assert all(row["total"] == pytest.approx(0.05) for row in rows)

    def test_no_progress_bars_reports_no_final_seconds(self, tmp_path: Path,
                                                       fake_whisper: dict) -> None:
        """Модель без полосы прогресса: финальные секунды не выдумываются.

        Whisper создаёт tqdm по кадрам мел-спектрограммы не всегда; тогда
        сообщать нечего, и этап не должен выдавать ложное завершение.
        """
        fake_whisper["segments"] = [{"start": 0.0, "end": 0.5, "text": "слово"}]
        fake_whisper["language"] = "ru"
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("stt"):
                run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4",
                        "auto", tmp_path / "t.json", tmp_path)
        progress = [row for row in (json.loads(line) for line in
                                    path.read_text(encoding="utf-8").splitlines())
                    if row["event"] == "stage_progress"]
        assert progress == []
        assert (tmp_path / "t.json").exists()

    def test_disabled_terminal_progress_still_reports_scanned_audio(self, tmp_path, fake_whisper, monkeypatch):
        import sys
        from absorbent.transcribe import _whisper_progress

        monkeypatch.setattr("absorbent.events._PROGRESS_INTERVAL_SECONDS", 0.0)
        path = tmp_path / "events.jsonl"
        with Journal(path), stage("stt"), _whisper_progress():
            with sys.modules["whisper.transcribe"].tqdm.tqdm(total=1000, disable=True) as progress:
                progress.update(250)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                current = [row for row in rows if row["event"] == "stage_progress"][-1]
                assert current["completed"] == pytest.approx(2.5)
                assert current["total"] == pytest.approx(10.0)

    def test_no_journal_leaves_whisper_untouched(self, tmp_path: Path, fake_whisper: dict) -> None:
        import sys

        fake_whisper["segments"] = []
        fake_whisper["language"] = None
        original = sys.modules["whisper.transcribe"].tqdm
        run_stt(_whisper_spec(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "auto",
                tmp_path / "t.json", tmp_path)
        assert sys.modules["whisper.transcribe"].tqdm is original


class TestVoxtralProgress:
    """Облачный STT: прогресс ограничен стартом и финалом по длительности дорожки."""

    def test_start_and_final_sent_seconds(self, tmp_path: Path, fake_openai,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        import absorbent.transcribe as transcribe_module

        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        monkeypatch.setattr(transcribe_module, "_extract_audio",
                            lambda input_path, audio_path: audio_path.write_bytes(b"wav"))
        durations: list[Path] = []

        def _duration(path: Path) -> float:
            durations.append(path)
            return 120.0

        monkeypatch.setattr(transcribe_module, "_audio_duration_seconds", _duration)
        fake_openai.audio.transcriptions.payload = {
            "text": "t", "language": "ru",
            "segments": [{"text": "раз", "start": 0.0, "end": 1.0, "speaker_id": None}],
            "usage": {},
        }
        provider = load_config(str(REPO_CONFIG)).providers["mistral"]
        path = tmp_path / "events.jsonl"
        with Journal(path):
            with stage("stt"):
                run_stt(_voxtral_spec(), provider, tmp_path / "in.mp4", "auto",
                        tmp_path / "t.json", tmp_path)
        rows = [row for row in (json.loads(line) for line in
                                path.read_text(encoding="utf-8").splitlines())
                if row["event"] == "stage_progress"]
        assert [(row["completed"], row["total"]) for row in rows] == [(0.0, 120.0), (120.0, 120.0)]
        assert all(row["unit"] == "seconds" and row["stage"] == "stt" for row in rows)


class TestVoxtralBackend:
    def _prepare(self, tmp_path: Path, fake_openai, monkeypatch: pytest.MonkeyPatch,
                 payload: dict) -> tuple[Path, Provider]:
        """Заглушить ffmpeg-хелперы и вернуть (out_json, provider mistral)."""
        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(transcribe_module, "_extract_audio",
                            lambda input_path, audio_path: audio_path.write_bytes(b"wav-data"))
        fake_openai.audio.transcriptions.payload = payload
        cfg = load_config(str(REPO_CONFIG))
        return tmp_path / "transcript.json", cfg.providers["mistral"]

    def test_voxtral_segments_mapped(self, tmp_path: Path, fake_openai,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        payload = {
            "text": "весь текст",
            "language": "ru",
            "segments": [
                {"text": " раз ", "start": 0.0, "end": 1.2, "speaker_id": "speaker_1",
                 "type": "transcription_segment"},
                {"text": "два", "start": 1.2, "end": 2.0, "speaker_id": None,
                 "type": "transcription_segment"},
                {"text": "", "start": 2.0, "end": 3.0, "speaker_id": None},  # пустой сегмент.
            ],
            "usage": {},
        }
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        detected = run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)
        assert detected == "ru"
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [
            {"start": 0.0, "end": 1.2, "text": "раз", "speaker": "speaker_1"},
            {"start": 1.2, "end": 2.0, "text": "два", "speaker": None},
        ]
        assert out.with_name("language.txt").read_text(encoding="utf-8") == "ru"
        # Параметры запроса: verbose_json, сегменты + диаризация, x-api-key.
        call = fake_openai.audio.transcriptions.calls[0]
        assert call["model"] == "voxtral-mini-latest"
        assert call["response_format"] == "verbose_json"
        assert call["extra_body"]["diarize"] is True
        assert call["extra_body"]["timestamp_granularities"] == "segment"
        assert call["extra_body"]["language"] is None
        assert call["extra_headers"] == {"X-Api-Key": "key123"}

    def test_voxtral_explicit_language(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        payload = {"text": "t", "language": "en", "segments": [], "usage": {}}
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "ru", out, tmp_path)
        assert fake_openai.audio.transcriptions.calls[0]["extra_body"]["language"] == "ru"

    def test_voxtral_fallback_single_segment(self, tmp_path: Path, fake_openai,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "key123")
        import absorbent.transcribe as transcribe_module

        payload = {"text": "только текст без сегментов", "language": None, "segments": [], "usage": {}}
        monkeypatch.setattr(transcribe_module, "_audio_duration_seconds", lambda path: 42.5)
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch, payload)
        detected = run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)
        assert detected is None
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == [{"start": 0.0, "end": 42.5, "text": "только текст без сегментов",
                         "speaker": None}]

    def test_voxtral_missing_key_raises(self, tmp_path: Path, fake_openai,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        out, provider = self._prepare(tmp_path, fake_openai, monkeypatch,
                                      {"text": "", "language": None, "segments": []})
        with pytest.raises(ConfigError, match="MISTRAL_API_KEY"):
            run_stt(_voxtral_spec(), provider, tmp_path / "input.mp4", "auto", out, tmp_path)

    def test_unknown_backend(self, tmp_path: Path) -> None:
        class Bogus:
            backend = "bogus"
            name = "x"
            api_model = "x"

        with pytest.raises(STTError, match="bogus"):
            run_stt(Bogus(), Provider(name="x", base_url=""), tmp_path / "in.mp4", "auto",
                    tmp_path / "t.json", tmp_path)


class TestAudioHelpers:
    def test_extract_audio_ffmpeg_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        def _fail(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no audio track")

        monkeypatch.setattr(subprocess, "run", _fail)
        with pytest.raises(STTError, match="no audio track"):
            transcribe_module._extract_audio(tmp_path / "in.mp4", tmp_path / "audio.wav")

    def test_audio_duration_seconds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="12.5\n", stderr=""),
        )
        assert transcribe_module._audio_duration_seconds(tmp_path / "audio.wav") == 12.5

    def test_audio_duration_seconds_failure_returns_zero(self, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        monkeypatch.setattr(
            subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err"),
        )
        assert transcribe_module._audio_duration_seconds(tmp_path / "audio.wav") == 0.0


    def test_extract_audio_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import absorbent.transcribe as transcribe_module

        def _ok(cmd, **kwargs):
            (tmp_path / "audio.wav").write_bytes(b"wav")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _ok)
        transcribe_module._extract_audio(tmp_path / "in.mp4", tmp_path / "audio.wav")
        assert (tmp_path / "audio.wav").read_bytes() == b"wav"
