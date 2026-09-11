"""Тесты извлечения кадров: кэш, вызов ffmpeg, ошибки."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from absorbent.extract import ExtractionError, extract_frames, has_video_stream


class TestExtractFrames:
    def test_rerun_reuses_cache(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        existing = [frames_dir / "frame_000001.jpg", frames_dir / "frame_000002.jpg"]
        for frame in existing:
            frame.write_bytes(b"fake-jpeg")
        # Подмена subprocess: если вызывается — тест падает.
        def _fail(*args, **kwargs):  # pragma: no cover
            raise AssertionError("ffmpeg не должен вызываться при заполненном кэше")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(subprocess, "run", _fail)
        try:
            result = extract_frames(tmp_path / "input.mp4", frames_dir, 1.0)
        finally:
            monkeypatch.undo()
        assert result == existing

    def test_runs_ffmpeg_when_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        frames_dir = tmp_path / "frames"
        input_path = tmp_path / "input.mp4"
        input_path.write_bytes(b"media")

        calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            assert "-nostdin" in cmd and "-loglevel" in cmd
            # Симулируем создание кадров ffmpeg'ом.
            (frames_dir / "frame_000001.jpg").write_bytes(b"jpeg1")
            (frames_dir / "frame_000002.jpg").write_bytes(b"jpeg2")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = extract_frames(input_path, frames_dir, 2.0)
        assert len(result) == 2
        assert calls and "fps=2.0" in calls[0]
        assert calls[0][-1] == str(frames_dir / "frame_%06d.jpg")

    def test_force_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000001.jpg").write_bytes(b"old")
        count = {"n": 0}

        def _fake_run(cmd, **kwargs):
            count["n"] += 1
            (frames_dir / "frame_000001.jpg").write_bytes(b"new")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = extract_frames(tmp_path / "input.mp4", frames_dir, 1.0, force=True)
        assert count["n"] == 1
        assert (frames_dir / "frame_000001.jpg").read_bytes() == b"new"
        assert result == [frames_dir / "frame_000001.jpg"]

    def test_ffmpeg_failure_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        frames_dir = tmp_path / "frames"

        def _fail_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: bad input")

        monkeypatch.setattr(subprocess, "run", _fail_run)
        with pytest.raises(ExtractionError, match="boom: bad input"):
            extract_frames(tmp_path / "input.mp4", frames_dir, 1.0)

    def test_real_ffmpeg_extracts_frames(self, tmp_path: Path) -> None:
        """Интеграционный тест на реальном ffmpeg (если установлен)."""
        ffmpeg = __import__("shutil").which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("ffmpeg не установлен")
        video = tmp_path / "video.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
            check=True,
        )
        frames = extract_frames(video, tmp_path / "frames", 1.0)
        assert len(frames) == 1
        assert frames[0].name == "frame_000001.jpg"


class TestHasVideoStream:
    def _probe(self, monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str
               ) -> list[list[str]]:
        calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        return calls

    def test_video_stream_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._probe(monkeypatch, 0, "0\n")
        assert has_video_stream(tmp_path / "lecture.mkv") is True
        # Запрос именно видеопотоков, поток вывода — csv без обёртки.
        assert calls[0][:6] == ["ffprobe", "-v", "error", "-select_streams", "v",
                                "-show_entries"]
        assert calls[0][-1] == str(tmp_path / "lecture.mkv")

    def test_no_video_stream(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._probe(monkeypatch, 0, "")
        assert has_video_stream(tmp_path / "audio.mp3") is False

    def test_ffprobe_error_is_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._probe(monkeypatch, 1, "")
        assert has_video_stream(tmp_path / "broken.mp4") is False
