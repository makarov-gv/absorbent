"""Тесты CLI: разбор аргументов, коды выхода, обработка директорий, ошибки."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from absorbent.cli import MEDIA_EXTENSIONS, build_parser, collect_inputs, main
from absorbent.config import ConfigError


def _write_config(tmp_path: Path) -> Path:
    """Конфигурация с одной моделью в каждой категории и без ключей API."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers:\n  p: {base_url: u, api_key_env: ''}\n"
        "stt:\n  s: {backend: whisper, whisper_model: tiny, default: true}\n"
        "vision:\n  v: {provider: p, default: true}\n"
        "llm:\n  l: {provider: p, default: true}\n",
        encoding="utf-8",
    )
    return config


class TestCollectInputs:
    def test_single_file(self, tmp_path: Path) -> None:
        video = tmp_path / "a.mp4"
        video.write_bytes(b"x")
        assert collect_inputs(str(video)) == [video]

    def test_directory_filters_and_sorts(self, tmp_path: Path) -> None:
        (tmp_path / "b.mkv").write_bytes(b"x")
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")  # не медиа.
        result = collect_inputs(str(tmp_path))
        assert [p.name for p in result] == ["a.mp4", "b.mkv"]

    def test_directory_supports_audio_extensions(self, tmp_path: Path) -> None:
        for ext in (".mp3", ".wav", ".opus"):
            (tmp_path / f"x{ext}").write_bytes(b"x")
        result = collect_inputs(str(tmp_path))
        assert len(result) == 3

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="не существует"):
            collect_inputs(str(tmp_path / "nope.mp4"))

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="не найдено медиафайлов"):
            collect_inputs(str(tmp_path))

    def test_extensions_set_matches_docs(self) -> None:
        assert ".mp4" in MEDIA_EXTENSIONS and ".mkv" in MEDIA_EXTENSIONS
        assert ".txt" not in MEDIA_EXTENSIONS


class TestParser:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["in.mp4"])
        assert args.language == "auto"
        assert args.output_language == "ru"
        assert args.fps == 1.0
        assert args.hash_threshold == 10
        assert args.min_scene_seconds == 5.0
        assert args.output_dir is None
        assert args.force is False
        assert args.skip_render is False
        assert args.slides == []
        assert args.slides_dpi == 150
        assert args.stt is None and args.vision_model is None and args.llm_model is None
        assert args.config is None
        assert args.flat is False

    def test_input_is_optional(self) -> None:
        """Без позиционного аргумента парсер не падает: вход спросит мастер."""
        assert build_parser().parse_args([]).input is None

    def test_explicit_values(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "in.mkv", "--config", "c.yaml", "--stt", "x", "--vision-model", "v",
            "--llm-model", "l", "--language", "ru", "--output-language", "en",
            "--fps", "2", "--hash-threshold", "5",
            "--min-scene-seconds", "2",
            "--output-dir", "out", "--force", "--skip-render", "--flat",
            "--slides", "deck.pdf", "--slides-dpi", "300",
        ])
        assert args.language == "ru" and args.output_language == "en"
        assert args.fps == 2.0 and args.hash_threshold == 5
        assert args.min_scene_seconds == 2.0
        assert args.skip_render is True
        assert args.stt == "x" and args.output_dir == "out" and args.force is True
        assert args.flat is True
        assert args.slides == ["deck.pdf"] and args.slides_dpi == 300

    def test_slides_flag_accumulates(self) -> None:
        args = build_parser().parse_args(["in.mp4", "--slides", "a.pdf",
                                          "--slides", "b.pptx"])
        assert args.slides == ["a.pdf", "b.pptx"]

    def test_bad_language_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mp4", "--language", "fr"])

    def test_bad_output_language_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["in.mp4", "--output-language", "fr"])

    def test_missing_input_exits_2(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2


class TestMain:
    def test_no_input_without_terminal_exits_2(self, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
        """Без TTY мастер не запускается: ошибка и код 2 вместо зависания на вводе."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2
        assert "мастер доступен только в терминале" in capsys.readouterr().err

    def test_wizard_arguments_run_pipeline(self, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        """Мастер возвращает argv; main разбирает его и запускает конвейер."""
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        deck = tmp_path / "deck.pdf"
        deck.write_bytes(b"%PDF")
        seen: dict = {}

        def _fake_process(input_path, cfg, args):
            seen["input"] = input_path
            seen["flat"] = args.flat
            seen["slides"] = args.slides
            seen["slides_dpi"] = args.slides_dpi
            return input_path.with_name("lec.md")

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("absorbent.wizard.run_wizard",
                            lambda cfg, config: [str(video), "--config", str(config), "--flat",
                                                 "--slides", str(deck), "--slides-dpi", "300"])
        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        assert main([]) == 0
        assert seen["input"] == video
        assert seen["flat"] is True
        assert seen["slides"] == [deck]  # слайды развёрнуты в пути до конвейера.
        assert seen["slides_dpi"] == 300

    def test_wizard_cancel_exits_130(self, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("absorbent.wizard.run_wizard", lambda cfg, config: None)
        assert main([]) == 130
        assert "отменено" in capsys.readouterr().err

    def test_slides_are_resolved_before_pipeline(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        """Каталог в --slides разворачивается в файлы презентаций."""
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()
        (slides_dir / "b.pdf").write_bytes(b"%PDF")
        (slides_dir / "a.pdf").write_bytes(b"%PDF")
        (slides_dir / "notes.txt").write_bytes(b"x")
        config = _write_config(tmp_path)
        seen: dict = {}

        def _fake_process(input_path, cfg, args):
            seen["slides"] = args.slides
            return input_path.with_name("lec.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(video), "--config", str(config), "--slides", str(slides_dir)])
        assert code == 0
        assert [path.name for path in seen["slides"]] == ["a.pdf", "b.pdf"]

    def test_missing_slides_path_exits_2(self, tmp_path: Path,
                                         capsys: pytest.CaptureFixture) -> None:
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        config = _write_config(tmp_path)
        code = main([str(video), "--config", str(config),
                     "--slides", str(tmp_path / "nope.pdf")])
        assert code == 2
        assert "не существует" in capsys.readouterr().err

    def test_processes_single_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        config = _write_config(tmp_path)
        processed: list[Path] = []

        def _fake_process(input_path, cfg, args):
            processed.append(input_path)
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(video), "--config", str(config)])
        assert code == 0
        assert processed == [video]

    def test_flat_flag_reaches_process_file(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        """--flat доезжает до process_file в аргументах."""
        video = tmp_path / "lec.mp4"
        video.write_bytes(b"x")
        config = _write_config(tmp_path)
        seen: dict = {}

        def _fake_process(input_path, cfg, args):
            seen["flat"] = args.flat
            return input_path.with_name("lec.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        assert main([str(video), "--config", str(config), "--flat"]) == 0
        assert seen["flat"] is True

    def test_processes_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.mkv").write_bytes(b"x")
        processed: list[str] = []

        def _fake_process(input_path, cfg, args):
            processed.append(input_path.name)
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(tmp_path), "--skip-render"])
        assert code == 0
        assert processed == ["a.mp4", "b.mkv"]

    def test_stage_error_continues_and_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        processed: list[str] = []

        def _fake_process(input_path, cfg, args):
            processed.append(input_path.name)
            if input_path.name == "a.mp4":
                raise RuntimeError("boom")
            return input_path.with_name("notes.md")

        monkeypatch.setattr("absorbent.pipeline.process_file", _fake_process)
        code = main([str(tmp_path)])
        assert code == 1
        assert processed == ["a.mp4", "b.mp4"]  # второй файл обработан несмотря на ошибку.

    def test_unknown_model_lists_valid(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "providers:\n  p: {base_url: u, api_key_env: ''}\n"
            "stt:\n  aa: {backend: whisper, whisper_model: tiny, default: true}\n"
            "vision:\n  bb: {provider: p, default: true}\n"
            "llm:\n  cc: {provider: p, default: true}\n",
            encoding="utf-8",
        )
        code = main([str(tmp_path / "x.mp4"), "--config", str(config), "--stt", "nope"])
        assert code == 2
        err = capsys.readouterr().err
        assert "неизвестная модель 'nope'" in err
        assert "aa" in err

    def test_missing_config_file(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["/tmp/whatever.mp4", "--config", "/nonexistent/config.yaml"])
        assert code == 2
        assert "не найден" in capsys.readouterr().err

    def test_empty_input_directory_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        code = main([str(tmp_path)])
        assert code == 2
        assert "медиафайлов" in capsys.readouterr().err
