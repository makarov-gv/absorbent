"""Тесты синтеза Markdown-конспекта: рендеры, выбор промпта, запись файла."""

from __future__ import annotations

from pathlib import Path

import pytest

from absorbent.config import ModelSpec, Provider
from absorbent.synthesize import _prompt_language, _render_keyframes, _render_transcript, run_synthesize

LLM_SPEC = ModelSpec(name="llm-a", provider="p", default=True)


def _provider(api_key_env: str = "TEST_KEY_A") -> Provider:
    return Provider(name="p", base_url="https://api.example.test/v1", api_key_env=api_key_env)


class TestPromptLanguage:
    def test_explicit_language_used(self) -> None:
        assert _prompt_language("ru") == "ru"
        assert _prompt_language("en") == "en"

    def test_unknown_falls_back_ru(self) -> None:
        # Язык конспекта всегда явный: неизвестное значение -> запас на 'ru'.
        assert _prompt_language("fr") == "ru"


class TestRenderTranscript:
    def test_lines_with_timestamps(self) -> None:
        transcript = [
            {"start": 0.0, "end": 1.0, "text": "привет", "speaker": None},
            {"start": 61.0, "end": 65.0, "text": "мир", "speaker": "speaker_1"},
        ]
        rendered = _render_transcript(transcript)
        assert rendered == "[00:00] привет\n[01:01] [speaker_1] мир"

    def test_empty_is_unavailable_marker(self) -> None:
        assert _render_transcript([]) == "(transcript unavailable)"


class TestRenderKeyframes:
    def test_sorted_by_name(self) -> None:
        descriptions = {
            "frame_00_00_05.jpg": {"timestamp": "00:00:05", "description": "пять"},
            "frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "ноль"},
        }
        rendered = _render_keyframes(descriptions, "images/")
        assert rendered.splitlines()[0].startswith("- images/frame_00_00_00.jpg [00:00:00]: ноль")
        assert rendered.splitlines()[1].startswith("- images/frame_00_00_05.jpg [00:00:05]: пять")

    def test_custom_prefix(self) -> None:
        """Плоский режим: кадры лежат в <имя>.images/, и модель видит этот путь."""
        descriptions = {"frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "ноль"}}
        rendered = _render_keyframes(descriptions, "lecture.images/")
        assert rendered == "- lecture.images/frame_00_00_00.jpg [00:00:00]: ноль"


class TestRunSynthesize:
    def test_writes_notes_md_verbatim(self, tmp_path: Path, fake_openai,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        # run_synthesize делает strip() ответа модели, поэтому сравниваем
        # с обрезанным текстом.
        markdown = "# Конспект\n\nТекст лекции.\n"
        fake_openai.set_replies([markdown])
        out = tmp_path / "out" / "notes.md"
        result = run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", out)
        assert result == out
        assert out.read_text(encoding="utf-8") == markdown.strip()

    def test_ru_prompt_with_transcript_and_frames(self, tmp_path: Path, fake_openai,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        transcript = [{"start": 0.0, "end": 1.0, "text": "тезис", "speaker": None}]
        descriptions = {"frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "слайд"}}
        run_synthesize(LLM_SPEC, _provider(), transcript, descriptions, "ru",
                       tmp_path / "notes.md")
        call = fake_openai.chat.completions.calls[0]
        messages = call["messages"]
        assert messages[0]["role"] == "system"
        assert "структурированный конспект" in messages[0]["content"]
        assert "images/frame_00_00_00.jpg" in messages[1]["content"]
        assert "[00:00] тезис" in messages[1]["content"]
        assert call["model"] == "llm-a"

    def test_en_prompt_explicit_language(self, tmp_path: Path, fake_openai,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ok"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "en", tmp_path / "notes.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "structured Markdown summary" in system

    def test_ru_prompt_preserves_verbatim_quotes(self, tmp_path: Path, fake_openai,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        # Русский промпт велит писать по-русски, сохраняя дословные фразы
        # на другом языке (английском) без перевода.
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "Пиши весь конспект на русском языке" in system
        assert "сохраняй как есть" in system

    def test_custom_images_prefix_reaches_prompt_and_frames(self, tmp_path: Path, fake_openai,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
        """Плоский режим: префикс картинок уходит и в системный промпт, и в список кадров."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        descriptions = {"frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "слайд"}}
        run_synthesize(LLM_SPEC, _provider(), [], descriptions, "ru", tmp_path / "lecture.md",
                       images_prefix="lecture.images/")
        call = fake_openai.chat.completions.calls[0]
        assert "![подпись](lecture.images/<имя_файла>)" in call["messages"][0]["content"]
        assert "images/" not in call["messages"][0]["content"].replace("lecture.images/", "")
        assert "- lecture.images/frame_00_00_00.jpg [00:00:00]: слайд" in call["messages"][1]["content"]

    def test_unavailable_transcript_passed(self, tmp_path: Path, fake_openai,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ok"])
        descriptions = {"frame_00_00_00.jpg": {"timestamp": "00:00:00", "description": "слайд"}}
        run_synthesize(LLM_SPEC, _provider(), [], descriptions, "ru", tmp_path / "notes.md")
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert "(transcript unavailable)" in content
