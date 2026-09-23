"""Тесты синтеза Markdown-конспекта: рендеры, выбор промпта, запись файла."""

from __future__ import annotations

from pathlib import Path

import pytest

from absorbent.config import ModelSpec, Provider
from absorbent.synthesize import (
    SYNTHESIS_ATTEMPTS,
    SynthesisError,
    _prompt_language,
    _render_keyframes,
    _render_photos,
    _render_slides,
    _render_transcript,
    run_synthesize,
)

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

    def test_empty_answer_retried_then_written(self, tmp_path: Path, fake_openai,
                                               monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
        """Пустой ответ провайдера повторяется, второй ответ доезжает до notes.md."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["", "# Конспект\n"])
        out = tmp_path / "notes.md"
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", out)
        assert out.read_text(encoding="utf-8") == "# Конспект"
        assert len(fake_openai.chat.completions.calls) == SYNTHESIS_ATTEMPTS
        assert "повтор запроса" in capsys.readouterr().out

    def test_empty_answer_twice_raises_with_diagnostics(self, tmp_path: Path, fake_openai,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture) -> None:
        """Два пустых ответа — ошибка этапа с диагностикой; notes.md не создаётся."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        completions = fake_openai.chat.completions
        completions.finish_reason = "length"
        completions.reasoning = "долгие размышления"
        fake_openai.set_replies(["", "   "])
        out = tmp_path / "notes.md"
        with pytest.raises(SynthesisError) as exc:
            run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", out)
        message = str(exc.value)
        assert "finish_reason=length" in message
        assert "completion_tokens=123" in message
        assert "reasoning_tokens=99" in message
        assert "reasoning_content=18 симв." in message
        assert "--skip-stt --skip-frames --skip-describe" in message
        assert not out.exists()
        assert capsys.readouterr().out.count("повтор запроса") == 1


class TestRenderSlides:
    def test_sorted_by_name_with_page_labels(self) -> None:
        slides = {
            "slide_003.png": {"page": 3, "source": "/decks/deck.pdf", "description": "третья"},
            "slide_010.png": {"page": 10, "source": "/decks/deck.pdf", "description": "десятая"},
        }
        rendered = _render_slides(slides, "images/")
        assert rendered.splitlines() == [
            "- images/slide_003.png [слайд 3]: третья",
            "- images/slide_010.png [слайд 10]: десятая",
        ]

    def test_custom_prefix(self) -> None:
        """Плоский режим: слайды лежат в <имя>.images/, и модель видит этот путь."""
        slides = {"slide_001.png": {"page": 1, "source": "/decks/deck.pdf",
                                    "description": "первая"}}
        assert _render_slides(slides, "lecture.images/") == (
            "- lecture.images/slide_001.png [слайд 1]: первая")

    def test_photos_go_to_their_own_list(self) -> None:
        """Снимок доски не смешивается со страницами колоды и не получает номер слайда."""
        materials = {
            "slide_001.png": {"page": 1, "source": "/d.pdf", "kind": "slide",
                              "description": "первая"},
            "photo_001.jpg": {"page": 1, "source": "/доска.jpg", "kind": "photo",
                              "description": "формула на доске"},
        }
        assert _render_slides(materials, "images/") == "- images/slide_001.png [слайд 1]: первая"
        assert _render_photos(materials, "images/") == (
            "- images/photo_001.jpg [фотография доски]: формула на доске")


class TestRenderPhotos:
    def test_sorted_by_name_without_page_numbers(self) -> None:
        materials = {
            "photo_002.jpg": {"page": 2, "source": "/доска.jpg", "kind": "photo",
                              "description": "вторая"},
            "photo_001.JPG": {"page": 1, "source": "/доска.jpg", "kind": "photo",
                              "description": "первая"},
        }
        rendered = _render_photos(materials, "lecture.images/")
        assert rendered.splitlines() == [
            "- lecture.images/photo_001.JPG [фотография доски]: первая",
            "- lecture.images/photo_002.jpg [фотография доски]: вторая",
        ]

    def test_pages_without_kind_are_not_photos(self) -> None:
        """Описания, снятые до появления снимков, поля kind не имеют."""
        pages = {"slide_001.png": {"page": 1, "source": "/d.pdf", "description": "страница"}}
        assert _render_photos(pages, "images/") == ""


class TestSlidesInMessage:
    def _slides(self) -> dict:
        return {"slide_003.png": {"page": 3, "source": "/decks/deck.pdf",
                                  "description": "тезисы третьей страницы"}}

    def test_section_added_with_slides(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md",
                       slides=self._slides())
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert ("Слайды презентации (справочно: часть колоды могла остаться за рамками "
                "лекции):\n- images/slide_003.png [слайд 3]: тезисы третьей страницы") in content
        # Транскрипт идёт первым: он объявлен основным источником конспекта.
        assert content.index("Транскрипт лекции (основной источник") < content.index(
            "Слайды презентации")

    def test_no_section_without_slides(self, tmp_path: Path, fake_openai,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md", slides=None)
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert "Слайды презентации" not in content

    def test_empty_slides_dict_means_no_slides(self, tmp_path: Path, fake_openai,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md", slides={})
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert "Слайды презентации" not in content

    def test_prefix_reaches_slides(self, tmp_path: Path, fake_openai,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "lecture.md",
                       images_prefix="lecture.images/", slides=self._slides())
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert "- lecture.images/slide_003.png [слайд 3]" in content


class TestPhotosInMessage:
    """Снимок доски уходит в синтез отдельным разделом и встраивается как иллюстрация."""

    def _materials(self) -> dict:
        return {
            "slide_003.png": {"page": 3, "source": "/decks/deck.pdf", "kind": "slide",
                              "description": "третья страница"},
            "photo_001.jpg": {"page": 1, "source": "/снимки/доска.jpg", "kind": "photo",
                              "description": "формула на доске"},
        }

    def test_photo_section_separate_from_slides(self, tmp_path: Path, fake_openai,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md",
                       slides=self._materials())
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        slides_part, photos_part = content.split("Фотографии доски")[0], content.split(
            "Фотографии доски (снимки, сделанные на лекции):")[1]
        assert "photo_001.jpg" not in slides_part
        assert "slide_003.png" not in photos_part
        assert photos_part.strip() == (
            "- images/photo_001.jpg [фотография доски]: формула на доске")

    def test_photo_only_materials_add_no_slide_section(self, tmp_path: Path, fake_openai,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
        """Только снимок доски: раздела презентации во входе нет."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        materials = {"photo_001.jpg": self._materials()["photo_001.jpg"]}
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md",
                       slides=materials)
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert "Слайды презентации" not in content
        assert "Фотографии доски (снимки, сделанные на лекции):" in content


class TestPromptSourceHierarchy:
    """Промпт объявляет транскрипт источником, а слайды и снимки — справочным материалом."""

    def test_ru_prompt_states_transcript_is_primary(self, tmp_path: Path, fake_openai,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ок"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "ru", tmp_path / "notes.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "транскрипт с таймкодами — это речь лектора и единственный источник" in system
        assert "слайд, содержание которого не прозвучало в транскрипте" in system
        assert "«слайд N»" in system
        assert "![Слайд N](images/<имя_файла>)" in system
        assert "![Фотография доски](images/<имя_файла>)" in system

    def test_en_prompt_states_transcript_is_primary(self, tmp_path: Path, fake_openai,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["ok"])
        run_synthesize(LLM_SPEC, _provider(), [], {}, "en", tmp_path / "notes.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "the only source of what actually happened in the lecture" in system
        assert "a slide whose content was never spoken about does not appear" in system
        assert "![Slide N](images/<filename>)" in system
        assert "![Board photo](images/<filename>)" in system
