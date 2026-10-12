"""Тесты выжимки транскрипта: промпты, кэш, повтор пустого ответа и журнал."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import absorbent.summarize as summarize_module
from absorbent.config import ModelSpec, Provider
from absorbent.events import request_fields
from absorbent.summarize import (
    SummaryError,
    _prompt_language,
    run_summarize,
)

LLM_SPEC = ModelSpec(name="llm-a", provider="p", default=True)


def _provider(api_key_env: str = "TEST_KEY_A") -> Provider:
    return Provider(name="p", base_url="https://api.example.test/v1", api_key_env=api_key_env)


class TestPromptLanguage:
    def test_explicit_language_used(self) -> None:
        assert _prompt_language("ru") == "ru"
        assert _prompt_language("en") == "en"

    def test_unknown_falls_back_ru(self) -> None:
        # Язык выжимки всегда явный: неизвестное значение -> запас на 'ru'.
        assert _prompt_language("fr") == "ru"


class TestRunSummarize:
    def test_writes_and_returns_summary(self, tmp_path: Path, fake_openai,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """Ответ модели обрезается по краям и записан в файл кэша."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["  Тема: пределы.  "])
        out = tmp_path / "cache" / "summary.md"
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out)
        assert result == "Тема: пределы."
        assert out.read_text(encoding="utf-8") == "Тема: пределы."

    def test_ru_prompt_renders_transcript(self, tmp_path: Path, fake_openai,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["выжимка"])
        transcript = [{"start": 0.0, "end": 1.0, "text": "тезис", "speaker": None}]
        run_summarize(LLM_SPEC, _provider(), transcript, "ru", tmp_path / "summary.md")
        call = fake_openai.chat.completions.calls[0]
        assert call["model"] == "llm-a"
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][1] == {"role": "user",
                                       "content": "Транскрипт лекции:\n[00:00] тезис"}

    def test_ru_prompt_requirements(self, tmp_path: Path, fake_openai,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["выжимка"])
        run_summarize(LLM_SPEC, _provider(), [], "ru", tmp_path / "summary.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "краткую выжимку лекции" in system
        assert "600 слов" in system

    def test_en_prompt_requirements(self, tmp_path: Path, fake_openai,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["summary"])
        run_summarize(LLM_SPEC, _provider(), [], "en", tmp_path / "summary.md")
        system = fake_openai.chat.completions.calls[0]["messages"][0]["content"]
        assert "concise summary of a lecture" in system
        assert "600 words" in system

    def test_empty_transcript_marker_in_message(self, tmp_path: Path, fake_openai,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["выжимка"])
        run_summarize(LLM_SPEC, _provider(), [], "ru", tmp_path / "summary.md")
        content = fake_openai.chat.completions.calls[0]["messages"][1]["content"]
        assert content == "Транскрипт лекции:\n(transcript unavailable)"

    def test_model_call_is_journaled_as_summarize(self, tmp_path: Path, fake_openai,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """Вызов модели идёт под этапом 'summarize' с провайдером и именем модели."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["выжимка"])
        seen: list[tuple] = []
        real_api_call = summarize_module.api_call

        @contextmanager
        def spy(stage_name, provider, model):
            with real_api_call(stage_name, provider, model):
                seen.append((stage_name, provider, model, request_fields()))
                yield

        monkeypatch.setattr(summarize_module, "api_call", spy)
        run_summarize(LLM_SPEC, _provider(), [], "ru", tmp_path / "summary.md")
        assert seen == [("summarize", "p", "llm-a",
                         {"stage": "summarize", "provider": "p", "model": "llm-a"})]


class TestCache:
    def test_reused_without_force(self, tmp_path: Path, fake_openai,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """Готовый файл кэша возвращается как есть, модель не вызывается."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        out = tmp_path / "summary.md"
        out.write_text("старая выжимка", encoding="utf-8")
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out)
        assert result == "старая выжимка"
        assert fake_openai.chat.completions.calls == []

    def test_force_reasks(self, tmp_path: Path, fake_openai,
                          monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["новая выжимка"])
        out = tmp_path / "summary.md"
        out.write_text("старая выжимка", encoding="utf-8")
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out, force=True)
        assert result == "новая выжимка"
        assert out.read_text(encoding="utf-8") == "новая выжимка"
        assert len(fake_openai.chat.completions.calls) == 1

    def test_stale_relative_to_transcript_reasked(self, tmp_path: Path, fake_openai,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """Выжимка старше транскрипта негодна: транскрипт с тех пор изменился."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["свежая выжимка"])
        transcript = tmp_path / "transcript.json"
        transcript.write_text("[]", encoding="utf-8")
        out = tmp_path / "summary.md"
        out.write_text("старая выжимка", encoding="utf-8")
        os.utime(out, (1000.0, 1000.0))
        os.utime(transcript, (2000.0, 2000.0))
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out, transcript_path=transcript)
        assert result == "свежая выжимка"
        assert len(fake_openai.chat.completions.calls) == 1

    def test_fresh_relative_to_transcript_reused(self, tmp_path: Path, fake_openai,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY_A", "k")
        transcript = tmp_path / "transcript.json"
        transcript.write_text("[]", encoding="utf-8")
        out = tmp_path / "summary.md"
        out.write_text("актуальная выжимка", encoding="utf-8")
        os.utime(transcript, (1000.0, 1000.0))
        os.utime(out, (2000.0, 2000.0))
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out, transcript_path=transcript)
        assert result == "актуальная выжимка"
        assert fake_openai.chat.completions.calls == []

    def test_same_mtime_counts_as_fresh(self, tmp_path: Path, fake_openai,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """Одинаковое время правки возьмёт кэш: выжимка не старше транскрипта."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        transcript = tmp_path / "transcript.json"
        transcript.write_text("[]", encoding="utf-8")
        out = tmp_path / "summary.md"
        out.write_text("выжимка", encoding="utf-8")
        os.utime(transcript, (1500.0, 1500.0))
        os.utime(out, (1500.0, 1500.0))
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out, transcript_path=transcript)
        assert result == "выжимка"
        assert fake_openai.chat.completions.calls == []

    def test_transcript_path_none_caches_by_file_presence(self, tmp_path: Path, fake_openai,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        """Без transcript_path свежесть не проверяется: файл есть — берём его."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        out = tmp_path / "summary.md"
        out.write_text("выжимка", encoding="utf-8")
        os.utime(out, (1000.0, 1000.0))
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out, transcript_path=None)
        assert result == "выжимка"
        assert fake_openai.chat.completions.calls == []

    def test_missing_transcript_does_not_prove_freshness(self, tmp_path: Path, fake_openai,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """Транскрипта нет — сравнить не с чем, выжимка считается заново."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["новая выжимка"])
        out = tmp_path / "summary.md"
        out.write_text("старая выжимка", encoding="utf-8")
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out,
                               transcript_path=tmp_path / "transcript.json")
        assert result == "новая выжимка"
        assert len(fake_openai.chat.completions.calls) == 1


class TestEmptyAnswer:
    def test_empty_then_ok_written(self, tmp_path: Path, fake_openai,
                                   monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture) -> None:
        """Пустой ответ провайдера повторяется, второй ответ доезжает до файла."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        fake_openai.set_replies(["", "выжимка"])
        out = tmp_path / "summary.md"
        result = run_summarize(LLM_SPEC, _provider(), [], "ru", out)
        assert result == "выжимка"
        assert "повтор запроса" in capsys.readouterr().out

    def test_empty_twice_raises_with_diagnostics(self, tmp_path: Path, fake_openai,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
        """Два пустых ответа — ошибка этапа с диагностикой; файл не создаётся."""
        monkeypatch.setenv("TEST_KEY_A", "k")
        completions = fake_openai.chat.completions
        completions.finish_reason = "length"
        completions.reasoning = "долгие размышления"
        fake_openai.set_replies(["", "   "])
        out = tmp_path / "summary.md"
        with pytest.raises(SummaryError) as exc:
            run_summarize(LLM_SPEC, _provider(), [], "ru", out)
        message = str(exc.value)
        assert "finish_reason=length" in message
        assert "completion_tokens=123" in message
        assert "reasoning_tokens=99" in message
        assert "reasoning_content=18 симв." in message
        assert "повторите прогон: выжимка пойдёт заново" in message
        assert not out.exists()
        assert capsys.readouterr().out.count("повтор запроса") == 1
