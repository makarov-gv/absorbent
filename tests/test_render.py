"""Тесты постпроцессинга: извлечение title, вызовы fmt-md-tables/pandoc.

Внешние программы (fmt-md-tables, pandoc) подменяются: тесты проверяют
формирование команд и побочные файлы, а не реальные вызовы.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from absorbent.render import (
    _extract_title,
    _tool,
    fmt_tables,
    render_all,
    render_html,
    render_pdf,
)


def _fake_which(name: str) -> str:
    """Все внешние инструменты «находятся» в PATH."""
    return f"/usr/bin/{name}"


def _notes(tmp_path: Path, with_title: bool = True) -> Path:
    """notes.md с таблицей и (опционально) первым заголовком."""
    body = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    if with_title:
        body = "# Конспект лекции\n\n" + body
    path = tmp_path / "notes.md"
    path.write_text(body, encoding="utf-8")
    return path


class TestExtractTitle:
    def test_splits_first_h1(self) -> None:
        title, body = _extract_title("# Заголовок\n\nТекст.\n")
        assert title == "Заголовок"
        assert body == "Текст."

    def test_leading_blank_lines_before_h1(self) -> None:
        title, body = _extract_title("\n\n# Поздний заголовок\nТело\n")
        assert title == "Поздний заголовок"
        assert body == "Тело"

    def test_no_h1_returns_none_and_full_text(self) -> None:
        title, body = _extract_title("Просто текст\n")
        assert title is None
        assert body == "Просто текст\n"


class TestTool:
    def test_returns_path_when_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        assert _tool("pandoc") == "/usr/bin/pandoc"

    def test_missing_tool_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="skip-render"):
            _tool("pandoc")


class TestFmtTables:
    def test_calls_external_script(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["check"] = kwargs.get("check")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        fmt_tables(notes)
        assert seen["cmd"] == ["/usr/bin/fmt-md-tables", "--no-backup", str(notes)]
        assert seen["check"] is True


class TestRenderHtml:
    def test_builds_standalone_html(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        out = tmp_path / "out" / "notes.html"
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["check"] = kwargs.get("check")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("<html>ok</html>", encoding="utf-8")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_html(notes, out)
        cmd = seen["cmd"]
        assert "--standalone" in cmd and "--embed-resources" in cmd
        assert "--mathml" in cmd
        assert seen["check"] is True
        # title уходит в метаданные pandoc.
        assert any(c.startswith("pagetitle=Конспект лекции") for c in cmd)
        # Обёртка .markdown-body и CSS задаются временными файлами (-H/-B/-A).
        assert "-H" in cmd and "-B" in cmd and "-A" in cmd
        header = Path(cmd[cmd.index("-H") + 1])
        header_text = header.read_text(encoding="utf-8")
        assert "<style>" in header_text and "markdown-body" in header_text
        before = Path(cmd[cmd.index("-B") + 1])
        assert before.read_text(encoding="utf-8").strip() == '<article class="markdown-body">'
        after = Path(cmd[cmd.index("-A") + 1])
        assert after.read_text(encoding="utf-8").strip() == "</article>"

    def test_html_without_title_omits_metadata(self, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path, with_title=False)
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            (tmp_path / "notes.html").write_text("x", encoding="utf-8")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_html(notes, tmp_path / "notes.html")
        assert not any(c.startswith("pagetitle=") for c in seen["cmd"])


class TestRenderPdf:
    def test_lualatex_with_fonts(self, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        out = tmp_path / "notes.pdf"
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            out.write_bytes(b"%PDF")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_pdf(notes, out)
        cmd = seen["cmd"]
        assert "--pdf-engine=lualatex" in cmd
        assert any(c == "mainfont=DejaVu Serif" for c in cmd)
        assert any(c == "geometry:margin=2cm" for c in cmd)
        assert any(c.startswith("title=Конспект лекции") for c in cmd)

    def test_pdf_without_title_omits_metadata(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path, with_title=False)
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            (tmp_path / "notes.pdf").write_bytes(b"%PDF")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_pdf(notes, tmp_path / "notes.pdf")
        assert not any(c.startswith("title=") for c in seen["cmd"])


class TestRenderAll:
    def test_runs_fmt_then_html_then_pdf(self, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        order: list[str] = []

        def _fake_run(cmd, **kwargs):
            order.append(cmd[0].rsplit("/", 1)[-1])
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"out")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        html, pdf = render_all(notes)
        assert html == tmp_path / "notes.html"
        assert pdf == tmp_path / "notes.pdf"
        assert html.exists() and pdf.exists()
        assert order == ["fmt-md-tables", "pandoc", "pandoc"]

    def test_missing_tool_fails_fast(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        notes = _notes(tmp_path)
        with pytest.raises(FileNotFoundError):
            render_all(notes)
