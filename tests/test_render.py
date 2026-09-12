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
    GITHUB_CSS,
    HTML_COLUMN_PX,
    _bg_default,
    _extract_title,
    _page_css,
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


def _capture(cmd: list[str]) -> dict:
    """Снимок временных файлов, которые pandoc получает в момент вызова.

    Каталог с ними живёт только внутри render_html/render_pdf, поэтому
    читать их нужно из подменённого subprocess.run, а не после возврата.
    """
    return {
        "body": Path(cmd[1]).read_text(encoding="utf-8"),
        "parts": {flag: Path(cmd[cmd.index(flag) + 1]).read_text(encoding="utf-8")
                  for flag in ("-H", "-B", "-A") if flag in cmd},
    }


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


class TestPageCss:
    """Фон страницы и ширина полосы: штатные стили pandoc их не задают верно."""

    def test_takes_theme_colors_from_stylesheet(self) -> None:
        css = _page_css(GITHUB_CSS.read_text(encoding="utf-8"))
        assert "background-color: #ffffff" in css
        assert "@media (prefers-color-scheme: dark)" in css
        assert "#0d1117" in css

    def test_color_lookup_uses_fallback_when_query_absent(self) -> None:
        assert _bg_default(".markdown-body { color: red; }",
                           "@media (prefers-color-scheme: dark)", "#000") == "#000"

    def test_color_lookup_uses_fallback_when_value_absent(self) -> None:
        css = "@media (prefers-color-scheme: dark) { .markdown-body { color: red; } }"
        assert _bg_default(css, "@media (prefers-color-scheme: dark)", "#000") == "#000"

    def test_color_lookup_takes_first_value_after_query(self) -> None:
        css = ("@media (prefers-color-scheme: light) { .x { --bgColor-default: #fff; } }\n"
               "@media (prefers-color-scheme: dark) { .x { --bgColor-default: #111; } }")
        assert _bg_default(css, "@media (prefers-color-scheme: light)", "#000") == "#fff"
        assert _bg_default(css, "@media (prefers-color-scheme: dark)", "#000") == "#111"

    def test_resets_pandoc_body_column(self) -> None:
        css = _page_css("")
        assert "max-width: none" in css
        assert f"max-width: {HTML_COLUMN_PX}px" in css


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
            seen.update(_capture(cmd))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("<html>ok</html>", encoding="utf-8")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_html(notes, out)
        cmd = seen["cmd"]
        assert "--standalone" in cmd and "--embed-resources" in cmd
        assert "--mathml" in cmd
        assert seen["check"] is True
        assert cmd[cmd.index("-V") + 1] == "lang=ru"
        # title уходит в метаданные pandoc.
        assert any(c.startswith("pagetitle=Конспект лекции") for c in cmd)
        # Заголовок остаётся и в теле: у HTML-версии должен быть видимый h1.
        assert seen["body"].startswith("# Конспект лекции")
        # CSS и обёртка задаются временными файлами (-H/-B/-A).
        assert "<style>" in seen["parts"]["-H"]
        assert "markdown-body.page" in seen["parts"]["-H"]
        assert 'class="markdown-body page"' in seen["parts"]["-B"]
        assert 'class="page-inner"' in seen["parts"]["-B"]
        assert seen["parts"]["-A"].strip() == "</div>\n</div>"

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

    def test_language_reaches_pandoc(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            (tmp_path / "notes.html").write_text("x", encoding="utf-8")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_html(notes, tmp_path / "notes.html", lang="en")
        assert seen["cmd"][seen["cmd"].index("-V") + 1] == "lang=en"


class TestRenderPdf:
    def test_lualatex_with_fonts(self, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = _notes(tmp_path)
        out = tmp_path / "notes.pdf"
        seen: dict = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen.update(_capture(cmd))
            out.write_bytes(b"%PDF")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        render_pdf(notes, out)
        cmd = seen["cmd"]
        assert "--pdf-engine=lualatex" in cmd
        assert any(c == "mainfont=DejaVu Serif" for c in cmd)
        assert any(c == "geometry:a4paper,margin=1.8cm" for c in cmd)
        assert any(c == "lang=ru" for c in cmd)
        assert any(c.startswith("title=Конспект лекции") for c in cmd)
        # Заголовок вынесен из тела в метаданные title.
        assert not seen["body"].startswith("# ")
        # Преамбула делает рисунки неразрывными с текстом и разрешает переносы.
        assert "\\renewenvironment{figure}" in seen["parts"]["-H"]
        assert "\\emergencystretch" in seen["parts"]["-H"]

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

    def test_explicit_output_paths(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """Плоский режим: HTML и PDF называются по имени медиафайла."""
        monkeypatch.setattr(shutil, "which", _fake_which)
        notes = tmp_path / "lecture.md"
        notes.write_text("# Лекция\n\n![кадр](lecture.images/frame_00_00_00.jpg)\n",
                         encoding="utf-8")
        order: list[str] = []

        def _fake_run(cmd, **kwargs):
            order.append(cmd[0].rsplit("/", 1)[-1])
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"out")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        html, pdf = render_all(notes, tmp_path / "lecture.html", tmp_path / "lecture.pdf")
        assert (html, pdf) == (tmp_path / "lecture.html", tmp_path / "lecture.pdf")
        assert html.exists() and pdf.exists()
        assert order == ["fmt-md-tables", "pandoc", "pandoc"]
