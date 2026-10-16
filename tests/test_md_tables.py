"""Тесты модуля постпроцессинга Markdown [md_tables]."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from absorbent.md_tables import (
    align_table,
    comment_marker,
    find_tables,
    fmt_cell,
    fmt_sep,
    format_lines,
    glued_block,
    is_sep,
    main,
    parse_row,
    process_file,
    read_table,
    separate_blocks,
    sep_alignment,
    table_row,
)

MODULE = Path(__file__).resolve().parents[1] / "src" / "absorbent" / "md_tables.py"


class TestCommentMarker:
    def test_known_suffixes(self) -> None:
        assert comment_marker(Path("a.md")) == ""
        assert comment_marker(Path("a.markdown")) == ""
        assert comment_marker(Path("a.pyi")) == "#"

    def test_case_insensitive(self) -> None:
        assert comment_marker(Path("a.PY")) == "#"

    def test_unknown_suffix_is_markdown(self) -> None:
        assert comment_marker(Path("a.txt")) == ""


class TestTableRow:
    def test_markdown_row(self) -> None:
        assert table_row("| a | b |", "") == ("", "| a | b |")

    def test_markdown_non_row(self) -> None:
        assert table_row("просто текст", "") is None

    def test_markdown_row_without_trailing_pipe(self) -> None:
        assert table_row("| a | b", "") is None

    def test_commented_row_keeps_prefix(self) -> None:
        assert table_row("# | a | b |", "#") == ("# ", "| a | b |")

    def test_indented_comment_prefix_kept_verbatim(self) -> None:
        assert table_row("\t#\t| a |", "#") == ("\t#\t", "| a |")

    def test_uncommented_line_is_ignored(self) -> None:
        assert table_row("x = 1", "#") is None

    def test_comment_without_pipes_is_ignored(self) -> None:
        assert table_row("# просто текст", "#") is None


class TestParseRow:
    def test_both_pipes_stripped(self) -> None:
        assert parse_row("| a | b |") == ["a", "b"]

    def test_no_outer_pipes(self) -> None:
        assert parse_row("a|b") == ["a", "b"]

    def test_empty_cells(self) -> None:
        assert parse_row("|   |") == [""]


class TestIsSep:
    def test_plain_dashes(self) -> None:
        assert is_sep(["---", "---"]) is True

    def test_alignment_colons(self) -> None:
        assert is_sep([":---", "---:"]) is True

    def test_text_is_not_separator(self) -> None:
        assert is_sep(["a"]) is False

    def test_too_few_dashes(self) -> None:
        assert is_sep([":--:"]) is False


class TestSepAlignment:
    def test_all_styles(self) -> None:
        assert sep_alignment("---") == "-"
        assert sep_alignment(":--") == ":-"
        assert sep_alignment("--:") == "-:"
        assert sep_alignment(":-:") == ":-:"


class TestFormatting:
    def test_fmt_cell_pads_left(self) -> None:
        assert fmt_cell("ab", 4) == " ab   "

    def test_fmt_sep_styles(self) -> None:
        assert fmt_sep(3, "-") == "-----"
        assert fmt_sep(3, ":-") == ":----"
        assert fmt_sep(3, "-:") == "----:"
        assert fmt_sep(3, ":-:") == ":---:"


class TestAlignTable:
    def test_no_rows(self) -> None:
        assert align_table([]) == []

    def test_empty_parsed_row(self) -> None:
        # Строка "" даёт одну пустую ячейку, а не пустой список.
        assert align_table(["|"]) == ["|"]

    def test_without_separator_returns_input(self) -> None:
        rows = ["| a | b |", "| 1 | 2 |"]
        assert align_table(rows) == rows

    def test_narrow_table_is_padded(self) -> None:
        result = align_table(["| A | B |", "|---|---|", "| 1 | 2 |"])
        assert result == ["| A   | B   |", "|-----|-----|", "| 1   | 2   |"]

    def test_alignment_styles_reach_separator(self) -> None:
        rows = ["| A | B | C | D |", "|:---|:---:|---:|---|", "| 1 | 2 | 3 | 4 |"]
        result = align_table(rows)
        assert result[1] == "|:----|:---:|----:|-----|"

    def test_extra_cells_beyond_header_are_ignored(self) -> None:
        rows = ["| a | b |", "|---|---|", "| 1 | 2 | 3 |"]
        result = align_table(rows)
        assert len(result) == 3

    def test_short_row_is_padded_with_empty_cells(self) -> None:
        rows = ["| aaa | bbb |", "|---|---|", "| 1 |"]
        result = align_table(rows)
        assert result[2] == "| 1   |     |"

    def test_wide_table_keeps_one_line_per_row(self) -> None:
        # Широкая таблица остаётся в одну строку на строку: перенесённую ячейку
        # pandoc читает как ещё одну строку таблицы, а перенесённую шапку — как
        # отмену таблицы целиком.
        long = "слова " * 40
        rows = [f"| {long} | {long} |", "|---|---|", f"| остаток | {long} |"]
        result = align_table(rows)
        assert len(result) == len(rows)
        assert all(len(line) > 120 for line in result)


class TestGluedBlock:
    def test_first_line_is_never_glued(self) -> None:
        assert glued_block(["| a | b |"], 0) is False

    def test_blank_previous_line(self) -> None:
        assert glued_block(["текст", "", "| a | b |", "|---|---|"], 2) is False

    def test_table_after_paragraph(self) -> None:
        assert glued_block(["Пример:", "| a | b |", "|---|---|"], 1) is True

    def test_table_after_list_item(self) -> None:
        assert glued_block(["- пункт", "| a | b |", "|---|---|"], 1) is True

    def test_table_row_inside_table(self) -> None:
        # Строка таблицы не отделяется: разделитель уже позади.
        lines = ["| a | b |", "|---|---|", "| 1 | 2 |"]
        assert glued_block(lines, 2) is False

    def test_pipe_row_without_separator(self) -> None:
        assert glued_block(["текст", "| a | b |", "| 1 | 2 |"], 1) is False

    def test_last_line_without_separator(self) -> None:
        assert glued_block(["текст", "| a | b |"], 1) is False

    def test_bullet_after_paragraph(self) -> None:
        assert glued_block(["**Достоинства:**", "- простота"], 1) is True

    def test_numbered_first_item_after_paragraph(self) -> None:
        assert glued_block(["План лекции:", "1. первый"], 1) is True

    def test_item_after_item(self) -> None:
        assert glued_block(["- один", "1. второй"], 1) is False

    def test_middle_item_after_continuation(self) -> None:
        # «2.» после продолжения пункта — текст пункта, а не начало списка.
        lines = ["1. первый", "   продолжение", "2. второй"]
        assert glued_block(lines, 2) is False


class TestSeparateBlocks:
    def test_inserts_blank_line_before_table(self) -> None:
        lines = ["Пример:", "| a | b |", "|---|---|"]
        assert separate_blocks(lines) == ["Пример:", "", "| a | b |", "|---|---|"]

    def test_inserts_blank_line_before_list(self) -> None:
        lines = ["План лекции:", "1. первый", "2. второй"]
        assert separate_blocks(lines) == ["План лекции:", "", "1. первый", "2. второй"]

    def test_keeps_fenced_code_block(self) -> None:
        # Внутри забора `|` — не разметка: пустая строка разорвала бы пример.
        lines = ["Пример:", "", "```", "| a | b |", "|---|---|", "```"]
        assert separate_blocks(lines) == lines

    def test_keeps_display_math(self) -> None:
        lines = ["Формула:", "", "$$", "- не пункт списка", "$$"]
        assert separate_blocks(lines) == lines

    def test_single_line_math_does_not_swallow_following_lines(self) -> None:
        lines = ["Формула:", "$$x = 1$$", "План:", "1. пункт"]
        result = separate_blocks(lines)
        assert result == ["Формула:", "$$x = 1$$", "План:", "", "1. пункт"]


class TestFormatLines:
    def test_separates_and_aligns_markdown(self) -> None:
        lines = ["Пример:", "| A | B |", "|---|---|", "| 1 | 2 |"]
        result, tables = format_lines(lines)
        assert tables == [(2, 4)]
        assert result == ["Пример:", "", "| A   | B   |", "|-----|-----|", "| 1   | 2   |"]

    def test_python_comments_are_not_separated(self) -> None:
        lines = ["x = 1", "# | A | B |", "# |---|---|", "# | 1 | 2 |"]
        result, tables = format_lines(lines, "#")
        assert tables == [(1, 3)]
        assert result == ["x = 1", "# | A   | B   |", "# |-----|-----|", "# | 1   | 2   |"]


class TestFindTables:
    def test_finds_one_table(self) -> None:
        lines = ["text", "| a | b |", "|---|---|", "| 1 | 2 |", "text"]
        assert find_tables(lines) == [(1, 3)]

    def test_pipes_without_separator_ignored(self) -> None:
        lines = ["| a | b |", "| 1 | 2 |"]
        assert find_tables(lines) == []

    def test_no_tables(self) -> None:
        assert find_tables(["text", "more"]) == []

    def test_two_tables(self) -> None:
        lines = ["| a |", "|---|", "text", "| b |", "|---|"]
        assert find_tables(lines) == [(0, 1), (3, 4)]

    def test_commented_tables_in_python(self) -> None:
        lines = ["x = 1", "# | a | b |", "# |---|---|", "# | 1 | 2 |"]
        assert find_tables(lines, "#") == [(1, 3)]


class TestReadTable:
    def test_prefix_and_payloads(self) -> None:
        lines = ["# | a | b |", "# |---|---|"]
        prefix, payloads = read_table(lines, 0, 1, "#")
        assert prefix == "# "
        assert payloads == ["| a | b |", "|---|---|"]


class TestProcessFile:
    def test_aligns_file_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        assert process_file(path, None) is True
        assert path.read_text(encoding="utf-8") == "| A   | B   |\n|-----|-----|\n| 1   | 2   |\n"

    def test_already_aligned_reports_and_keeps_file(
            self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        aligned = "| A   | B   |\n|-----|-----|\n| 1   | 2   |\n"
        path = tmp_path / "notes.md"
        path.write_text(aligned, encoding="utf-8")
        assert process_file(path, None) is True
        assert "already aligned" in capsys.readouterr().out

    def test_no_tables_reports(self, tmp_path: Path,
                               capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("просто текст\n", encoding="utf-8")
        assert process_file(path, None) is True
        assert "no tables found" in capsys.readouterr().out

    def test_read_error_returns_false(self, tmp_path: Path,
                                      capsys: pytest.CaptureFixture) -> None:
        assert process_file(tmp_path, None) is False
        assert "Error reading" in capsys.readouterr().err

    def test_quiet_suppresses_output(self, tmp_path: Path,
                                     capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        assert process_file(path, None, quiet=True) is True
        assert capsys.readouterr().out == ""

    def test_backup_is_created(self, tmp_path: Path,
                               capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        backup_dir = tmp_path / "backup"
        assert process_file(path, backup_dir) is True
        assert (backup_dir / "notes.md").exists()
        assert "Backup:" in capsys.readouterr().out

    def test_python_commented_table_is_aligned(self, tmp_path: Path) -> None:
        path = tmp_path / "nb.py"
        path.write_text("# | A | B |\n# |---|---|\n# | 1 | 2 |\n", encoding="utf-8")
        assert process_file(path, None) is True
        assert "# | A   | B   |" in path.read_text(encoding="utf-8")

    def test_two_tables_both_aligned(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n\ntext\n\n| C |\n|---|\n| 3 |\n",
                        encoding="utf-8")
        assert process_file(path, None) is True
        text = path.read_text(encoding="utf-8")
        assert "| A   | B   |" in text
        assert "| C   |" in text

    def test_separates_glued_blocks(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("Пример:\n| A | B |\n|---|---|\n| 1 | 2 |\n\nПлан:\n1. пункт\n",
                        encoding="utf-8")
        assert process_file(path, None) is True
        assert path.read_text(encoding="utf-8") == (
            "Пример:\n\n| A   | B   |\n|-----|-----|\n| 1   | 2   |\n\nПлан:\n\n1. пункт\n")


class TestProcessFileQuiet:
    """Тихий режим: конвейер зовёт модуль без вывода в stdout/stderr."""

    def test_quiet_read_error(self, tmp_path: Path,
                              capsys: pytest.CaptureFixture) -> None:
        assert process_file(tmp_path, None, quiet=True) is False
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_quiet_no_tables(self, tmp_path: Path,
                             capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("просто текст\n", encoding="utf-8")
        assert process_file(path, None, quiet=True) is True
        assert capsys.readouterr().out == ""

    def test_quiet_already_aligned(self, tmp_path: Path,
                                   capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A   | B   |\n|-----|-----|\n| 1   | 2   |\n", encoding="utf-8")
        assert process_file(path, None, quiet=True) is True
        assert capsys.readouterr().out == ""

    def test_quiet_backup(self, tmp_path: Path,
                          capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        assert process_file(path, tmp_path / "b", quiet=True) is True
        assert (tmp_path / "b" / "notes.md").exists()
        assert capsys.readouterr().out == ""


class TestMain:
    def test_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["absorbent-fmt-tables", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_aligns_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "notes.md"
        path.write_text("| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv",
                            ["absorbent-fmt-tables", "--no-backup", str(path)])
        main()
        assert "| A   | B   |" in path.read_text(encoding="utf-8")

    def test_dry_run_prints_without_writing(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture) -> None:
        original = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        path = tmp_path / "notes.md"
        path.write_text(original, encoding="utf-8")
        monkeypatch.setattr(sys, "argv",
                            ["absorbent-fmt-tables", "--dry-run", str(path)])
        main()
        out = capsys.readouterr().out
        assert "lines 1-3" in out
        assert path.read_text(encoding="utf-8") == original

    def test_dry_run_without_tables(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("просто текст\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv",
                            ["absorbent-fmt-tables", "--dry-run", str(path)])
        main()
        assert "no tables" in capsys.readouterr().out

    def test_missing_file_warns_and_continues(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
        monkeypatch.setattr(sys, "argv",
                            ["absorbent-fmt-tables", str(tmp_path / "нет.md")])
        main()
        assert "File not found" in capsys.readouterr().err

    def test_module_guard_runs_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["absorbent-fmt-tables", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(MODULE), run_name="__main__")
        assert exc.value.code == 0
