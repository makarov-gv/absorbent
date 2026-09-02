"""Тесты точек входа `python -m absorbent` и `python cli.py`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

SRC_CLI = Path(__file__).resolve().parents[1] / "src" / "absorbent" / "cli.py"
SRC_MAIN = Path(__file__).resolve().parents[1] / "src" / "absorbent" / "__main__.py"


class TestEntryPoints:
    def test_cli_py_as_main_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`if __name__ == '__main__'` в cli.py: без аргументов -> exit 2."""
        monkeypatch.setattr(sys, "argv", ["absorbent"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SRC_CLI), run_name="__main__")
        assert exc.value.code == 2

    def test_main_py_as_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`python -m absorbent` без аргументов -> exit 2 (argparse)."""
        monkeypatch.setattr(sys, "argv", ["absorbent"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SRC_MAIN), run_name="__main__")
        assert exc.value.code == 2

    def test_main_py_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["absorbent", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(SRC_MAIN), run_name="__main__")
        assert exc.value.code == 0

    def test_main_py_imported_not_as_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ветка `__name__ != '__main__'`: импорт файла не запускает CLI."""
        result = runpy.run_path(str(SRC_MAIN), run_name="absorbent.__main__")
        assert "main" in result  # модуль загружен, но без запуска конвейера.
