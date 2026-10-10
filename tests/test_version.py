"""Проверки происхождения установленного пакета и честного отсутствия ревизии."""

import json
from pathlib import Path
import subprocess

import pytest

from absorbent import version as provenance
from absorbent.cli import main


@pytest.fixture
def source(tmp_path, monkeypatch):
    package = tmp_path / "src" / "absorbent"
    package.mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(package / "version.py"))
    monkeypatch.setattr(provenance, "version", lambda name: "0.1.1")
    return tmp_path, package


def test_embedded_revision_does_not_consult_git(source, monkeypatch):
    root, package = source
    info = {"version": "0.1.1", "commit": "a" * 40, "dirty": False}
    (package / "_build_info.json").write_text(json.dumps(info))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Git при установленном wheel"))
    assert provenance.get_version_info() == info
    assert provenance.version_string() == f"Absorbent 0.1.1 (commit {'a' * 40}; dirty false)"


@pytest.mark.parametrize("changes", [
    {"version": "9.9.9"}, {"commit": 12}, {"commit": "short"}, {"dirty": "false"},
])
def test_corrupt_provenance_rejected(source, changes):
    root, package = source
    info = {"version": "0.1.1", "commit": "a" * 40, "dirty": False} | changes
    (package / "_build_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="сведения"):
        provenance.get_version_info()


def test_unknown_revision_is_not_zero_or_development_head(source):
    assert provenance.get_version_info() == {"version": "0.1.1", "commit": None, "dirty": False}
    assert "commit unknown" in provenance.version_string()


def test_embedded_unknown_revision(source):
    root, package = source
    info = {"version": "0.1.1", "commit": None, "dirty": False}
    (package / "_build_info.json").write_text(json.dumps(info))
    assert provenance.get_version_info() == info


@pytest.mark.parametrize("case", ["no_git", "foreign_project", "not_repository", "parent_repository"])
def test_source_fallback_does_not_borrow_foreign_head(source, monkeypatch, case):
    root, package = source
    name = "other" if case == "foreign_project" else "absorbent"
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    if case == "no_git":
        monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    elif case == "parent_repository":
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, str(root.parent)))
    assert provenance.get_version_info()["commit"] is None


def test_editable_reports_own_git_changes(source):
    root, package = source
    (root / "pyproject.toml").write_text('[project]\nname = "absorbent"\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    message = root / "message"
    message.write_text("Исходное дерево\n")
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-q", "-F", str(message)], check=True)
    message.unlink()
    info = provenance.get_version_info()
    assert len(info["commit"]) == 40 and info["dirty"] is False
    (root / "pyproject.toml").write_text('[project]\nname = "absorbent"\nversion = "0.1.1"\n')
    assert provenance.get_version_info()["dirty"] is True


def test_version_exits_before_configuration_and_pipeline(monkeypatch, capsys):
    monkeypatch.setattr("absorbent.cli.load_config", lambda *a: pytest.fail("Конфигурация загружена"))
    with pytest.raises(SystemExit) as result:
        main(["--version", "--config", "/does/not/exist"])
    assert result.value.code == 0
    assert capsys.readouterr().out.startswith("Absorbent ")
