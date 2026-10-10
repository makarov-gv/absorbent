"""Версия установленного пакета и проверяемое происхождение сборки."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib


def get_version_info() -> dict[str, str | bool | None]:
    """Получить встроенную ревизию; Git допустим только для собственного исходного дерева."""
    package_version = version("absorbent")
    embedded = Path(__file__).with_name("_build_info.json")
    if embedded.is_file():
        info = json.loads(embedded.read_text(encoding="utf-8"))
        commit = info.get("commit")
        if (info.get("version") != package_version
                or (commit is not None and not isinstance(commit, str))
                or (commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit) is None)
                or not isinstance(info.get("dirty"), bool)):
            raise ValueError("Некорректные сведения о сборке Absorbent")
        return {"version": package_version, "commit": commit, "dirty": info["dirty"]}
    root = Path(__file__).resolve().parents[2]
    project = root / "pyproject.toml"
    if (shutil.which("git") and project.is_file()
            and tomllib.loads(project.read_text())["project"]["name"] == "absorbent"):
        git_root = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True)
        if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == root:
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
            return {"version": package_version, "commit": commit, "dirty": dirty}
    return {"version": package_version, "commit": None, "dirty": False}


def version_string() -> str:
    """Представить версию в стабильном формате командной строки."""
    info = get_version_info()
    return f"Absorbent {info['version']} (commit {info['commit'] or 'unknown'}; dirty {str(info['dirty']).lower()})"
