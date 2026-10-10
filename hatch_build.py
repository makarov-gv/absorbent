"""Встроенные сведения о версии и исходной ревизии сборки."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Добавить происхождение пакета без изменения исходного дерева."""

    def initialize(self, version, build_data):
        root = Path(self.root)
        package_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        supplied = os.environ.get("ABSORBENT_BUILD_COMMIT") or None
        if supplied is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", supplied):
            raise ValueError("ABSORBENT_BUILD_COMMIT должен содержать полный SHA")
        supplied = supplied.lower() if supplied else None
        git_root = (subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        ) if shutil.which("git") else None)
        if (git_root is not None and git_root.returncode == 0
                and Path(git_root.stdout.strip()).resolve() == root.resolve()):
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            if supplied is not None and supplied != commit:
                raise ValueError("Git HEAD и ABSORBENT_BUILD_COMMIT не совпадают")
            dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
        else:
            embedded = root / "src/absorbent/_build_info.json"
            info = json.loads(embedded.read_text()) if embedded.exists() else {}
            commit = supplied or info.get("commit")
            dirty = info.get("dirty", False)
            if info and info.get("version") != package_version:
                raise ValueError("Версия sdist не совпадает с pyproject.toml")
            if supplied and info.get("commit") and supplied != info["commit"]:
                raise ValueError("Ревизия sdist и ABSORBENT_BUILD_COMMIT не совпадают")
        if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Некорректная встроенная ревизия")
        if not isinstance(dirty, bool):
            raise ValueError("Некорректный признак dirty")
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump({"version": package_version, "commit": commit, "dirty": dirty}, handle)
        self._metadata_file = handle.name
        target = "absorbent/_build_info.json" if self.target_name == "wheel" else "src/absorbent/_build_info.json"
        build_data["force_include"][handle.name] = target

    def finalize(self, version, build_data, artifact_path):
        Path(self._metadata_file).unlink(missing_ok=True)
