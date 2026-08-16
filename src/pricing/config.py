"""Configuration loading.

The whole pipeline reads its settings from a single YAML file so that a run can
be reproduced from that file plus the raw data. Paths in the config are relative
to the project root, and are resolved to absolute paths on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root, found by walking up from this file."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """A thin wrapper around the parsed YAML with path resolution helpers."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, *keys: str) -> Path:
        """Resolve a config value that holds a relative path."""
        node: Any = self.raw
        for key in keys:
            node = node[key]
        return (self.root / str(node)).resolve()

    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes to."""
        for keys in (
            ("data", "raw_dir"),
            ("data", "interim_dir"),
            ("data", "processed_dir"),
            ("reporting", "figures_dir"),
            ("reporting", "tables_dir"),
            ("reporting", "models_dir"),
        ):
            self.path(*keys).mkdir(parents=True, exist_ok=True)

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])


def load_config(path: str | Path | None = None) -> Config:
    """Load the pipeline configuration from YAML."""
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a mapping")
    return Config(raw=raw, root=project_root())
