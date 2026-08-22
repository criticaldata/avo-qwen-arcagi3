"""Configuration: .env loading, per-model configs, budgets.

Precedence: built-in defaults < configs/*.yaml < CLI flags. API keys come only
from the environment (or .env) — never from YAML, never from the command line,
so they cannot end up in shell history or run artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .transport import ModelConfig


def load_env_file(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines; '#' comments; no expansion).

    Values already present in the environment win.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text())
    return data or {}


@dataclass
class Budgets:
    """Hard caps; the runner kills the run cleanly at any of them."""

    max_actions_per_level: int = 1000
    max_actions_per_game: int = 4000
    max_tokens_per_game: int = 30_000_000
    max_usd_per_game: float = 25.0
    max_wall_clock_s: int = 0  # 0 = unlimited


@dataclass
class RunSettings:
    game: str
    mode: str  # "online" | "local" | "mock"
    model: ModelConfig
    budgets: Budgets = field(default_factory=Budgets)
    context_reset_input_tokens: int = 90_000
    escalation_after_actions: int = 300
    escalation_self_resets: int = 2
    plan_max_len: int = 20
    containment_venv: Path = Path(".containment-venv")
    sandbox_timeout_s: int = 120
    sandbox_memory_mb: int = 2048
    sandbox_max_output_chars: int = 20_000
    full_grid_every: int = 1
    runs_dir: Path = Path("runs")
    configs_dir: Path = Path("configs")
    tag: str = ""


def build_model_config(model_id: str, configs_dir: str | Path = "configs") -> ModelConfig:
    """Model config from configs/models.yaml, keyed by served model id.

    A model absent from the file still runs, with defaults and zero prices (the
    cost meter then reports 0.0 and the runner warns loudly).
    """
    data = load_yaml(Path(configs_dir) / "models.yaml")
    defaults: dict = data.get("defaults", {})
    entry: dict = (data.get("models") or {}).get(model_id, {})
    merged = {**defaults, **entry, "id": model_id}
    known = {f.name for f in fields(ModelConfig)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(
            f"unknown keys in models.yaml for {model_id}: {', '.join(sorted(unknown))}"
        )
    return ModelConfig(**merged)


def build_budgets(configs_dir: str | Path = "configs", **overrides) -> Budgets:
    data = load_yaml(Path(configs_dir) / "budgets.yaml")
    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    known = {f.name for f in fields(Budgets)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"unknown keys in budgets.yaml: {', '.join(sorted(unknown))}")
    return Budgets(**merged)


def load_human_baselines(configs_dir: str | Path = "configs") -> dict[str, list[int]]:
    """Official per-level human action counts, keyed by game id prefix."""
    data = load_yaml(Path(configs_dir) / "human_baselines.yaml")
    return {str(k): [int(x) for x in v] for k, v in (data.get("games") or {}).items()}


def load_forbidden_modules(configs_dir: str | Path = "configs") -> list[str]:
    from .tools import DEFAULT_FORBIDDEN_MODULES

    data = load_yaml(Path(configs_dir) / "containment.yaml")
    extra = [str(m) for m in (data.get("forbidden_modules") or [])]
    return sorted(set(DEFAULT_FORBIDDEN_MODULES) | set(extra))
