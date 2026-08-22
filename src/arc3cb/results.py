"""Regenerate results tables from run artifacts (metrics.json + usage.jsonl).

Every number in docs/results.md must be reproducible by pointing this module at
the runs directory; nothing is hand-entered.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_human_baselines
from .scoring import LevelResult, game_rhae


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        try:
            m = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m["run_dir"] = str(metrics_path.parent)
        runs.append(m)
    return runs


def recompute_rhae(run: dict, baselines_map: dict[str, list[int]]) -> float | None:
    prefix = str(run.get("game_id", "")).split("-")[0]
    baselines = run.get("human_baselines") or baselines_map.get(prefix)
    if not baselines:
        return None
    levels = [
        LevelResult(r["level"], r["completed"], r["agent_actions"])
        for r in run.get("level_results", [])
    ]
    try:
        return game_rhae(levels, baselines)
    except ValueError:
        return None


def render_results(runs_dir: Path, configs_dir: Path = Path("configs")) -> str:
    runs = load_runs(runs_dir)
    if not runs:
        return f"no runs with metrics.json under {runs_dir}"
    baselines_map = load_human_baselines(configs_dir)
    header = (
        "| game | model | mode | RHAE | levels | actions | tokens | cost USD | stop | run dir |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        rhae = recompute_rhae(run, baselines_map)
        run["_rhae"] = rhae
        by_model.setdefault(str(run.get("model")), []).append(run)
        lines.append(
            "| {game} | {model} | {mode} | {rhae} | {lv}/{win} | {actions} | {tokens} "
            "| {cost:.2f} | {stop} | {rd} |".format(
                game=run.get("game_id"),
                model=run.get("model"),
                mode=run.get("mode"),
                rhae=f"{rhae:.2f}" if rhae is not None else "—",
                lv=run.get("levels_completed"),
                win=run.get("win_levels"),
                actions=run.get("actions"),
                tokens=run.get("total_tokens"),
                cost=float(run.get("cost_usd") or 0),
                stop=run.get("stop_reason"),
                rd=Path(run["run_dir"]).name,
            )
        )
    lines.append("")
    lines.append("per-model summary (mock runs excluded from RHAE means):")
    for model, model_runs in sorted(by_model.items()):
        scored = [r["_rhae"] for r in model_runs if r["_rhae"] is not None and r.get("mode") != "mock"]
        total_cost = sum(float(r.get("cost_usd") or 0) for r in model_runs)
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in model_runs)
        total_levels = sum(int(r.get("levels_completed") or 0) for r in model_runs)
        lines.append(
            f"  {model}: {len(model_runs)} runs, mean RHAE over scored games "
            f"{f'{sum(scored) / len(scored):.2f}' if scored else '—'}, "
            f"levels {total_levels}, tokens {total_tokens}, cost ${total_cost:.2f}"
        )
    return "\n".join(lines)
