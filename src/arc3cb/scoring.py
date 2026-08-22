"""Local RHAE computation for instant feedback during runs.

Implements the scoring the official scorecards actually compute (the shipped
``arc-agi`` calculator, v0.9.9), which the docs at docs.arcprize.org/methodology
describe. RHAE = Relative Human Action Efficiency:

- per COMPLETED level: ``score = min((human_baseline / ai_actions)^2 * 100, 115)``
  (the 1.15x-baseline cap is applied post-square, at 115 on the 0-100 scale);
- each level is weighted by its 1-based index within the game; uncompleted or
  unreached levels score 0 but keep their weight in the denominator;
- the game score is additionally capped at the completed-weight fraction:
  finishing only levels 1-4 of 5 caps the game at (1+2+3+4)/(1+2+3+4+5);
- the campaign score is the plain mean over game scores (0-100 scale).

Action counting: every in-play RESET counts as an action; only the initial
play-creating RESET is uncharged — callers must count accordingly.

Official competition-mode scorecards remain the source of truth for any number
reported publicly; this module exists so a run can print its provisional RHAE
the moment it ends.
"""

from __future__ import annotations

from dataclasses import dataclass

LEVEL_SCORE_CAP = 115.0


@dataclass
class LevelResult:
    level: int  # 1-based index within the game
    completed: bool
    agent_actions: int  # actions charged to this level (in-play RESETs included)


def level_score(human_actions: int, agent_actions: int) -> float:
    if human_actions <= 0:
        raise ValueError(f"human baseline must be positive, got {human_actions}")
    if agent_actions <= 0:
        raise ValueError(
            f"agent actions must be positive for a completed level, got {agent_actions}"
        )
    return min((human_actions / agent_actions) ** 2 * 100.0, LEVEL_SCORE_CAP)


def game_rhae(results: list[LevelResult], human_baselines: list[int]) -> float:
    """Game score on the official 0-100 scale.

    ``human_baselines[i]`` is the official human action count for level i+1;
    every level the game defines must be represented by the baselines list.
    """
    if not human_baselines:
        raise ValueError("no human baselines for game")
    by_level = {r.level: r for r in results}
    num = 0.0
    total_weight = 0.0
    completed_weight = 0.0
    for i, human in enumerate(human_baselines, start=1):
        weight = float(i)
        total_weight += weight
        r = by_level.get(i)
        if r is not None and r.completed:
            completed_weight += weight
            num += weight * level_score(human, r.agent_actions)
    score = num / total_weight
    return min(score, completed_weight / total_weight * 100.0)


def campaign_rhae(game_scores: list[float]) -> float:
    """Mean over games (each already 0-100), as reported on the leaderboard."""
    if not game_scores:
        return 0.0
    return sum(game_scores) / len(game_scores)
