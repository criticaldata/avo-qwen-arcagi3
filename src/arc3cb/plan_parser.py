"""Parse model responses: python blocks, [PLAYBOOK] updates, and [ACTIONS] plans.

The [ACTIONS] grammar (documented for the model in prompts.py):

    [ACTIONS]
    ACTION3
    ACTION6 x=12 y=40 | expect: (12,40)=3; (13,40)=0
    ACTION1 | expect: levels=1; state=NOT_FINISHED
    [/ACTIONS]

One action per line. An optional ``| expect:`` clause carries assertions that the
frame AFTER the action must satisfy: cell assertions ``(x,y)=color`` (x=column,
y=row, color 0..15), ``levels=N`` (levels completed), and ``state=NAME``. The runner executes the
queue one action at a time, halts at the first failed assertion, and re-invokes
the model with the observed diff.

Parse errors are returned to the model verbatim, so messages are written to be
actionable for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VALID_STATES = {"NOT_PLAYED", "NOT_STARTED", "NOT_FINISHED", "WIN", "GAME_OVER"}


class PlanParseError(ValueError):
    pass


@dataclass
class Expectation:
    cells: list[tuple[int, int, int]] = field(default_factory=list)  # (x, y, color)
    levels: int | None = None
    state: str | None = None

    def is_empty(self) -> bool:
        return not self.cells and self.levels is None and self.state is None


@dataclass
class PlannedAction:
    name: str
    x: int | None = None
    y: int | None = None
    expect: Expectation | None = None
    raw: str = ""

    def describe(self) -> str:
        if self.x is not None:
            return f"{self.name} x={self.x} y={self.y}"
        return self.name


Block = tuple[str, str]  # (kind, content); kind in {"python", "actions", "playbook"}

_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
_TAG_RES = {
    "actions": re.compile(r"\[ACTIONS\](.*?)\[/ACTIONS\]", re.DOTALL | re.IGNORECASE),
    "playbook": re.compile(r"\[PLAYBOOK\](.*?)\[/PLAYBOOK\]", re.DOTALL | re.IGNORECASE),
}


def extract_blocks(text: str) -> list[Block]:
    """All protocol blocks in the response, in order of appearance."""
    found: list[tuple[int, str, str]] = []
    for m in _FENCE_RE.finditer(text):
        found.append((m.start(), "python", m.group(1)))
    for kind, rx in _TAG_RES.items():
        for m in rx.finditer(text):
            found.append((m.start(), kind, m.group(1).strip()))
    found.sort(key=lambda t: t[0])
    return [(kind, content) for _, kind, content in found]


_CELL_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)")
_LEVELS_RE = re.compile(r"\blevels\s*=\s*(-?\d+)", re.IGNORECASE)
_STATE_RE = re.compile(r"\bstate\s*=\s*([A-Za-z_]+)")
_COORD_RE = re.compile(r"\b([xy])\s*=\s*(\d+)\b", re.IGNORECASE)


def _parse_expectation(text: str, grid_size: int) -> Expectation:
    exp = Expectation()
    rest = text
    for m in _CELL_RE.finditer(text):
        x, y, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            raise PlanParseError(
                f"expectation cell ({x},{y}) out of range 0..{grid_size - 1}"
            )
        if not 0 <= v <= 15:
            raise PlanParseError(f"expectation color {v} out of range 0..15 in ({x},{y})={v}")
        exp.cells.append((x, y, v))
    rest = _CELL_RE.sub(" ", rest)
    m = _LEVELS_RE.search(rest)
    if m:
        exp.levels = int(m.group(1))
        rest = _LEVELS_RE.sub(" ", rest, count=1)
    m = _STATE_RE.search(rest)
    if m:
        state = m.group(1).upper()
        if state not in VALID_STATES:
            raise PlanParseError(
                f"unknown state {state!r}; valid states: {', '.join(sorted(VALID_STATES))}"
            )
        exp.state = state
        rest = _STATE_RE.sub(" ", rest, count=1)
    leftover = rest.replace("expect", "").replace(":", "").replace(";", "").replace(",", "")
    leftover = leftover.strip()
    if leftover:
        raise PlanParseError(
            f"unrecognized expectation text {leftover!r}; use (x,y)=color, levels=N, state=NAME"
        )
    return exp


def parse_plan(
    block: str,
    valid_actions: set[str],
    coord_actions: set[str] = frozenset({"ACTION6"}),
    grid_size: int = 64,
    max_len: int = 20,
) -> list[PlannedAction]:
    """Parse the body of one [ACTIONS] block into a validated queue."""
    plan: list[PlannedAction] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        head, expect_parts = parts[0], parts[1:]
        tokens = head.replace(",", " ").split()
        name = tokens[0].upper()
        if name not in valid_actions:
            raise PlanParseError(
                f"unknown action {tokens[0]!r} in line {raw_line.strip()!r}; "
                f"valid actions: {', '.join(sorted(valid_actions))}"
            )
        action = PlannedAction(name=name, raw=raw_line.strip())
        coords_text = " ".join(tokens[1:])
        coords = {k.lower(): int(v) for k, v in _COORD_RE.findall(coords_text)}
        stripped = _COORD_RE.sub(" ", coords_text).replace(",", " ").strip()
        if stripped:
            raise PlanParseError(
                f"could not parse {stripped!r} in line {raw_line.strip()!r}; "
                f"coordinate form is: {name} x=INT y=INT"
            )
        if name in coord_actions:
            if "x" not in coords or "y" not in coords:
                raise PlanParseError(
                    f"{name} requires coordinates, e.g. {name} x=12 y=40 "
                    f"(got line {raw_line.strip()!r})"
                )
            if not (0 <= coords["x"] < grid_size and 0 <= coords["y"] < grid_size):
                raise PlanParseError(
                    f"coordinates ({coords['x']},{coords['y']}) out of range "
                    f"0..{grid_size - 1} in line {raw_line.strip()!r}"
                )
            action.x, action.y = coords["x"], coords["y"]
        elif coords:
            raise PlanParseError(
                f"{name} takes no coordinates (line {raw_line.strip()!r})"
            )
        if expect_parts:
            exp_text = " ; ".join(expect_parts)
            if not re.match(r"\s*expect\b", exp_text, re.IGNORECASE):
                raise PlanParseError(
                    f"text after | must start with 'expect:' (line {raw_line.strip()!r})"
                )
            action.expect = _parse_expectation(exp_text, grid_size)
            if action.expect.is_empty():
                action.expect = None
        plan.append(action)
    if not plan:
        raise PlanParseError("[ACTIONS] block contains no actions")
    if len(plan) > max_len:
        raise PlanParseError(
            f"plan has {len(plan)} actions; maximum is {max_len} — send a shorter queue, "
            "you will be re-invoked after it executes"
        )
    for a, b in zip(plan, plan[1:], strict=False):
        if a.name == "RESET" and b.name == "RESET":
            raise PlanParseError(
                "plan contains two consecutive RESET actions; RESET discards the current "
                "attempt and must never be issued twice in a row"
            )
    return plan


def check_expectations(
    exp: Expectation, grid: list[list[int]], levels_completed: int, state: str
) -> list[str]:
    """Human/model-readable list of failed assertions (empty = all held)."""
    fails: list[str] = []
    for x, y, want in exp.cells:
        if y >= len(grid) or x >= len(grid[y]):
            fails.append(f"cell ({x},{y}) is outside the current {len(grid[0])}x{len(grid)} grid")
            continue
        got = grid[y][x]
        if got != want:
            fails.append(f"expected ({x},{y})={want}, observed ({x},{y})={got}")
    if exp.levels is not None and levels_completed != exp.levels:
        fails.append(
            f"expected levels={exp.levels}, observed levels_completed={levels_completed}"
        )
    if exp.state is not None and state != exp.state:
        fails.append(f"expected state={exp.state}, observed state={state}")
    return fails
