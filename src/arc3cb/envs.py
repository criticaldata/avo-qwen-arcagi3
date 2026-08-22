"""Game environments behind one interface.

- ``ArcOnlineEnv`` (in arcclient.py): the official ARC-AGI-3 HTTP API, used for
  competition-mode scorecards.
- ``MockEnv`` (here): a small deterministic built-in game so the whole loop —
  plan queue, expectations, diffs, resets, scoring — can run end to end in CI
  with no keys and no network. It is NOT an ARC game and is never scored for
  reporting.

The runner only sees this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Frame:
    grid: list[list[int]]  # last (settled) grid of the burst
    state: str  # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int
    win_levels: int = 0
    grids: list[list[list[int]]] = field(default_factory=list)  # full burst, if any
    available: list[str] = field(default_factory=list)  # currently valid action names
    guid: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def burst(self) -> int:
        return max(1, len(self.grids))


class GameEnv(Protocol):
    game_id: str

    def available_actions(self) -> set[str]: ...

    def reset(self) -> Frame: ...

    def act(self, name: str, x: int | None = None, y: int | None = None) -> Frame: ...

    def close(self) -> dict: ...


class MockEnv:
    """Deterministic 2-level 'reach the target' game for tests and smoke runs.

    Mechanics (intentionally simple but exercising every harness feature):
    - 64x64 grid, color 5 border, agent is color 3, target is color 2.
    - ACTION1/2/3/4 move the agent up/down/left/right by one cell.
    - ACTION5 is a no-op. ACTION6 x= y= teleports the agent there if in bounds.
    - Walking into the border (or teleporting onto it) is GAME_OVER; RESET
      restarts the current level.
    - Reaching the target completes the level (levels_completed += 1); two levels = WIN.
    """

    SIZE = 64
    STARTS = [(10, 10), (50, 12)]
    TARGETS = [(20, 10), (12, 44)]
    LEVELS = 2

    def __init__(self, game_id: str = "mockgame"):
        self.game_id = game_id
        self.levels = 0
        self.state = "NOT_PLAYED"
        self.pos = self.STARTS[0]
        self._frames = 0

    def available_actions(self) -> set[str]:
        return {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"}

    def _grid(self) -> list[list[int]]:
        g = [[0] * self.SIZE for _ in range(self.SIZE)]
        for i in range(self.SIZE):
            g[0][i] = g[self.SIZE - 1][i] = g[i][0] = g[i][self.SIZE - 1] = 5
        if self.levels < self.LEVELS:
            tx, ty = self.TARGETS[self.levels]
            g[ty][tx] = 2
        if self.state != "GAME_OVER":
            x, y = self.pos
            g[y][x] = 3
        return g

    def _frame(self) -> Frame:
        self._frames += 1
        return Frame(
            grid=self._grid(),
            state=self.state,
            levels_completed=self.levels,
            win_levels=self.LEVELS,
            available=sorted(self.available_actions()),
            guid=f"mock-{self._frames}",
        )

    def reset(self) -> Frame:
        self.state = "NOT_FINISHED"
        self.pos = self.STARTS[min(self.levels, self.LEVELS - 1)]
        return self._frame()

    def act(self, name: str, x: int | None = None, y: int | None = None) -> Frame:
        if name == "RESET":
            return self.reset()
        if self.state != "NOT_FINISHED":
            return self._frame()  # ignored, same as the real API refusing input
        moves = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
        px, py = self.pos
        if name in moves:
            dx, dy = moves[name]
            px, py = px + dx, py + dy
        elif name == "ACTION6":
            if x is None or y is None:
                raise ValueError("ACTION6 requires coordinates")
            px, py = x, y
        elif name != "ACTION5":
            raise ValueError(f"unknown action {name!r}")
        if px <= 0 or px >= self.SIZE - 1 or py <= 0 or py >= self.SIZE - 1:
            self.state = "GAME_OVER"
            return self._frame()
        self.pos = (px, py)
        if self.levels < self.LEVELS and self.pos == self.TARGETS[self.levels]:
            self.levels += 1
            if self.levels >= self.LEVELS:
                self.state = "WIN"
            else:
                self.pos = self.STARTS[self.levels]
        return self._frame()

    def close(self) -> dict:
        return {"game_id": self.game_id, "levels_completed": self.levels, "state": self.state}
