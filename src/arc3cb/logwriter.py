"""Frame log: serialize game frames to a structured, machine-parseable log.txt.

Conventions used everywhere in this project:
- A grid is ``list[list[int]]`` indexed ``grid[y][x]`` (row-major), origin at the
  top-left. ``x`` is the column (0..width-1), ``y`` is the row (0..height-1).
- Cell values are ARC colors 0..15, rendered as single uppercase hex digits, so a
  64x64 frame is 64 lines of 64 characters.
- Coordinates in diffs and expectations are written ``(x,y)``.

Entry layout (one blank-line-separated block per executed action)::

    [FRAME 12 | action ACTION6 x=3 y=7 | levels 1/7 | state NOT_FINISHED]
    [AVAILABLE] ACTION1 ACTION2 ACTION6 RESET
    [DIFF] (3,7) 0->5; (4,7) 5->0
    [GRID]
    <hex rows>
    [/GRID]

Between frame entries the runner records accepted plans as ``[PLAN n]`` blocks
and free-form ``[MARK]`` lines; both are skipped by frame parsing.

The log is both the agent's long-term memory (read back through the sandboxed
Python tool and the ``gamelog`` workspace helper) and the record retrodiction
replays hypotheses against, so it must be losslessly parseable: ``parse_log``
reconstructs every frame either from an embedded ``[GRID]`` block or by
applying ``[DIFF]`` entries to the previous frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEX = "0123456789ABCDEF"

CellDiff = tuple[int, int, int, int]  # (x, y, old, new)


class LogFormatError(ValueError):
    pass


def render_grid(grid: list[list[int]]) -> str:
    """Render a grid as one hex character per cell, one line per row."""
    lines = []
    for row in grid:
        try:
            lines.append("".join(HEX[v] for v in row))
        except (IndexError, TypeError) as e:
            raise LogFormatError(f"cell value outside 0..15 in row {len(lines)}: {row!r}") from e
    return "\n".join(lines)


def parse_grid(text: str) -> list[list[int]]:
    grid = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            grid.append([HEX.index(c) for c in line.upper()])
        except ValueError as e:
            raise LogFormatError(f"non-hex character in grid line {line!r}") from e
    return grid


def diff_grids(old: list[list[int]], new: list[list[int]]) -> list[CellDiff]:
    """Cells that changed between two same-shaped grids, as (x, y, old, new)."""
    if len(old) != len(new) or any(len(a) != len(b) for a, b in zip(old, new, strict=False)):
        # Shape change (level transition etc.): treat every cell of `new` as changed.
        return [(x, y, -1, v) for y, row in enumerate(new) for x, v in enumerate(row)]
    out: list[CellDiff] = []
    for y, (ra, rb) in enumerate(zip(old, new, strict=False)):
        if ra == rb:
            continue
        for x, (a, b) in enumerate(zip(ra, rb, strict=False)):
            if a != b:
                out.append((x, y, a, b))
    return out


def format_diff(diffs: list[CellDiff], max_cells: int = 120) -> str:
    if not diffs:
        return "[DIFF] none"
    if len(diffs) > max_cells:
        xs = [d[0] for d in diffs]
        ys = [d[1] for d in diffs]
        return (
            f"[DIFF] {len(diffs)} cells changed in bbox "
            f"({min(xs)},{min(ys)})-({max(xs)},{max(ys)})"
        )
    body = "; ".join(f"({x},{y}) {a}->{b}" for x, y, a, b in diffs)
    return f"[DIFF] {body}"


@dataclass
class LogEntry:
    frame_index: int
    action: str  # e.g. "RESET" or "ACTION6 x=3 y=7"
    levels_completed: int
    win_levels: int
    state: str
    grid: list[list[int]]
    available: list[str] = field(default_factory=list)
    diffs: list[CellDiff] = field(default_factory=list)
    note: str = ""


_HEADER_RE = re.compile(
    r"^\[FRAME (?P<idx>\d+) \| action (?P<action>[^|]+) \| levels (?P<lv>\d+)/(?P<win>\d+)"
    r" \| state (?P<state>[A-Z_]+)(?P<extra>[^\]]*)\]$"
)
_DIFF_CELL_RE = re.compile(r"\((\d+),(\d+)\) (-?\d+)->(\d+)")


class LogWriter:
    """Append-only writer for log.txt.

    ``full_grid_every=1`` (default) embeds the full grid in every entry, which is
    what the spec asks for and what makes retrodiction tooling trivial. A sparser
    setting still keeps the log lossless because grids are forced whenever the
    diff could not be listed in full.
    """

    def __init__(self, path: str | Path, full_grid_every: int = 1, max_diff_cells: int = 120):
        self.path = Path(path)
        self.full_grid_every = max(1, int(full_grid_every))
        self.max_diff_cells = max_diff_cells
        self._prev_grid: list[list[int]] | None = None
        self._count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def append(
        self,
        frame_index: int,
        action: str,
        grid: list[list[int]],
        levels_completed: int,
        win_levels: int,
        state: str,
        available: list[str] | None = None,
        note: str = "",
        burst: int = 1,
    ) -> list[CellDiff]:
        """Append one frame; returns the cell diff against the previous frame."""
        diffs = [] if self._prev_grid is None else diff_grids(self._prev_grid, grid)
        extra = f" | burst {burst}" if burst > 1 else ""
        lines = [
            f"[FRAME {frame_index} | action {action} | levels {levels_completed}/{win_levels}"
            f" | state {state}{extra}]"
        ]
        if available:
            lines.append("[AVAILABLE] " + " ".join(available))
        if note:
            lines.append(f"[NOTE] {note}")
        if self._prev_grid is not None:
            lines.append(format_diff(diffs, self.max_diff_cells))
        shape_changed = self._prev_grid is not None and (
            len(self._prev_grid) != len(grid)
            or any(len(a) != len(b) for a, b in zip(self._prev_grid, grid, strict=False))
        )
        listed_all = self._prev_grid is not None and len(diffs) <= self.max_diff_cells
        must_embed = (
            self._prev_grid is None
            or shape_changed
            or not listed_all
            or self._count % self.full_grid_every == 0
        )
        if must_embed:
            lines.append("[GRID]")
            lines.append(render_grid(grid))
            lines.append("[/GRID]")
        with self.path.open("a") as f:
            f.write("\n".join(lines) + "\n\n")
        self._prev_grid = [row[:] for row in grid]
        self._count += 1
        return diffs

    def append_plan(self, invocation: int, plan_text: str) -> None:
        """Record an accepted plan so the log tells the whole story of the run."""
        with self.path.open("a") as f:
            f.write(f"[PLAN {invocation}]\n{plan_text.strip()}\n[/PLAN]\n\n")

    def mark(self, text: str) -> None:
        """Write a free-form marker line (level transitions, resets, escalations)."""
        with self.path.open("a") as f:
            f.write(f"[MARK] {text}\n\n")


def parse_log(path: str | Path) -> list[LogEntry]:
    """Reconstruct every logged frame, applying diffs where grids were elided."""
    entries: list[LogEntry] = []
    prev: list[list[int]] | None = None
    blocks = [b for b in Path(path).read_text().split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.splitlines()
        if lines[0].startswith(("[MARK]", "[PLAN ")):
            continue
        m = _HEADER_RE.match(lines[0])
        if not m:
            raise LogFormatError(f"bad header line: {lines[0]!r}")
        i = 1
        available: list[str] = []
        if i < len(lines) and lines[i].startswith("[AVAILABLE] "):
            available = lines[i][len("[AVAILABLE] "):].split()
            i += 1
        note = ""
        if i < len(lines) and lines[i].startswith("[NOTE] "):
            note = lines[i][len("[NOTE] "):]
            i += 1
        diffs: list[CellDiff] = []
        diff_summarized = False
        if i < len(lines) and lines[i].startswith("[DIFF]"):
            diff_summarized = "cells changed in bbox" in lines[i]
            diffs = [
                (int(x), int(y), int(a), int(b))
                for x, y, a, b in _DIFF_CELL_RE.findall(lines[i])
            ]
            i += 1
        if i < len(lines) and lines[i] == "[GRID]":
            try:
                end = lines.index("[/GRID]", i)
            except ValueError as e:
                raise LogFormatError("unterminated [GRID] block") from e
            grid = parse_grid("\n".join(lines[i + 1:end]))
        else:
            if prev is None:
                raise LogFormatError("first entry has no [GRID] block")
            if diff_summarized:
                raise LogFormatError("elided grid with summarized diff — log is lossy")
            grid = [row[:] for row in prev]
            for x, y, _old, new in diffs:
                grid[y][x] = new
        entries.append(
            LogEntry(
                frame_index=int(m.group("idx")),
                action=m.group("action").strip(),
                levels_completed=int(m.group("lv")),
                win_levels=int(m.group("win")),
                state=m.group("state"),
                grid=grid,
                available=available,
                diffs=diffs,
                note=note,
            )
        )
        prev = grid
    return entries
