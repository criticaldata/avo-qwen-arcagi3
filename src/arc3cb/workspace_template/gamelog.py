"""Standalone reader for the harness's log.txt, for use by agent-authored code.

This file is copied into every run workspace so sandboxed python can simply
``import gamelog`` — it depends only on numpy and must never import the harness
or any game engine. It parses the same format arc3cb.logwriter writes (the
harness test suite keeps the two in sync).

Conventions: grids are numpy int arrays indexed grid[y][x]; coordinates are
(x, y) = (column, row); colors are ints 0..15.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque

import numpy as np

HEX = "0123456789ABCDEF"

_HEADER_RE = re.compile(
    r"^\[FRAME (?P<idx>\d+) \| action (?P<action>[^|]+) \| levels (?P<lv>\d+)/(?P<win>\d+)"
    r" \| state (?P<state>[A-Z_]+)(?P<extra>[^\]]*)\]$"
)
_DIFF_CELL_RE = re.compile(r"\((\d+),(\d+)\) (-?\d+)->(\d+)")
_COORD_RE = re.compile(r"\b([xy])=(\d+)\b")


class Step:
    """One logged frame."""

    def __init__(self, i, action, grid, levels_completed, win_levels, state, available, diff, note):
        self.i = i
        self.step = i
        self.action = action  # full string, e.g. "ACTION6 x=3 y=7"
        self.grid = grid  # numpy (H, W) int array
        self.board = grid
        self.levels_completed = levels_completed
        self.win_levels = win_levels
        self.state = state
        self.available = available  # list of action names
        self.diff = diff  # exact [(x, y, old, new)] vs previous frame; [] for frame 0
        self.note = note
        m = dict(_COORD_RE.findall(action))
        self.x = int(m["x"]) if "x" in m else None
        self.y = int(m["y"]) if "y" in m else None
        self.action_name = action.split()[0]

    def __repr__(self):
        return (
            f"Step({self.i}, {self.action!r}, levels {self.levels_completed}/"
            f"{self.win_levels}, {self.state})"
        )


def _parse_grid(lines):
    return np.array([[HEX.index(c) for c in line.strip().upper()] for line in lines if line.strip()])


def load(path="log.txt"):
    """All frames from the log, oldest first, grids fully reconstructed."""
    with open(path) as f:
        text = f.read()
    steps = []
    prev = None
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        if lines[0].startswith(("[MARK]", "[PLAN ")):
            continue
        m = _HEADER_RE.match(lines[0])
        if not m:
            continue
        i = 1
        available = []
        if i < len(lines) and lines[i].startswith("[AVAILABLE] "):
            available = lines[i][len("[AVAILABLE] "):].split()
            i += 1
        note = ""
        if i < len(lines) and lines[i].startswith("[NOTE] "):
            note = lines[i][len("[NOTE] "):]
            i += 1
        diffs = []
        if i < len(lines) and lines[i].startswith("[DIFF]"):
            diffs = [
                (int(x), int(y), int(a), int(b))
                for x, y, a, b in _DIFF_CELL_RE.findall(lines[i])
            ]
            i += 1
        if i < len(lines) and lines[i] == "[GRID]":
            end = lines.index("[/GRID]", i)
            grid = _parse_grid(lines[i + 1:end])
        else:
            grid = prev.copy()
            for x, y, _old, new in diffs:
                grid[y, x] = new
        if prev is not None and prev.shape == grid.shape:
            # Always exact, even when the log line was a summarized bbox.
            ys, xs = np.nonzero(prev != grid)
            diffs = [(int(x), int(y), int(prev[y, x]), int(grid[y, x])) for y, x in zip(ys, xs, strict=False)]
        steps.append(
            Step(
                i=int(m.group("idx")),
                action=m.group("action").strip(),
                grid=grid,
                levels_completed=int(m.group("lv")),
                win_levels=int(m.group("win")),
                state=m.group("state"),
                available=available,
                diff=diffs,
                note=note,
            )
        )
        prev = grid
    return steps


def _as_grid(x):
    return x.grid if isinstance(x, Step) else np.asarray(x)


def diff(a, b):
    """Changed cells between two steps/grids as [(x, y, old, new)]."""
    ga, gb = _as_grid(a), _as_grid(b)
    if ga.shape != gb.shape:
        ys, xs = np.nonzero(np.ones_like(gb, dtype=bool))
        return [(int(x), int(y), -1, int(gb[y, x])) for y, x in zip(ys, xs, strict=False)]
    ys, xs = np.nonzero(ga != gb)
    return [(int(x), int(y), int(ga[y, x]), int(gb[y, x])) for y, x in zip(ys, xs, strict=False)]


def changed(a, b):
    return bool(diff(a, b))


class Component:
    def __init__(self, color, cells):
        self.color = int(color)
        self.cells = cells  # [(x, y), ...]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))  # (x0, y0, x1, y1)
        self.size = len(cells)
        self.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        x0, y0 = self.bbox[0], self.bbox[1]
        norm = tuple(sorted((x - x0, y - y0) for x, y in cells))
        self.hash = hashlib.sha256(repr((self.color, norm)).encode()).hexdigest()[:16]

    def __repr__(self):
        return f"Component(color={self.color}, size={self.size}, bbox={self.bbox})"


def objects(board, colors=None, connectivity=4):
    """Connected same-color components. Color 0 is treated as background unless
    ``colors`` explicitly includes 0; pass a set of colors to restrict."""
    grid = _as_grid(board)
    h, w = grid.shape
    if colors is None:
        wanted = set(int(v) for v in np.unique(grid)) - {0}
    else:
        wanted = {int(c) for c in colors}
    seen = np.zeros_like(grid, dtype=bool)
    if connectivity == 8:
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out = []
    for y in range(h):
        for x in range(w):
            v = int(grid[y, x])
            if seen[y, x] or v not in wanted:
                continue
            cells = []
            q = deque([(x, y)])
            seen[y, x] = True
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dy, dx in neigh:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and int(grid[ny, nx]) == v:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            out.append(Component(v, cells))
    return out
