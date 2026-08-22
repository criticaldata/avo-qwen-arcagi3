"""Keep the workspace gamelog helper in sync with the harness log format."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from arc3cb.logwriter import LogWriter  # noqa: E402

GAMELOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "arc3cb"
    / "workspace_template"
    / "gamelog.py"
)


def load_gamelog():
    spec = importlib.util.spec_from_file_location("gamelog", GAMELOG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gamelog_parses_harness_log(tmp_path):
    gamelog = load_gamelog()
    w = LogWriter(tmp_path / "log.txt")
    g0 = [[0] * 8 for _ in range(8)]
    g0[2][3] = 5
    w.append(0, "RESET", g0, levels_completed=0, win_levels=3, state="NOT_FINISHED",
             available=["ACTION1", "ACTION6", "RESET"])
    w.append_plan(1, "ACTION6 x=4 y=2 | expect: (4,2)=5")
    g1 = [row[:] for row in g0]
    g1[2][3] = 0
    g1[2][4] = 5
    w.append(1, "ACTION6 x=4 y=2", g1, levels_completed=1, win_levels=3, state="NOT_FINISHED")
    w.mark("level 1 reached")

    steps = gamelog.load(str(tmp_path / "log.txt"))
    assert len(steps) == 2
    s0, s1 = steps
    assert s0.action == "RESET"
    assert s0.available == ["ACTION1", "ACTION6", "RESET"]
    assert s0.grid[2, 3] == 5  # grid[y][x]
    assert s1.action_name == "ACTION6"
    assert (s1.x, s1.y) == (4, 2)
    assert s1.levels_completed == 1
    assert s1.win_levels == 3
    assert sorted(s1.diff) == [(3, 2, 5, 0), (4, 2, 0, 5)]
    assert gamelog.diff(s0, s1) == sorted(s1.diff)
    assert not gamelog.changed(s1, s1)


def test_gamelog_reconstructs_elided_grids(tmp_path):
    gamelog = load_gamelog()
    w = LogWriter(tmp_path / "log.txt", full_grid_every=100)
    g0 = [[0] * 6 for _ in range(6)]
    w.append(0, "RESET", g0, levels_completed=0, win_levels=1, state="NOT_FINISHED")
    g1 = [row[:] for row in g0]
    g1[5][1] = 9
    w.append(1, "ACTION1", g1, levels_completed=0, win_levels=1, state="NOT_FINISHED")
    steps = gamelog.load(str(tmp_path / "log.txt"))
    assert steps[1].grid[5, 1] == 9


def test_gamelog_objects(tmp_path):
    gamelog = load_gamelog()
    grid = np.zeros((6, 6), dtype=int)
    grid[1, 1] = grid[1, 2] = 3  # horizontal domino, color 3
    grid[4, 4] = 7
    comps = gamelog.objects(grid)
    assert len(comps) == 2
    domino = next(c for c in comps if c.color == 3)
    dot = next(c for c in comps if c.color == 7)
    assert domino.size == 2
    assert domino.bbox == (1, 1, 2, 1)  # (x0, y0, x1, y1)
    assert dot.cells == [(4, 4)]
    # translation-invariant shape hash
    grid2 = np.zeros((6, 6), dtype=int)
    grid2[0, 3] = grid2[0, 4] = 3
    domino2 = next(c for c in gamelog.objects(grid2) if c.color == 3)
    assert domino2.hash == domino.hash
