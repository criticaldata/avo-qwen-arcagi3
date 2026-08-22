from arc3cb.logwriter import (
    LogWriter,
    diff_grids,
    format_diff,
    parse_grid,
    parse_log,
    render_grid,
)


def make_grid(size=8, fill=0):
    return [[fill] * size for _ in range(size)]


def test_render_parse_roundtrip():
    g = make_grid()
    g[2][5] = 15
    g[7][0] = 10
    assert parse_grid(render_grid(g)) == g


def test_diff_grids_reports_x_y_convention():
    a = make_grid()
    b = make_grid()
    b[3][6] = 4  # row y=3, column x=6
    assert diff_grids(a, b) == [(6, 3, 0, 4)]


def test_diff_shape_change_marks_all_cells():
    a = make_grid(4)
    b = make_grid(5)
    diffs = diff_grids(a, b)
    assert len(diffs) == 25
    assert all(old == -1 for _, _, old, _ in diffs)


def test_format_diff_caps_listing():
    diffs = [(x, 0, 0, 1) for x in range(10)]
    assert "cells changed in bbox (0,0)-(9,0)" in format_diff(diffs, max_cells=5)
    assert "(0,0) 0->1" in format_diff(diffs, max_cells=10)


def test_writer_and_parse_log_roundtrip(tmp_path):
    path = tmp_path / "log.txt"
    w = LogWriter(path)
    g0 = make_grid()
    w.append(0, "RESET", g0, levels_completed=0, win_levels=2, state="NOT_FINISHED",
             available=["ACTION1", "ACTION4", "RESET"])
    g1 = [row[:] for row in g0]
    g1[1][2] = 3
    diffs = w.append(1, "ACTION4", g1, levels_completed=0, win_levels=2, state="NOT_FINISHED")
    assert diffs == [(2, 1, 0, 3)]
    w.mark("escalation tier 1")
    w.append_plan(2, "ACTION4 | expect: (3,1)=3")
    g2 = [row[:] for row in g1]
    g2[1][2] = 0
    g2[1][3] = 3
    w.append(2, "ACTION4 x=1 y=2", g2, levels_completed=1, win_levels=2, state="WIN", burst=3)

    entries = parse_log(path)
    assert [e.frame_index for e in entries] == [0, 1, 2]
    assert entries[0].action == "RESET"
    assert entries[0].available == ["ACTION1", "ACTION4", "RESET"]
    assert entries[1].grid == g1
    assert entries[2].grid == g2
    assert entries[2].action == "ACTION4 x=1 y=2"
    assert entries[2].levels_completed == 1
    assert entries[2].win_levels == 2
    assert entries[2].state == "WIN"
    assert entries[2].diffs == [(2, 1, 3, 0), (3, 1, 0, 3)]


def test_parse_log_reconstructs_elided_grids(tmp_path):
    path = tmp_path / "log.txt"
    w = LogWriter(path, full_grid_every=100)
    g0 = make_grid()
    w.append(0, "RESET", g0, levels_completed=0, win_levels=2, state="NOT_FINISHED")
    g1 = [row[:] for row in g0]
    g1[4][4] = 7
    w.append(1, "ACTION1", g1, levels_completed=0, win_levels=2, state="NOT_FINISHED")
    text = path.read_text()
    assert text.count("[GRID]") == 1  # second entry elided, diff-only
    entries = parse_log(path)
    assert entries[1].grid == g1
