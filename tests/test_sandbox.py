"""Real containment-venv tests; skipped unless the venv exists (CI builds it)."""

from pathlib import Path

import pytest

from arc3cb.tools import Sandbox, verify_containment

VENV = Path(".containment-venv")

pytestmark = pytest.mark.skipif(
    not (VENV / "bin" / "python").exists(),
    reason="containment venv not built (run scripts/setup_containment_venv.sh)",
)


def test_sandbox_runs_code_with_numpy(tmp_path):
    sb = Sandbox(VENV, tmp_path, tmp_path / "log.txt", timeout_s=30)
    result = sb.run("import numpy as np\nprint(int(np.arange(5).sum()))")
    assert result.exit_code == 0
    assert result.stdout.strip() == "10"


def test_sandbox_timeout(tmp_path):
    sb = Sandbox(VENV, tmp_path, tmp_path / "log.txt", timeout_s=2)
    result = sb.run("while True:\n    pass")
    assert result.timed_out


def test_sandbox_output_truncated(tmp_path):
    sb = Sandbox(VENV, tmp_path, tmp_path / "log.txt", timeout_s=30, max_output_chars=100)
    result = sb.run("print('x' * 10000)")
    assert len(result.stdout) < 300
    assert "truncated" in result.stdout


def test_containment_verified(tmp_path):
    report = verify_containment(VENV, out_path=tmp_path / "containment.json")
    assert report["contained"] is True
    assert (tmp_path / "containment.json").exists()
    assert report["modules"]["arc_agi"]["import_failed"] is True
    assert report["modules"]["arcengine"]["import_failed"] is True
    assert report["modules"]["numpy"]["import_failed"] is False


def test_gamelog_importable_inside_sandbox(tmp_path):
    """Agent code must be able to `import gamelog` and read the harness log."""
    from arc3cb.logwriter import LogWriter

    template = Path("src/arc3cb/workspace_template")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "gamelog.py").write_text((template / "gamelog.py").read_text())
    w = LogWriter(ws / "log.txt")
    grid = [[0] * 4 for _ in range(4)]
    grid[1][2] = 7
    w.append(0, "RESET", grid, levels_completed=0, win_levels=1, state="NOT_FINISHED",
             available=["ACTION1", "RESET"])
    sb = Sandbox(VENV, ws, ws / "log.txt", timeout_s=30)
    result = sb.run(
        "import gamelog\nsteps = gamelog.load()\n"
        "print(len(steps), steps[0].grid[1, 2], steps[0].state)"
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "1 7 NOT_FINISHED"
