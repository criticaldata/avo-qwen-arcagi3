import pytest

from arc3cb.plan_parser import (
    Expectation,
    PlanParseError,
    check_expectations,
    extract_blocks,
    parse_plan,
)

VALID = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"}


def test_extract_blocks_in_order():
    text = (
        "thinking...\n```python\nprint(1)\n```\n"
        "[PLAYBOOK]\nrules\n[/PLAYBOOK]\n"
        "[ACTIONS]\nACTION1\n[/ACTIONS]\n"
    )
    kinds = [k for k, _ in extract_blocks(text)]
    assert kinds == ["python", "playbook", "actions"]


def test_parse_simple_plan():
    plan = parse_plan("ACTION1\nACTION5\n", VALID)
    assert [a.name for a in plan] == ["ACTION1", "ACTION5"]
    assert plan[0].expect is None


def test_parse_coordinates_and_expectations():
    plan = parse_plan(
        "ACTION6 x=12 y=40 | expect: (12,40)=3; (13,40)=0; levels=1; state=WIN", VALID
    )
    a = plan[0]
    assert (a.x, a.y) == (12, 40)
    assert a.expect.cells == [(12, 40, 3), (13, 40, 0)]
    assert a.expect.levels == 1
    assert a.expect.state == "WIN"


def test_parse_tolerates_commas_and_case():
    plan = parse_plan("action6 x=1, y=2\n", VALID)
    assert plan[0].name == "ACTION6"
    assert (plan[0].x, plan[0].y) == (1, 2)


@pytest.mark.parametrize(
    "block,frag",
    [
        ("ACTION9", "unknown action"),
        ("ACTION6", "requires coordinates"),
        ("ACTION6 x=64 y=2", "out of range"),
        ("ACTION1 x=3 y=3", "takes no coordinates"),
        ("ACTION1 | (2,2)=1", "must start with 'expect:'"),
        ("ACTION1 | expect: (2,2)=99", "out of range"),
        ("ACTION1 | expect: state=BANANA", "unknown state"),
        ("ACTION1 | expect: wibble", "unrecognized expectation"),
        ("", "no actions"),
        ("RESET\nRESET", "two consecutive RESET"),
    ],
)
def test_parse_errors_are_actionable(block, frag):
    with pytest.raises(PlanParseError, match=frag):
        parse_plan(block, VALID)


def test_plan_length_cap():
    with pytest.raises(PlanParseError, match="maximum is 3"):
        parse_plan("ACTION1\n" * 4, VALID, max_len=3)


def test_check_expectations_reports_mismatches():
    grid = [[0, 0], [0, 7]]
    exp = Expectation(cells=[(1, 1, 7), (0, 0, 5)], levels=2, state="WIN")
    fails = check_expectations(exp, grid, levels_completed=1, state="NOT_FINISHED")
    assert "expected (0,0)=5, observed (0,0)=0" in fails
    assert "expected levels=2, observed levels_completed=1" in fails
    assert "expected state=WIN, observed state=NOT_FINISHED" in fails
    assert len(fails) == 3


def test_check_expectations_pass():
    grid = [[0, 0], [0, 7]]
    exp = Expectation(cells=[(1, 1, 7)], levels=1, state="WIN")
    assert check_expectations(exp, grid, levels_completed=1, state="WIN") == []
