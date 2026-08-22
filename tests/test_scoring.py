import pytest

from arc3cb.scoring import LevelResult, campaign_rhae, game_rhae, level_score


def test_level_score_matches_methodology_examples():
    # docs.arcprize.org/methodology: 10/10 -> 1.0, 10/20 -> 0.25, 10/100 -> 0.01
    assert level_score(10, 10) == pytest.approx(100.0)
    assert level_score(10, 20) == pytest.approx(25.0)
    assert level_score(10, 100) == pytest.approx(1.0)


def test_level_score_capped_post_square_at_115():
    # 1.15x-baseline cap applied post-square: (10/8)^2*100 = 156.25 -> 115
    assert level_score(10, 8) == 115.0
    # just under the cap: (10/94*10)... use h=100, a=94: (100/94)^2*100 ~ 113.17
    assert level_score(100, 94) == pytest.approx((100 / 94) ** 2 * 100)


def test_level_score_rejects_nonpositive():
    with pytest.raises(ValueError):
        level_score(0, 10)
    with pytest.raises(ValueError):
        level_score(10, 0)


def test_game_rhae_weighted_by_level_index():
    baselines = [10, 10]
    # Only level 2 completed at parity: (0*1 + 100*2)/3 = 66.67, capped by
    # completed-weight fraction 2/3 -> 66.67
    res = [LevelResult(level=2, completed=True, agent_actions=10)]
    assert game_rhae(res, baselines) == pytest.approx(200 / 3)


def test_game_rhae_completion_cap():
    # methodology example: 5 levels, first 4 completed -> max 10/15 = 66.7%
    baselines = [10] * 5
    res = [
        LevelResult(level=i, completed=True, agent_actions=1)  # hyper-efficient (115 each)
        for i in (1, 2, 3, 4)
    ]
    assert game_rhae(res, baselines) == pytest.approx(10 / 15 * 100)


def test_game_rhae_incomplete_levels_keep_weight():
    baselines = [10, 10, 10]
    res = [
        LevelResult(level=1, completed=True, agent_actions=10),
        LevelResult(level=2, completed=False, agent_actions=500),
    ]
    # (1*100 + 0 + 0) / 6 = 16.67; completion cap 1/6*100 -> 16.67
    assert game_rhae(res, baselines) == pytest.approx(100 / 6)


def test_game_rhae_perfect_run():
    baselines = [10, 20, 30]
    res = [LevelResult(level=i, completed=True, agent_actions=baselines[i - 1]) for i in (1, 2, 3)]
    assert game_rhae(res, baselines) == pytest.approx(100.0)


def test_campaign_mean():
    assert campaign_rhae([100.0, 50.0, 0.0]) == pytest.approx(50.0)
    assert campaign_rhae([]) == 0.0
