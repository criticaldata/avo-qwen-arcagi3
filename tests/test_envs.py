from arc3cb.envs import MockEnv


def test_mock_env_full_playthrough():
    env = MockEnv()
    f = env.reset()
    assert f.state == "NOT_FINISHED"
    assert f.levels_completed == 0
    assert f.win_levels == 2
    assert f.grid[10][10] == 3  # agent at start (x=10,y=10) -> grid[y][x]
    assert f.grid[10][20] == 2  # level-1 target

    for _ in range(10):  # walk right from (10,10) to (20,10)
        f = env.act("ACTION4")
    assert f.levels_completed == 1
    assert f.state == "NOT_FINISHED"
    assert f.grid[12][50] == 3  # level-2 start

    f = env.act("ACTION6", x=12, y=43)  # teleport next to level-2 target
    f = env.act("ACTION2")  # down onto (12,44)
    assert f.levels_completed == 2
    assert f.state == "WIN"


def test_mock_env_border_is_game_over_and_reset_restarts_level():
    env = MockEnv()
    env.reset()
    f = env.act("ACTION6", x=1, y=10)
    for _ in range(2):
        f = env.act("ACTION3")
    assert f.state == "GAME_OVER"
    # inputs ignored while game over
    assert env.act("ACTION1").state == "GAME_OVER"
    f = env.act("RESET")
    assert f.state == "NOT_FINISHED"
    assert f.grid[10][10] == 3
