"""End-to-end tests of the game loop against the deterministic MockEnv.

A scripted FakeTransport plays the model; a FakeSandbox stands in for the
containment venv (real sandbox execution is covered by test_sandbox.py when a
containment venv exists). These tests are the offline smoke test: loop runs end
to end, logs frames, executes a plan queue, survives a context reset, handles
GAME_OVER, escalates, and produces a score.
"""

from __future__ import annotations

import json
from collections import deque

import pytest

from arc3cb.config import Budgets, RunSettings
from arc3cb.envs import MockEnv
from arc3cb.logwriter import parse_log
from arc3cb.runner import Runner
from arc3cb.tools import SandboxResult
from arc3cb.transport import ChatResult, ModelConfig, UsageMeter


class FakeTransport:
    def __init__(self, responses, prompt_tokens=1000):
        self.model_cfg = ModelConfig(id="fake-model", price_input_per_mtok=1.0,
                                     price_output_per_mtok=1.0)
        self.meter = UsageMeter()
        self.responses = deque(responses)
        self.prompt_tokens = prompt_tokens
        self.requests = []

    def chat(self, messages, purpose="agent", max_output_tokens=None):
        self.requests.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        text = self.responses.popleft()
        result = ChatResult(
            text=text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=100,
            finish_reason="stop",
            latency_s=0.0,
            raw={},
        )
        self.meter.record(self.model_cfg, purpose, result, 1)
        return result


class FakeSandbox:
    def __init__(self):
        self.codes = []

    def run(self, code):
        self.codes.append(code)
        return SandboxResult(stdout="fake-ok", stderr="", exit_code=0, timed_out=False)


def make_runner(tmp_path, responses, prompt_tokens=1000, **settings_overrides):
    defaults = dict(
        game="mockgame",
        mode="mock",
        model=ModelConfig(id="fake-model"),
        budgets=Budgets(max_actions_per_game=100, max_actions_per_level=100,
                        max_tokens_per_game=10**9, max_usd_per_game=10**9),
    )
    defaults.update(settings_overrides)
    settings = RunSettings(**defaults)
    transport = FakeTransport(responses, prompt_tokens=prompt_tokens)
    runner = Runner(
        settings, MockEnv(), transport, tmp_path / "run", sandbox=FakeSandbox()
    )
    return runner, transport


WIN_LEVEL_1 = "[ACTIONS]\n" + "ACTION4\n" * 9 + "ACTION4 | expect: levels=1\n[/ACTIONS]"
WIN_LEVEL_2 = "[ACTIONS]\nACTION6 x=12 y=43\nACTION2 | expect: levels=2; state=WIN\n[/ACTIONS]"


def test_full_win_and_scoring(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            "[PLAYBOOK]\nWORKING MODEL: moving right seems safe (ASSUMED)\n[/PLAYBOOK]\n"
            + WIN_LEVEL_1,
            WIN_LEVEL_2,
        ],
    )
    runner.human_baselines = [10, 4]
    metrics = runner.run()

    assert metrics["stop_reason"] == "win"
    assert metrics["levels_completed"] == 2
    assert metrics["state"] == "WIN"
    # 10 moves for level 1 + teleport + move for level 2; opening RESET uncharged.
    assert metrics["actions"] == 12
    assert metrics["level_results"] == [
        {"level": 1, "completed": True, "agent_actions": 10},
        {"level": 2, "completed": True, "agent_actions": 2},
    ]
    # (1*100 + 2*115)/3 = 110, capped by completion fraction at 100.
    assert metrics["local_rhae"] == pytest.approx(100.0)
    assert metrics["cost_usd"] > 0

    run_dir = tmp_path / "run"
    assert (run_dir / "workspace" / "playbook.md").read_text().startswith("WORKING MODEL")
    assert (run_dir / "workspace" / "gamelog.py").exists()
    entries = parse_log(run_dir / "workspace" / "log.txt")
    assert len(entries) == 13  # opening reset + 12 actions
    assert entries[0].action == "RESET"
    assert entries[-1].state == "WIN"
    assert (run_dir / "metrics.json").exists()
    actions = [json.loads(line) for line in (run_dir / "actions.jsonl").read_text().splitlines()]
    assert len(actions) == 13


def test_expectation_mismatch_halts_and_reports(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            # Wrong prediction: after one step right from (10,10), (11,10) is agent
            # color 3, but we claim 7. Remaining plan must not execute.
            "[ACTIONS]\nACTION4 | expect: (11,10)=7\nACTION4\nACTION4\n[/ACTIONS]",
            WIN_LEVEL_1.replace("ACTION4\n" * 9, "ACTION4\n" * 8),  # 9 more moves to goal
            WIN_LEVEL_2,
        ],
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    assert metrics["surprises"] == 1
    reinvoke = transport.requests[1][-1]["content"]
    assert "EXPECTATION FAILED" in reinvoke
    assert "expected (11,10)=7, observed (11,10)=3" in reinvoke


def test_game_over_triggers_runner_reset(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            # Teleport next to the border, then walk into it -> GAME_OVER.
            "[ACTIONS]\nACTION6 x=1 y=10\nACTION3\nACTION5\n[/ACTIONS]",
            WIN_LEVEL_1,
            WIN_LEVEL_2,
        ],
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    reinvoke = transport.requests[1][-1]["content"]
    assert "GAME_OVER" in reinvoke
    entries = parse_log(tmp_path / "run" / "workspace" / "log.txt")
    recovery = [e for e in entries if "recovery" in e.note]
    assert len(recovery) == 1
    # teleport + walk + recovery RESET + 10 + 2
    assert metrics["actions"] == 15


def test_python_blocks_run_in_sandbox(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            "```python\nprint('hello')\n```",
            WIN_LEVEL_1,
            WIN_LEVEL_2,
        ],
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    assert runner.sandbox.codes == ["print('hello')"]
    reinvoke = transport.requests[1][-1]["content"]
    assert "fake-ok" in reinvoke


def test_context_reset_builds_fresh_session(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            "[PLAYBOOK]\nWORKING MODEL: right is safe (CHECKED)\n[/PLAYBOOK]\n"
            "[ACTIONS]\nACTION4\n[/ACTIONS]",
            WIN_LEVEL_1.replace("ACTION4\n" * 9, "ACTION4\n" * 8),
            WIN_LEVEL_2,
        ],
        prompt_tokens=50_000,
        context_reset_input_tokens=10_000,
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    assert metrics["fresh_sessions"] >= 1
    # After the reset the conversation is system + fresh user message only.
    fresh_request = transport.requests[1]
    assert len(fresh_request) == 2
    assert "already in progress" in fresh_request[-1]["content"]
    assert "WORKING MODEL: right is safe" in fresh_request[-1]["content"]


def test_unsafe_reset_blocked(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        [
            # A RESET as the very first action would reset the ENTIRE game.
            "[ACTIONS]\nRESET\n[/ACTIONS]",
            # After one real action a RESET is a legal level reset...
            "[ACTIONS]\nACTION4\nRESET\n[/ACTIONS]",
            # ...but a second RESET right after (0 actions since) is blocked again.
            "[ACTIONS]\nRESET\n[/ACTIONS]",
            WIN_LEVEL_1,
            WIN_LEVEL_2,
        ],
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    first = transport.requests[1][-1]["content"]
    assert "would reset the ENTIRE game" in first
    third = transport.requests[3][-1]["content"]
    assert "would reset the ENTIRE game" in third
    # the mid-plan RESET executed (level reset), so exactly one self-reset counted
    assert metrics["actions"] == 14  # ACTION4 + RESET + 10 + 2


def test_parse_failure_limit_kills_run(tmp_path):
    runner, transport = make_runner(
        tmp_path,
        ["no blocks here", "still nothing", "nope"],
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "plan_parse_failed"
    assert metrics["parse_retries"] == 3
    nudge = transport.requests[1][-1]["content"]
    assert "no python block and no [ACTIONS] block" in nudge


def test_escalation_directive_injected(tmp_path):
    wander = "[ACTIONS]\nACTION1\nACTION2\n[/ACTIONS]"
    runner, transport = make_runner(
        tmp_path,
        [wander, wander, wander, WIN_LEVEL_1, WIN_LEVEL_2],
        escalation_after_actions=4,
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    # After 4 wandering actions the next prompt must carry the directive.
    assert any(
        "[ESCALATION tier 1]" in req[-1]["content"] for req in transport.requests
    )


def test_action_cap_stops_cleanly(tmp_path):
    wander = "[ACTIONS]\nACTION1\nACTION2\n[/ACTIONS]"
    runner, transport = make_runner(
        tmp_path,
        [wander] * 3,
        budgets=Budgets(max_actions_per_game=4, max_actions_per_level=100,
                        max_tokens_per_game=10**9, max_usd_per_game=10**9),
    )
    metrics = runner.run()
    assert metrics["stop_reason"] == "action_cap"
    assert metrics["actions"] == 4
    assert (tmp_path / "run" / "metrics.json").exists()  # partial results recorded


def test_unavailable_action_halts(tmp_path):
    class NoAction5Env(MockEnv):
        def available_actions(self):
            return {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"}

    settings = RunSettings(game="mockgame", mode="mock", model=ModelConfig(id="fake-model"),
                           budgets=Budgets())
    transport = FakeTransport(
        ["[ACTIONS]\nACTION4\n[/ACTIONS]", WIN_LEVEL_1, WIN_LEVEL_2]
    )
    runner = Runner(settings, NoAction5Env(), transport, tmp_path / "run", sandbox=FakeSandbox())
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    # ACTION5 not offered: a plan containing it is rejected at parse time.
    transport2 = FakeTransport(["[ACTIONS]\nACTION5\n[/ACTIONS]", "x", "y"])
    runner2 = Runner(settings, NoAction5Env(), transport2, tmp_path / "run2", sandbox=FakeSandbox())
    runner2.run()
    retry = transport2.requests[1][-1]["content"]
    assert "ACTION5 is not currently available" in retry


def test_context_limit_error_forces_emergency_fresh_session(tmp_path):
    from arc3cb.transport import ContextLimitError

    class OverflowOnceTransport(FakeTransport):
        def __init__(self, responses):
            super().__init__(responses)
            self.overflowed = False

        def chat(self, messages, purpose="agent", max_output_tokens=None):
            if not self.overflowed and len(self.requests) == 1:
                self.requests.append([dict(m) for m in messages])
                self.overflowed = True
                raise ContextLimitError(400, "prompt exceeds maximum context length")
            return super().chat(messages, purpose, max_output_tokens)

    settings = RunSettings(game="mockgame", mode="mock", model=ModelConfig(id="fake-model"),
                           budgets=Budgets())
    transport = OverflowOnceTransport(
        ["[ACTIONS]\nACTION4\n[/ACTIONS]", WIN_LEVEL_1.replace("ACTION4\n" * 9, "ACTION4\n" * 8),
         WIN_LEVEL_2]
    )
    runner = Runner(settings, MockEnv(), transport, tmp_path / "run", sandbox=FakeSandbox())
    metrics = runner.run()
    assert metrics["stop_reason"] == "win"
    assert metrics["fresh_sessions"] == 1
    # The request after the overflow is a rebuilt 2-message fresh session.
    rebuilt = transport.requests[2]
    assert len(rebuilt) == 2
    assert "exceeded the model's context window" in rebuilt[-1]["content"]
