"""The game loop: invocations, plan queue, expectation checks, budgets, memory.

One Runner instance plays one game with one model. The control flow follows the
retrodiction architecture:

  invoke model -> execute blocks (python / playbook / plan queue) -> re-invoke
  with what changed -> ... until WIN or a budget kills the run cleanly.

The plan queue executes with zero model calls and halts (re-invoking the model)
on: queue exhausted, level change, state change, GAME_OVER (the runner issues
the recovery RESET itself), a planned action no longer available, or the first
failed expectation. Conversations are dropped and rebuilt from playbook.md +
log.txt when the input context passes the configured threshold, and a binding
escalation directive is injected when a level has soaked up too many actions.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .config import RunSettings
from .envs import Frame, GameEnv
from .logwriter import LogWriter
from .plan_parser import (
    PlannedAction,
    PlanParseError,
    check_expectations,
    extract_blocks,
    parse_plan,
)
from .scoring import LevelResult, game_rhae
from .tools import Sandbox
from .transport import TransportError

PARSE_RETRY_LIMIT = 3
MISMATCH_LIST_LIMIT = 5

REINVOKE_REASONS = {
    "queue_empty": "your plan was fully executed",
    "level_change": "the levels counter changed",
    "state_change": "the game state changed",
    "game_over": "the attempt hit GAME_OVER; the runner issued the recovery RESET for "
    "you and the attempt restarted",
    "unavailable_action": "the next planned action was not in the available set, so the "
    "rest of the plan was discarded",
    "prediction_mismatch": "an expectation failed, so the rest of the plan was discarded",
    "consecutive_reset": "the plan would have issued a RESET on an already-fresh attempt "
    "(which resets the WHOLE game); the rest of the plan was discarded",
}


@dataclass
class RunState:
    invocation: int = 0
    frame_index: int = 0
    actions_taken: int = 0  # every command sent after the opening RESET
    levels_completed: int = 0
    win_levels: int = 0
    level_start_actions: int = 0
    level_self_resets: int = 0
    escalation_tier: int = 0
    escalated_at_actions: int = 0
    fresh_sessions: int = 0
    parse_retries: int = 0
    surprises: int = 0
    stop_reason: str = ""
    level_results: list[LevelResult] = field(default_factory=list)
    last_executed_action: str = ""
    context_tokens: int = 0


class Runner:
    def __init__(
        self,
        settings: RunSettings,
        env: GameEnv,
        transport,  # CerebrasTransport-compatible: .chat(messages, purpose) + .meter
        run_dir: str | Path,
        human_baselines: list[int] | None = None,
        priming_note: str | None = None,
        sandbox: Sandbox | None = None,
    ):
        self.settings = settings
        self.priming_note = priming_note
        self.env = env
        self.transport = transport
        self.run_dir = Path(run_dir)
        self.workspace = self.run_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._seed_workspace()
        self.log = LogWriter(self.workspace / "log.txt", full_grid_every=settings.full_grid_every)
        self.playbook_path = self.workspace / "playbook.md"
        self.transcript_path = self.run_dir / "transcript.jsonl"
        self.actions_path = self.run_dir / "actions.jsonl"
        self.sandbox = sandbox or Sandbox(
            venv_dir=settings.containment_venv,
            workdir=self.workspace,
            log_path=self.workspace / "log.txt",
            timeout_s=settings.sandbox_timeout_s,
            max_output_chars=settings.sandbox_max_output_chars,
            memory_mb=settings.sandbox_memory_mb,
        )
        self.human_baselines = human_baselines or []
        self.state = RunState()
        self._started = time.monotonic()
        self._frame: Frame | None = None

    # -- setup ---------------------------------------------------------------

    def _seed_workspace(self) -> None:
        template = Path(__file__).parent / "workspace_template"
        for item in template.iterdir():
            dest = self.workspace / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)

    # -- persistence ---------------------------------------------------------

    def _record(self, path: Path, entry: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_frame(self, action_desc: str, frame: Frame, note: str = "") -> list:
        diffs = self.log.append(
            self.state.frame_index,
            action_desc,
            frame.grid,
            levels_completed=frame.levels_completed,
            win_levels=frame.win_levels,
            state=frame.state,
            available=frame.available,
            note=note,
            burst=frame.burst,
        )
        self._record(
            self.actions_path,
            {
                "frame_index": self.state.frame_index,
                "action": action_desc,
                "levels_completed": frame.levels_completed,
                "win_levels": frame.win_levels,
                "state": frame.state,
                "guid": frame.guid,
                "ts": time.time(),
            },
        )
        return diffs

    # -- budgets -------------------------------------------------------------

    def _budget_stop(self) -> str | None:
        b = self.settings.budgets
        if self.state.actions_taken >= b.max_actions_per_game:
            return "action_cap"
        if self.state.actions_taken - self.state.level_start_actions >= b.max_actions_per_level:
            return "level_action_cap"
        if self.transport.meter.total_tokens >= b.max_tokens_per_game:
            return "token_cap"
        if self.transport.meter.cost_usd >= b.max_usd_per_game:
            return "cost_cap"
        if b.max_wall_clock_s and time.monotonic() - self._started >= b.max_wall_clock_s:
            return "wall_clock_cap"
        return None

    # -- level / escalation bookkeeping ---------------------------------------

    def _on_level_change(self, new_levels: int) -> None:
        st = self.state
        actions_this_level = st.actions_taken - st.level_start_actions
        delta = new_levels - st.levels_completed
        for i in range(delta):
            level = st.levels_completed + 1 + i
            # A multi-level jump in one action is rare; charge the actions to the
            # first completed level and a floor of 1 to the rest (local feedback
            # only — official scorecards are authoritative).
            st.level_results.append(
                LevelResult(
                    level=level,
                    completed=True,
                    agent_actions=max(1, actions_this_level) if i == 0 else 1,
                )
            )
        st.levels_completed = new_levels
        st.level_start_actions = st.actions_taken
        st.level_self_resets = 0
        st.escalation_tier = 0
        st.escalated_at_actions = 0
        self.log.mark(f"level {new_levels} reached after {actions_this_level} actions")

    def _maybe_escalate(self) -> None:
        st = self.state
        threshold = self.settings.escalation_after_actions
        if threshold <= 0:
            return
        actions_here = st.actions_taken - st.level_start_actions
        if st.escalation_tier == 0:
            if actions_here >= 2 * threshold:
                st.escalation_tier = 2
                st.escalated_at_actions = actions_here
            elif actions_here >= threshold or (
                st.level_self_resets >= self.settings.escalation_self_resets
            ):
                st.escalation_tier = 1
                st.escalated_at_actions = actions_here
            else:
                return
            self.log.mark(f"escalation tier {st.escalation_tier} armed at {actions_here} actions")
        elif st.escalation_tier == 1 and actions_here - st.escalated_at_actions >= threshold:
            st.escalation_tier = 2
            self.log.mark(f"escalation tier 2 armed at {actions_here} actions")

    # -- plan execution -------------------------------------------------------

    def _execute_plan(self, plan: list[PlannedAction]) -> tuple[str, list[str]]:
        """Run the queue; returns (reason, feedback lines)."""
        st = self.state
        feedback: list[str] = []
        for idx, action in enumerate(plan):
            cap = self._budget_stop()
            if cap:
                return cap, feedback
            assert self._frame is not None
            if action.name == "RESET" and st.last_executed_action == "RESET":
                feedback.append(
                    f"halted before step {idx + 1} ({action.describe()}): the previous "
                    "executed action was already a RESET"
                )
                return "consecutive_reset", feedback
            if action.name not in set(self._frame.available) | {"RESET"}:
                feedback.append(
                    f"halted before step {idx + 1}: {action.describe()} is not in the "
                    f"available set [{' '.join(self._frame.available)}]"
                )
                return "unavailable_action", feedback
            prev_levels = self._frame.levels_completed
            prev_state = self._frame.state
            try:
                frame = self.env.act(action.name, x=action.x, y=action.y)
            except Exception as e:  # env/API failure is terminal for the run
                feedback.append(f"environment error on {action.describe()}: {e}")
                return "env_error", feedback
            st.frame_index += 1
            st.actions_taken += 1
            if action.name == "RESET":
                st.level_self_resets += 1
            st.last_executed_action = action.name
            self._frame = frame
            st.win_levels = frame.win_levels or st.win_levels
            diffs = self._log_frame(action.describe(), frame)
            executed = f"step {idx + 1}/{len(plan)} {action.describe()} executed"

            if frame.levels_completed > prev_levels:
                self._on_level_change(frame.levels_completed)
                if frame.state == "WIN":
                    return "win", feedback
                feedback.append(f"{executed} — LEVEL COMPLETED ({frame.levels_completed}"
                                f"/{frame.win_levels})")
                return "level_change", feedback
            if frame.state == "WIN":
                self._on_level_change(max(frame.levels_completed, st.levels_completed))
                return "win", feedback
            if frame.state == "GAME_OVER":
                feedback.append(f"{executed} — GAME_OVER; issuing recovery RESET")
                try:
                    frame = self.env.reset()
                except Exception as e:
                    feedback.append(f"environment error on recovery RESET: {e}")
                    return "env_error", feedback
                st.frame_index += 1
                st.actions_taken += 1  # in-play RESETs count as actions
                st.last_executed_action = "RESET"
                self._frame = frame
                self._log_frame("RESET", frame, note="runner-issued recovery after GAME_OVER")
                return "game_over", feedback
            if frame.state != prev_state:
                feedback.append(f"{executed} — state changed to {frame.state}")
                return "state_change", feedback
            if action.expect:
                fails = check_expectations(
                    action.expect, frame.grid, frame.levels_completed, frame.state
                )
                if fails:
                    st.surprises += 1
                    listed = "; ".join(fails[:MISMATCH_LIST_LIMIT])
                    if len(fails) > MISMATCH_LIST_LIMIT:
                        listed += f" (+{len(fails) - MISMATCH_LIST_LIMIT} more)"
                    feedback.append(
                        f"{executed} — EXPECTATION FAILED: {listed}. Observed diff: "
                        + self._short_diff(diffs)
                    )
                    return "prediction_mismatch", feedback
        return "queue_empty", feedback

    @staticmethod
    def _short_diff(diffs: list) -> str:
        from .logwriter import format_diff

        return format_diff(diffs, max_cells=40)

    # -- conversation ---------------------------------------------------------

    def _frame_text(self, diffs: list | None = None) -> str:
        assert self._frame is not None
        f = self._frame
        return prompts.frame_message(
            self.state.frame_index,
            self.state.last_executed_action or "RESET",
            f.grid,
            f.levels_completed,
            f.win_levels,
            f.state,
            f.available,
            diffs=diffs,
        )

    def _playbook(self) -> str:
        return self.playbook_path.read_text() if self.playbook_path.exists() else ""

    def _chat(self, conversation: list[dict]) -> tuple[str, str]:
        """One model call with a single retry on transport failure."""
        for attempt in (1, 2):
            try:
                result = self.transport.chat(conversation, purpose="agent")
                self.state.context_tokens = result.prompt_tokens
                return result.text, result.finish_reason
            except TransportError as e:
                if attempt == 2:
                    raise
                self.log.mark(f"transport failure, retrying once: {e}")
                time.sleep(5)
        raise AssertionError("unreachable")

    # -- main loop -------------------------------------------------------------

    def run(self) -> dict:
        st = self.state
        try:
            frame = self.env.reset()  # opening RESET: creates the play, uncharged
        except Exception as e:
            st.stop_reason = "env_error"
            return self._finish(error=str(e))
        self._frame = frame
        st.win_levels = frame.win_levels
        if frame.levels_completed:
            st.levels_completed = frame.levels_completed
        self._log_frame("RESET", frame, note="opening reset")

        conversation: list[dict] = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
        user_msg = prompts.initial_prompt(
            self.env.game_id, self._frame_text(), priming_note=self.priming_note
        )
        consecutive_parse_failures = 0

        while True:
            cap = self._budget_stop()
            if cap:
                st.stop_reason = cap
                break
            self._maybe_escalate()
            if st.escalation_tier > 0:
                user_msg += "\n\n" + prompts.escalation_directive(
                    st.escalation_tier,
                    st.actions_taken - st.level_start_actions,
                    st.level_self_resets,
                )
            conversation.append({"role": "user", "content": user_msg})
            st.invocation += 1
            try:
                text, finish_reason = self._chat(conversation)
            except TransportError as e:
                st.stop_reason = "provider_error"
                return self._finish(error=str(e))
            conversation.append({"role": "assistant", "content": text})
            self._record(
                self.transcript_path,
                {
                    "invocation": st.invocation,
                    "user": user_msg,
                    "assistant": text,
                    "finish_reason": finish_reason,
                    "context_tokens": st.context_tokens,
                    "escalation_tier": st.escalation_tier,
                    "actions_taken": st.actions_taken,
                    "levels_completed": st.levels_completed,
                },
            )

            feedback: list[str] = []
            reason = ""
            plan: list[PlannedAction] | None = None
            had_parse_error = False
            blocks = extract_blocks(text)
            actions_blocks = [c for k, c in blocks if k == "actions"]
            if len(actions_blocks) > 1:
                feedback.append(
                    prompts.parse_retry_prompt("more than one [ACTIONS] block in the reply")
                )
                had_parse_error = True
            else:
                for kind, content in blocks:
                    if kind == "python":
                        result = self.sandbox.run(content)
                        feedback.append(result.render())
                    elif kind == "playbook":
                        self.playbook_path.write_text(content + "\n")
                        feedback.append(f"[playbook.md updated: {len(content)} chars]")
                    elif kind == "actions":
                        try:
                            assert self._frame is not None
                            plan = parse_plan(
                                content,
                                valid_actions=set(self._frame.available) | {"RESET"},
                                max_len=self.settings.plan_max_len,
                            )
                        except PlanParseError as e:
                            feedback.append(prompts.parse_retry_prompt(str(e)))
                            had_parse_error = True
                            plan = None

            ran_python = any(k == "python" for k, _ in blocks)
            if had_parse_error:
                st.parse_retries += 1
                consecutive_parse_failures += 1
            elif plan or ran_python:
                consecutive_parse_failures = 0
            else:
                feedback.append(prompts.no_block_prompt())
                st.parse_retries += 1
                consecutive_parse_failures += 1

            if plan:
                self.log.append_plan(
                    st.invocation, "\n".join(a.raw for a in plan)
                )
                reason, plan_feedback = self._execute_plan(plan)
                feedback.extend(plan_feedback)
                if reason == "win":
                    st.stop_reason = "win"
                    break
                if reason in (
                    "action_cap",
                    "level_action_cap",
                    "token_cap",
                    "cost_cap",
                    "wall_clock_cap",
                    "env_error",
                ):
                    st.stop_reason = reason
                    break

            if consecutive_parse_failures >= PARSE_RETRY_LIMIT:
                st.stop_reason = "plan_parse_failed"
                break

            if finish_reason == "length":
                feedback.append(
                    "(your previous reply was cut off at the output token limit; "
                    "reply more concisely)"
                )

            reason_text = REINVOKE_REASONS.get(reason, reason) if reason else "reply processed"
            fresh = st.context_tokens > self.settings.context_reset_input_tokens
            if fresh:
                st.fresh_sessions += 1
                conversation = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
                user_msg = prompts.fresh_session_prompt(
                    self.env.game_id,
                    st.frame_index,
                    reason_text,
                    self._playbook(),
                    self._frame_text(),
                )
                self.log.mark(
                    f"context reset at {st.context_tokens} input tokens "
                    f"(fresh session {st.fresh_sessions})"
                )
            else:
                user_msg = prompts.reinvoke_prompt(reason_text, feedback, self._frame_text())

        return self._finish()

    # -- results ---------------------------------------------------------------

    def _finish(self, error: str = "") -> dict:
        st = self.state
        meter = self.transport.meter
        rhae = None
        if self.human_baselines:
            try:
                rhae = game_rhae(st.level_results, self.human_baselines)
            except ValueError:
                rhae = None
        metrics = {
            "game_id": self.env.game_id,
            "model": self.transport.model_cfg.id,
            "mode": self.settings.mode,
            "stop_reason": st.stop_reason,
            "error": error,
            "state": self._frame.state if self._frame else "NOT_PLAYED",
            "levels_completed": st.levels_completed,
            "win_levels": st.win_levels,
            "actions": st.actions_taken,
            "invocations": st.invocation,
            "fresh_sessions": st.fresh_sessions,
            "parse_retries": st.parse_retries,
            "surprises": st.surprises,
            "escalation_tier": st.escalation_tier,
            "prompt_tokens": meter.prompt_tokens,
            "completion_tokens": meter.completion_tokens,
            "total_tokens": meter.total_tokens,
            "cost_usd": round(meter.cost_usd, 4),
            "wall_seconds": round(time.monotonic() - self._started, 1),
            "level_results": [
                {"level": r.level, "completed": r.completed, "agent_actions": r.agent_actions}
                for r in st.level_results
            ],
            "human_baselines": self.human_baselines,
            "local_rhae": round(rhae, 2) if rhae is not None else None,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        try:
            close_info = self.env.close()
        except Exception as e:
            close_info = {"close_error": str(e)}
        (self.run_dir / "env_close.json").write_text(json.dumps(close_info, indent=2) + "\n")
        return metrics
