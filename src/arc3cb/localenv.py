"""Local game execution via the official ``arc-agi`` toolkit.

``pip install 'arc3cb[local]'`` pulls ``arc-agi`` (requires Python >= 3.12),
which downloads each game's metadata and Python source from the official API
into ``environment_files/`` on first use and then runs it fully offline against
the open ``arcengine``. This is the cheap-iteration mode: ~thousands of FPS, no
rate limits, no scorecards. Runs recorded here can be replayed onto an official
competition-mode scorecard with scripts/replay_runs.py.

The adapter is defensive about toolkit versions: it reads both the 0.9.3+
field names (levels_completed/win_levels) and the older score/win_score.
"""

from __future__ import annotations

from typing import Any

from .arcclient import ACTION_ID_TO_NAME
from .envs import Frame


class LocalModeError(RuntimeError):
    pass


def _to_grid(obj: Any) -> list[list[int]]:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return [list(row) for row in obj]


class ArcLocalEnv:
    """GameEnv over arc_agi.Arcade in NORMAL (download+local) or OFFLINE mode."""

    def __init__(
        self,
        game_ref: str,
        environments_dir: str = "environment_files",
        recordings_dir: str | None = None,
        offline_only: bool = False,
        seed: int = 0,
    ):
        try:
            from arc_agi import Arcade, OperationMode
            from arcengine import GameAction
        except ImportError as e:
            raise LocalModeError(
                "local mode needs the official toolkit: pip install 'arc3cb[local]' "
                "(the arc-agi package requires Python >= 3.12). Alternatively use "
                "--mode online or --mode mock."
            ) from e
        import os

        self._GameAction = GameAction
        mode = OperationMode.OFFLINE if offline_only else OperationMode.NORMAL
        # The toolkit loads .env/.env.example at import time and gives the
        # OPERATION_MODE env var precedence over the constructor argument; pin
        # it explicitly so --mode local can never silently play online.
        os.environ["OPERATION_MODE"] = mode.value if hasattr(mode, "value") else str(mode)
        kwargs: dict = {"operation_mode": mode, "environments_dir": environments_dir}
        if recordings_dir:
            kwargs["recordings_dir"] = recordings_dir
        self.arcade = Arcade(**kwargs)
        self.game_id = self._resolve(game_ref)
        # make() creates the play and issues the opening RESET itself.
        self.env = self.arcade.make(self.game_id, seed=seed)
        self._opening_frame: Frame | None = self._read_frame(None)
        self._last: Frame | None = self._opening_frame

    def _resolve(self, game_ref: str) -> str:
        try:
            envs = self.arcade.get_environments()
        except Exception:
            return game_ref
        ids = []
        for e in envs:
            gid = e.get("game_id") if isinstance(e, dict) else getattr(e, "game_id", str(e))
            if gid:
                ids.append(gid)
        matches = [g for g in ids if g.startswith(game_ref)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise LocalModeError(f"ambiguous game {game_ref!r}: {', '.join(matches)}")
        return game_ref

    def _read_frame(self, step_result: Any) -> Frame:
        obs = step_result if step_result is not None else getattr(self.env, "observation_space", None)
        if obs is None:
            raise LocalModeError("toolkit returned no observation")
        state = getattr(obs, "state", "NOT_FINISHED")
        state = getattr(state, "name", None) or str(state)
        levels = getattr(obs, "levels_completed", None)
        if levels is None:
            levels = getattr(obs, "score", 0)
        win = getattr(obs, "win_levels", None)
        if win is None:
            win = getattr(obs, "win_score", 0)
        raw_frames = getattr(obs, "frame", None) or []
        grids = [_to_grid(g) for g in raw_frames]
        if not grids and self._last is not None:
            grids = [self._last.grid]
        available = []
        for a in getattr(obs, "available_actions", None) or []:
            name = ACTION_ID_TO_NAME.get(int(getattr(a, "value", a)))
            if name:
                available.append(name)
        if not available:
            try:
                available = [a.name for a in self.env.action_space]
            except Exception:
                available = list(ACTION_ID_TO_NAME.values())
        if "RESET" not in available:
            available.append("RESET")
        frame = Frame(
            grid=grids[-1] if grids else [],
            state=state,
            levels_completed=int(levels),
            win_levels=int(win),
            grids=grids,
            available=available,
            guid=str(getattr(obs, "guid", "") or ""),
        )
        self._last = frame
        return frame

    def available_actions(self) -> set[str]:
        return set(self._last.available) if self._last else {"RESET"}

    def reset(self) -> Frame:
        if self._opening_frame is not None:
            # make() already reset; a second RESET on a fresh play would restart
            # the whole game.
            frame, self._opening_frame = self._opening_frame, None
            return frame
        result = self.env.reset()
        return self._read_frame(result)

    def act(self, name: str, x: int | None = None, y: int | None = None) -> Frame:
        if name == "RESET":
            return self.reset()
        action = self._GameAction.from_name(name)
        data = {"x": int(x), "y": int(y)} if name == "ACTION6" else None
        result = self.env.step(action, data) if data else self.env.step(action)
        if result is None:
            # The toolkit swallows engine exceptions and returns None; treating
            # that as a no-op frame would silently record a lie.
            raise LocalModeError(
                f"engine returned no frame for {name} (the toolkit swallowed an "
                "engine exception; see its log output)"
            )
        return self._read_frame(result)

    def close(self) -> dict:
        try:
            card = self.arcade.get_scorecard()
            if hasattr(card, "model_dump"):
                card = card.model_dump()
            if isinstance(card, dict):
                card.pop("api_key", None)
            return {"game_id": self.game_id, "local_scorecard": card}
        except Exception as e:
            return {"game_id": self.game_id, "local_scorecard_error": str(e)}
