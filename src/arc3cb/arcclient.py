"""Official ARC-AGI-3 HTTP API client and the online GameEnv.

Facts this module encodes (verified against docs.arcprize.org and the shipped
``arc-agi`` toolkit, 2026-08):

- Base URL ``https://three.arcprize.org`` (override with ``ARC_BASE_URL``);
  auth via the ``X-API-Key`` header from ``ARC_API_KEY``.
- Games are stateful with session affinity: AWSALB cookies set by the server
  must ride along on every subsequent request. The client keeps one cookie jar
  and can serialize it (plus the play guid) so a replay can reattach after a
  crash without issuing a fresh RESET.
- ``POST /api/cmd/RESET`` body ``{game_id, card_id, guid?}``: without a guid it
  starts a new play; with one it resets — a level reset if any ACTION happened
  since the last level transition, otherwise a FULL game reset (hence the
  never-two-RESETs-in-a-row discipline).
- ``POST /api/cmd/ACTION{1..7}`` body ``{game_id, guid, reasoning?}`` plus
  ``x``/``y`` (0..63) for ACTION6.
- Frame responses: ``{game_id, guid, frame, state, levels_completed,
  win_levels, action_input, available_actions}``; ``frame`` is one or more
  64x64 grids (animation bursts). Older servers used ``score``/``win_score`` —
  both spellings are accepted here.
- Competition mode is requested at scorecard open (``competition_mode: true``)
  and enforced server-side: one scorecard, one play per game, game resets
  demoted to level resets, no in-flight scorecard reads.
- Rate limit 600 requests/minute; 429 bodies say RATE_LIMIT_EXCEEDED.

Retry policy: a 429 or a connection failure means the command was NOT executed,
so those are retried with backoff. A read timeout on ``/api/cmd/*`` is
ambiguous (the action may have landed) and is surfaced to the caller instead of
retried — re-sending could double-execute an action.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import httpx

from .envs import Frame

DEFAULT_BASE_URL = "https://three.arcprize.org"

ACTION_ID_TO_NAME = {0: "RESET", **{i: f"ACTION{i}" for i in range(1, 8)}}


class ArcApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"ARC API error {status}: {body[:1000]}")


class ArcTransportError(RuntimeError):
    pass


def _available_names(raw: dict) -> list[str]:
    names = []
    for a in raw.get("available_actions") or []:
        name = ACTION_ID_TO_NAME.get(int(a))
        if name:
            names.append(name)
    if "RESET" not in names:
        names.append("RESET")
    return names


def parse_frame(raw: dict, prev_grid: list[list[int]] | None = None) -> Frame:
    grids = raw.get("frame") or []
    grid = grids[-1] if grids else (prev_grid or [])
    levels = raw.get("levels_completed", raw.get("score", 0))
    win = raw.get("win_levels", raw.get("win_score", 0))
    return Frame(
        grid=grid,
        state=str(raw.get("state", "NOT_FINISHED")),
        levels_completed=int(levels),
        win_levels=int(win),
        grids=grids,
        available=_available_names(raw),
        guid=raw.get("guid") or "",
        raw=raw,
    )


class ArcClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 6,
        backoff_base_s: float = 2.0,
        http_transport: httpx.BaseTransport | None = None,
    ):
        key = api_key or os.environ.get("ARC_API_KEY")
        if not key:
            raise ArcTransportError(
                "ARC_API_KEY is not set; get one at https://arcprize.org/platform and put it "
                "in .env (see .env.example)"
            )
        self.base_url = (base_url or os.environ.get("ARC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": key, "Accept": "application/json"},
            timeout=timeout_s,
            transport=http_transport,
        )

    def close(self) -> None:
        self._client.close()

    # -- cookie persistence (session affinity survival for replays) ----------

    def cookies_as_dict(self) -> dict[str, str]:
        return dict(self._client.cookies)

    def restore_cookies(self, cookies: dict[str, str]) -> None:
        for k, v in cookies.items():
            self._client.cookies.set(k, v)

    # -- low level -----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Request with retries limited to cases where the command did not run."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self._client.request(method, path, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # Request never reached the server: safe to retry.
                last_err = ArcTransportError(f"connect failure on {path}: {e!r}")
            except httpx.HTTPError as e:
                # Read timeout etc.: the command may have executed — do not retry.
                raise ArcTransportError(
                    f"ambiguous transport failure on {path} (command may have executed): {e!r}"
                ) from e
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429 or resp.status_code in (502, 503):
                    # Throttled/rejected before execution: safe to retry.
                    last_err = ArcApiError(resp.status_code, resp.text)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            time.sleep(min(float(retry_after), 60.0))
                            continue
                        except ValueError:
                            pass
                else:
                    raise ArcApiError(resp.status_code, resp.text)
            if attempt <= self.max_retries:
                time.sleep(random.uniform(0, min(self.backoff_base_s * 2 ** (attempt - 1), 60.0)))
        raise ArcTransportError(f"giving up on {path}: {last_err}")

    # -- API surface ---------------------------------------------------------

    def list_games(self) -> list[dict]:
        return self._request("GET", "/api/games")

    def resolve_game_id(self, prefix: str) -> str:
        """Full versioned game_id from a name prefix (versions drift)."""
        games = self.list_games()
        matches = [g["game_id"] for g in games if g["game_id"].startswith(prefix)]
        if not matches:
            raise ArcTransportError(
                f"no game matching {prefix!r}; available: "
                + ", ".join(sorted(g["game_id"] for g in games))
            )
        if len(matches) > 1:
            raise ArcTransportError(f"ambiguous game prefix {prefix!r}: {', '.join(matches)}")
        return matches[0]

    def baselines_by_prefix(self) -> dict[str, list[int]]:
        return {
            g["game_id"].split("-")[0]: [int(x) for x in g.get("baseline_actions") or []]
            for g in self.list_games()
        }

    def open_scorecard(
        self,
        tags: list[str],
        source_url: str | None = None,
        opaque: dict | None = None,
        competition: bool = False,
    ) -> str:
        payload: dict = {"tags": tags}
        if source_url:
            payload["source_url"] = source_url
        if opaque:
            payload["opaque"] = opaque
        if competition:
            payload["competition_mode"] = True
        data = self._request("POST", "/api/scorecard/open", payload)
        return data["card_id"]

    def close_scorecard(self, card_id: str) -> dict:
        return self._request("POST", "/api/scorecard/close", {"card_id": card_id})

    def get_scorecard(self, card_id: str) -> dict:
        return self._request("GET", f"/api/scorecard/{card_id}")

    def cmd(
        self,
        name: str,
        game_id: str,
        card_id: str | None = None,
        guid: str | None = None,
        x: int | None = None,
        y: int | None = None,
        reasoning: dict | str | None = None,
    ) -> dict:
        payload: dict = {"game_id": game_id}
        if name == "RESET":
            if not card_id:
                raise ValueError("RESET requires card_id")
            payload["card_id"] = card_id
            if guid:
                payload["guid"] = guid
        else:
            if not guid:
                raise ValueError(f"{name} requires guid (RESET first)")
            payload["guid"] = guid
        if name == "ACTION6":
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
            payload["x"] = int(x)
            payload["y"] = int(y)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return self._request("POST", f"/api/cmd/{name}", payload)


class ArcOnlineEnv:
    """GameEnv over the live API. One instance = one play of one game."""

    def __init__(self, client: ArcClient, game_id: str, card_id: str, send_reasoning: bool = False):
        self.client = client
        self.game_id = game_id
        self.card_id = card_id
        self.send_reasoning = send_reasoning
        self.guid: str | None = None
        self._last_frame: Frame | None = None

    def available_actions(self) -> set[str]:
        if self._last_frame is None:
            return {"RESET"}
        return set(self._last_frame.available)

    def _track(self, raw: dict) -> Frame:
        prev = self._last_frame.grid if self._last_frame else None
        frame = parse_frame(raw, prev_grid=prev)
        if frame.guid:
            self.guid = frame.guid
        self._last_frame = frame
        return frame

    def reset(self) -> Frame:
        raw = self.client.cmd("RESET", self.game_id, card_id=self.card_id, guid=self.guid)
        return self._track(raw)

    def act(
        self, name: str, x: int | None = None, y: int | None = None, reasoning: str | None = None
    ) -> Frame:
        if name == "RESET":
            return self.reset()
        raw = self.client.cmd(
            name,
            self.game_id,
            guid=self.guid,
            x=x,
            y=y,
            reasoning=({"text": reasoning} if (reasoning and self.send_reasoning) else None),
        )
        return self._track(raw)

    def close(self) -> dict:
        return {"game_id": self.game_id, "guid": self.guid}

    # -- crash-resilience for replays ----------------------------------------

    def save_state(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "game_id": self.game_id,
                    "card_id": self.card_id,
                    "guid": self.guid,
                    "cookies": self.client.cookies_as_dict(),
                }
            )
        )

    @classmethod
    def reattach(cls, client: ArcClient, path: str | Path) -> ArcOnlineEnv:
        data = json.loads(Path(path).read_text())
        client.restore_cookies(data.get("cookies") or {})
        env = cls(client, data["game_id"], data["card_id"])
        env.guid = data.get("guid")
        return env
