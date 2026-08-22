#!/usr/bin/env python3
"""Replay recorded local runs through the official API onto a scorecard.

This is the fallback for live-session latency limits (scorecards auto-close
after ~15 minutes of inactivity) and the standard way to produce ONE official
competition-mode scorecard for a whole campaign after playing games locally.
The replay is a verified re-execution of the recorded actions, not a new
attempt: after every action the API frame's levels_completed and state must
match the recording (full boards too with --strict-boards) or the game aborts.

Typical campaign flow:
  1. throwaway test:   replay_runs.py --runs runs/<a> runs/<b>
  2. open the card:    replay_runs.py --competition --open-only
  3. replay all runs:  replay_runs.py --card-id <id> --keep-open --runs runs/*
  4. close + save:     replay_runs.py --close-card <id>

Crash resilience: after every step the play's guid and session-affinity cookies
are saved under .replay-state/; re-running with --resume continues a partially
replayed game without issuing a fresh RESET.

Requires ARC_API_KEY. Disclose replays in the results doc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arc3cb.arcclient import ArcClient, ArcOnlineEnv  # noqa: E402
from arc3cb.config import load_env_file  # noqa: E402
from arc3cb.logwriter import parse_log  # noqa: E402

_ACTION_RE = re.compile(r"^(RESET|ACTION[1-7])(?:\s+x=(\d+)\s+y=(\d+))?$")


def load_actions(run_dir: Path) -> list[dict]:
    path = run_dir / "actions.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — is this an arc3cb run dir?")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not records[0]["action"].startswith("RESET"):
        raise SystemExit(f"{path}: recording must start with the opening RESET")
    return records


class ReplayDiverged(RuntimeError):
    pass


def _save_state(env: ArcOnlineEnv, state_path: Path, next_index: int) -> None:
    """One atomic write (temp + rename) holding cookies, guid, and next_index."""
    state = {
        "game_id": env.game_id,
        "card_id": env.card_id,
        "guid": env.guid,
        "cookies": env.client.cookies_as_dict(),
        "next_index": next_index,
    }
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(state_path)


def replay_game(client: ArcClient, card_id: str, run_dir: Path, strict_boards: bool, resume: bool) -> dict:
    records = load_actions(run_dir)
    game_prefix = json.loads((run_dir / "metrics.json").read_text())["game_id"].split("-")[0]
    game_id = client.resolve_game_id(game_prefix)
    state_dir = Path(".replay-state")
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{game_prefix}.json"

    boards = None
    log_path = run_dir / "workspace" / "log.txt"
    if (strict_boards or resume) and log_path.exists():
        boards = {e.frame_index: e.grid for e in parse_log(log_path)}

    start_index = 0
    resumed = False
    if resume and state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved.get("card_id") != card_id:
            raise SystemExit(
                f"{game_prefix}: saved replay state targets card {saved.get('card_id')}, "
                f"not {card_id}; delete {state_path} to start over or pass the right --card-id"
            )
        env = ArcOnlineEnv.reattach(client, state_path)
        start_index = saved.get("next_index", 0)
        resumed = True
        print(f"{game_prefix}: resuming at record {start_index}")
    else:
        env = ArcOnlineEnv(client, game_id, card_id)

    for i in range(start_index, len(records)):
        rec = records[i]
        m = _ACTION_RE.match(rec["action"])
        if not m:
            raise ReplayDiverged(f"{game_prefix}: unparseable recorded action {rec['action']!r}")
        name, x, y = m.group(1), m.group(2), m.group(3)
        frame = env.act(name, x=int(x) if x else None, y=int(y) if y else None)
        if frame.levels_completed != rec["levels_completed"] or frame.state != rec["state"]:
            raise ReplayDiverged(
                f"{game_prefix}: DIVERGED at record {i} ({rec['action']}): recorded "
                f"levels={rec['levels_completed']}/state={rec['state']}, live "
                f"levels={frame.levels_completed}/state={frame.state}. The resulting "
                "scorecard would not be a faithful re-execution; aborting this game. "
                "(A record replayed right after --resume may have double-executed if "
                "the original failure was ambiguous — restart this game on a fresh card.)"
            )
        # The first record after a resume is the one whose original send failed
        # ambiguously; verify its full board when the log is available, so a
        # double-execution that happens not to move levels/state still aborts.
        check_board = boards is not None and (
            strict_boards or (resumed and i == start_index)
        )
        if check_board and rec["frame_index"] in boards and frame.grid != boards[rec["frame_index"]]:
            raise ReplayDiverged(
                f"{game_prefix}: board mismatch at record {i}"
                + ("" if strict_boards else " (post-resume verification)")
            )
        _save_state(env, state_path, i + 1)
    print(f"{game_prefix}: replayed {len(records) - start_index} records, final "
          f"levels {records[-1]['levels_completed']}/{records[-1]['win_levels']}")
    state_path.unlink(missing_ok=True)
    return {"game_id": game_id, "records": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="run directories to replay")
    parser.add_argument("--card-id", default=None, help="existing open scorecard to replay onto")
    parser.add_argument("--competition", action="store_true", help="open the card in competition mode")
    parser.add_argument("--open-only", action="store_true", help="open a card, print its id, exit")
    parser.add_argument("--close-card", default=None, help="close this card, save scorecard JSON, exit")
    parser.add_argument("--keep-open", action="store_true", help="do not close the card afterwards")
    parser.add_argument("--strict-boards", action="store_true", help="also verify full boards")
    parser.add_argument("--resume", action="store_true", help="continue from saved .replay-state")
    parser.add_argument("--tags", default="arc3cb-replay")
    parser.add_argument("--source-url", default=None)
    args = parser.parse_args()

    load_env_file()
    client = ArcClient()

    if args.close_card:
        card = client.close_scorecard(args.close_card)
        card.pop("api_key", None)
        out = Path(f"scorecard-{args.close_card}.json")
        out.write_text(json.dumps(card, indent=2) + "\n")
        print(f"closed; saved {out}")
        print(f"https://arcprize.org/scorecards/{args.close_card}")
        return 0

    card_id = args.card_id
    if not card_id:
        card_id = client.open_scorecard(
            tags=[t for t in args.tags.split(",") if t],
            source_url=args.source_url,
            competition=args.competition,
        )
        print(f"opened scorecard {card_id} (competition={args.competition})")
    if args.open_only:
        print("open-only: card left open; pass --card-id to replay onto it")
        return 0
    if not args.runs:
        raise SystemExit("no --runs given")

    failures = []
    for run_dir in args.runs:
        try:
            replay_game(client, card_id, Path(run_dir), args.strict_boards, args.resume)
        except ReplayDiverged as e:
            # A divergence aborts THAT game only; the rest of the campaign
            # continues and the card still gets closed per the flags below.
            print(f"REPLAY FAILED: {e}", file=sys.stderr)
            failures.append(str(run_dir))

    if failures:
        print(
            f"{len(failures)} game(s) failed to replay: {', '.join(failures)}",
            file=sys.stderr,
        )

    if not args.keep_open:
        card = client.close_scorecard(card_id)
        card.pop("api_key", None)
        out = Path(f"scorecard-{card_id}.json")
        out.write_text(json.dumps(card, indent=2) + "\n")
        print(f"closed; saved {out}")
        print(f"https://arcprize.org/scorecards/{card_id}")
    else:
        print(f"card {card_id} left open")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
