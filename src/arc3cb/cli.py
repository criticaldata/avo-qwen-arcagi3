"""arc3cb command line.

    arc3cb run <game> --model <id> --mode local|online|mock [--cost-cap USD] ...
    arc3cb models                # live served catalog + experiment-matrix check
    arc3cb probe --model <id>    # context + image capability probe
    arc3cb verify-containment
    arc3cb results [runs_dir]    # table regenerated from run artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import (
    RunSettings,
    build_budgets,
    build_model_config,
    load_env_file,
    load_forbidden_modules,
    load_human_baselines,
)
from .runner import Runner
from .tools import ContainmentError, verify_containment
from .transport import CerebrasTransport, ModelConfig, TransportError, UsageMeter


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="play one game end to end")
    p.add_argument("game", help="game id or prefix, or 'mockgame' for --mode mock")
    p.add_argument("--model", required=True, help="Cerebras model id (see `arc3cb models`)")
    p.add_argument("--mode", choices=["online", "local", "mock"], default="local")
    p.add_argument("--cost-cap", type=float, default=None, help="max USD for this run")
    p.add_argument("--max-actions", type=int, default=None, help="max actions per game")
    p.add_argument("--max-actions-per-level", type=int, default=None)
    p.add_argument("--token-cap", type=int, default=None, help="max total tokens")
    p.add_argument("--wall-clock-cap", type=int, default=None, help="max seconds")
    p.add_argument("--context-reset", type=int, default=90_000,
                   help="input-token threshold that triggers a fresh session")
    p.add_argument("--escalation-after", type=int, default=300,
                   help="actions stuck on one level before the escalation directive")
    p.add_argument("--plan-max-len", type=int, default=20)
    p.add_argument("--image-prime", action="store_true",
                   help="one vision priming call on the opening frame (vision models only)")
    p.add_argument("--competition", action="store_true",
                   help="online mode: open the scorecard in competition mode")
    p.add_argument("--card-id", default=None,
                   help="online mode: play onto an existing open scorecard instead of a new one")
    p.add_argument("--keep-card-open", action="store_true",
                   help="online mode: do not close the scorecard at the end")
    p.add_argument("--tags", default="arc3cb", help="comma-separated scorecard tags")
    p.add_argument("--source-url", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--containment-venv", default=".containment-venv")
    p.add_argument("--seed", type=int, default=0, help="local mode: engine seed")
    p.add_argument("--tag", default="", help="free-form label recorded in the run dir name")


def cmd_run(args: argparse.Namespace) -> int:
    load_env_file()
    model_cfg = build_model_config(args.model, args.configs_dir)
    budgets = build_budgets(
        args.configs_dir,
        max_usd_per_game=args.cost_cap,
        max_actions_per_game=args.max_actions,
        max_actions_per_level=args.max_actions_per_level,
        max_tokens_per_game=args.token_cap,
        max_wall_clock_s=args.wall_clock_cap,
    )
    settings = RunSettings(
        game=args.game,
        mode=args.mode,
        model=model_cfg,
        budgets=budgets,
        context_reset_input_tokens=args.context_reset,
        escalation_after_actions=args.escalation_after,
        plan_max_len=args.plan_max_len,
        containment_venv=Path(args.containment_venv),
        runs_dir=Path(args.runs_dir),
        configs_dir=Path(args.configs_dir),
        tag=args.tag,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = f"{stamp}_{args.game}_{args.model}" + (f"_{args.tag}" if args.tag else "")
    run_dir = settings.runs_dir / label.replace("/", "-")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prove containment before anything else; abort the run if it fails.
    try:
        verify_containment(
            settings.containment_venv,
            load_forbidden_modules(args.configs_dir),
            run_dir / "containment.json",
        )
    except ContainmentError as e:
        print(f"CONTAINMENT FAILED: {e}", file=sys.stderr)
        return 1

    # 2. Model transport: enumerate what is actually served, verify context.
    meter = UsageMeter(run_dir / "usage.jsonl")
    try:
        transport = CerebrasTransport(model_cfg, meter)
        served = transport.verify_served(min_context=settings.context_reset_input_tokens)
    except TransportError as e:
        print(f"transport error: {e}", file=sys.stderr)
        return 1
    if model_cfg.price_input_per_mtok == 0 and model_cfg.price_output_per_mtok == 0:
        print(
            f"WARNING: no prices configured for {model_cfg.id} in configs/models.yaml — "
            "the cost meter will report $0.00 and the cost cap cannot bind."
        )

    image_prime = args.image_prime
    if image_prime and not (
        model_cfg.supports_images == "probe" and transport.probe_image_support()
    ):
        print("image priming requested but the endpoint rejected image input; continuing text-only")
        image_prime = False

    # 3. Environment.
    baselines_map = load_human_baselines(args.configs_dir)
    game_prefix = args.game.split("-")[0]
    scorecard_url = None
    client = None
    card_id = None
    if args.mode == "online":
        from .arcclient import ArcClient, ArcOnlineEnv

        client = ArcClient()
        game_id = client.resolve_game_id(args.game)
        try:
            baselines_map = client.baselines_by_prefix()
        except Exception:
            pass
        card_id = args.card_id or client.open_scorecard(
            tags=[t for t in args.tags.split(",") if t],
            source_url=args.source_url,
            competition=args.competition,
        )
        env = ArcOnlineEnv(client, game_id, card_id)
        print(f"scorecard: {card_id} (competition={args.competition})")
    elif args.mode == "local":
        from .localenv import ArcLocalEnv, LocalModeError

        try:
            env = ArcLocalEnv(args.game, recordings_dir=str(run_dir / "recordings"), seed=args.seed)
        except LocalModeError as e:
            print(f"local mode unavailable: {e}", file=sys.stderr)
            return 1
    else:
        from .envs import MockEnv

        env = MockEnv()

    human_baselines = baselines_map.get(game_prefix, [])
    if not human_baselines and args.mode != "mock":
        print(f"WARNING: no human baselines for {game_prefix}; local RHAE will be null")

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "argv": sys.argv[1:],
                "settings": {
                    "game": settings.game,
                    "mode": settings.mode,
                    "context_reset_input_tokens": settings.context_reset_input_tokens,
                    "escalation_after_actions": settings.escalation_after_actions,
                    "plan_max_len": settings.plan_max_len,
                    "budgets": vars(budgets),
                },
                "model": {k: v for k, v in vars(model_cfg).items()},
                "served_entry": served,
                "started": stamp,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    runner = Runner(settings, env, transport, run_dir, human_baselines=human_baselines)
    if image_prime and args.mode in ("local", "mock"):
        # Peek at the opening frame (local resets are free; the runner's own
        # opening RESET follows and lands on the same fresh state). Online mode
        # skips priming: an extra RESET there would burn a play.
        from . import vision

        opening = env.reset()
        note = vision.prime(transport, opening.grid)
        if note:
            runner.priming_note = note
            (run_dir / "prime.json").write_text(json.dumps({"note": note}, indent=2) + "\n")

    metrics = runner.run()

    if args.mode == "online" and client and card_id and not args.keep_card_open:
        try:
            card = client.close_scorecard(card_id)
            card.pop("api_key", None)
            (run_dir / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
            scorecard_url = f"https://arcprize.org/scorecards/{card_id}"
        except Exception as e:
            print(f"warning: failed to close scorecard: {e}", file=sys.stderr)

    print(json.dumps(metrics, indent=2))
    if scorecard_url:
        print(f"scorecard: {scorecard_url}")
    print(f"run artifacts: {run_dir}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    load_env_file()
    meter = UsageMeter()
    try:
        t = CerebrasTransport(ModelConfig(id="catalog"), meter)
        served = t.list_models()
        public = {m.get("id"): m for m in t.public_catalog()}
    except TransportError as e:
        print(f"transport error: {e}", file=sys.stderr)
        return 1
    print(f"served models at {t.base_url}:")
    for m in sorted(served, key=lambda m: m.get("id", "")):
        mid = m.get("id")
        p = public.get(mid) or {}
        limits = p.get("limits") or {}
        pricing = p.get("pricing") or {}
        extra = ""
        if limits:
            extra += f"  ctx={limits.get('max_context_length')} out={limits.get('max_completion_tokens')}"
        if pricing:
            try:
                extra += (
                    f"  ${float(pricing.get('prompt', 0)) * 1e6:.2f}/"
                    f"${float(pricing.get('completion', 0)) * 1e6:.2f} per Mtok"
                )
            except (TypeError, ValueError):
                pass
        caps = p.get("capabilities") or {}
        if caps.get("vision"):
            extra += "  vision"
        print(f"  - {mid}{extra}")
    import re

    qwen38 = [m.get("id") for m in served if re.search(r"qwen[-_.]?3\.?8", m.get("id", ""), re.I)]
    if qwen38:
        print(f"\nqwen3.8-style id detected: {', '.join(qwen38)} — promote to top of the matrix")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    load_env_file()
    model_cfg = build_model_config(args.model, args.configs_dir)
    meter = UsageMeter()
    try:
        t = CerebrasTransport(model_cfg, meter)
        entry = t.verify_served(min_context=args.context_reset)
    except TransportError as e:
        print(f"probe failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: entry[k] for k in sorted(entry)}, indent=2, default=str))
    if model_cfg.supports_images == "probe":
        ok = t.probe_image_support()
        print(f"image input: {'ACCEPTED' if ok else 'rejected'}")
    else:
        print("image input: not probed (supports_images=never)")
    return 0


def cmd_verify_containment(args: argparse.Namespace) -> int:
    try:
        verify_containment(
            args.containment_venv, load_forbidden_modules(args.configs_dir), args.out
        )
    except ContainmentError as e:
        print(f"CONTAINMENT FAILED: {e}", file=sys.stderr)
        return 1
    print(f"containment verified; report at {args.out}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    from .results import render_results

    print(render_results(Path(args.runs_dir), configs_dir=Path(args.configs_dir)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc3cb")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)

    p = sub.add_parser("models", help="enumerate the live served model catalog")

    p = sub.add_parser("probe", help="verify a model is served; probe image support")
    p.add_argument("--model", required=True)
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--context-reset", type=int, default=90_000)

    p = sub.add_parser("verify-containment", help="prove engine imports fail in the sandbox venv")
    p.add_argument("--containment-venv", default=".containment-venv")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--out", default="containment.json")

    p = sub.add_parser("results", help="regenerate results tables from run artifacts")
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--configs-dir", default="configs")

    args = parser.parse_args(argv)
    return {
        "run": cmd_run,
        "models": cmd_models,
        "probe": cmd_probe,
        "verify-containment": cmd_verify_containment,
        "results": cmd_results,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
