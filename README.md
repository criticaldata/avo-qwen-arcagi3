# arc3cb — ARC-AGI-3 world-model harness for open-weight models on Cerebras

`arc3cb` is a general-purpose [ARC-AGI-3](https://docs.arcprize.org) agent
harness whose only model backend is the Cerebras shared inference API
(OpenAI-compatible). The goal: the first credible public data point for
open-weight models run through a frontier-style world-model harness on
ARC-AGI-3, reported as RHAE on the 25 public games with official
competition-mode scorecards.

**The gap this fills.** The best documented open-weight results on the public
set are Tufa Labs' Duck harness (Qwen 3.6 27B under Kaggle single-GPU
constraints, ~1.6 mean score self-measured; Milestone 1 winner) and the
Polyphony Agent (Qwen3.6 self-hosted, 19.8% on the community leaderboard).
Closed models in specialized harnesses reach 95–100 RHAE (Tycho 100.0,
Retrodict 99.86, NVIDIA AVO 100 self-reported). Nobody has run a capable
open-weight model through such a harness with unconstrained tokens — Cerebras
shared inference (~1,850–3,000 tok/s) makes that experiment affordable and
fast.

## Architecture

Retrodiction-first, adapted from ideas documented publicly by
[Retrodict](https://github.com/ryanbbrown/retrodict) (log-as-context, plan
queues with expectations, playbook memory, escalation — reimplemented from
scratch; see [Credits](#credits-and-licenses)), with world-model-carryover
lessons from [duck-harness](https://github.com/Tufalabs/duck-harness) and
engineering patterns from [Tycho](https://github.com/NIMI-research/tycho):

- **Log-as-context.** Every frame is appended to a structured
  `workspace/log.txt`: header (`[FRAME n | action A | levels a/b | state S]`),
  `[AVAILABLE]` actions, a `[DIFF]` line of changed cells, and the full 64x64
  grid as hex text. The log is losslessly parseable; a standalone `gamelog.py`
  helper (numpy) is copied into every workspace so agent code can load frames,
  diff them, and segment objects.
- **Retrodiction before action.** The system prompt binds the agent to test
  every hypothesis about game mechanics against recorded history in Python
  before spending live actions. Being wrong against the log costs nothing.
- **Plan queue with expectations.** The agent emits an `[ACTIONS]` block; each
  action can carry the exact cells (plus `levels=` / `state=`) the board must
  show afterward. The runner executes the queue one action at a time with zero
  model calls, halts at the first mismatch, and re-invokes the model with the
  observed diff. GAME_OVER triggers a runner-issued recovery RESET.
- **Playbook memory.** `[PLAYBOOK]` blocks rewrite `playbook.md` (working
  model with CHECKED/ASSUMED claims + current-level working memory). When the
  conversation's input tokens pass the reset threshold (default 90k), the
  conversation is discarded and a fresh session is built from the system
  prompt, `playbook.md`, and `log.txt` — only files survive.
- **Escalation tiers.** After N actions stuck on one level (default 300,
  configurable; or 2 self-issued RESETs) a binding directive is injected:
  inventory what the log leaves unexplained, promote checked rules into an
  executable `step(state, action)` simulator under `scratch/`, verify it
  retrodicts every recorded frame, then search it for a route to the goal.
  Tier 2 (after another N) forces frontier enumeration.
- **Sandboxed agent code.** Agent Python runs in a dedicated containment venv
  with numpy/scipy/networkx and — provably — no game-engine packages:
  `containment.json` records that `arc_agi`/`arcengine` imports fail, and the
  run aborts if they do not.
- **Generality.** The system prompt contains zero game-specific content — no
  game IDs, no per-game heuristics, no hand-coded mechanics. One prompt plays
  all 25 games. See [Validity](#validity).

```
src/arc3cb/
  runner.py          # game loop, plan queue, expectation checks, budgets, caps
  transport.py       # Cerebras OpenAI-compatible client, retries, usage metering
  prompts.py         # game-agnostic system + re-invocation prompts
  logwriter.py       # frames -> log.txt, diff derivation, parse-back for replay
  plan_parser.py     # [ACTIONS] block extraction and validation
  tools.py           # sandboxed Python tool (containment venv)
  scoring.py         # local RHAE (official shipped-calculator semantics)
  arcclient.py       # official ARC-AGI-3 API client (cookies, scorecards, replay state)
  localenv.py        # local execution via the official arc-agi toolkit
  envs.py            # env interface + deterministic MockEnv for CI
  vision.py          # optional one-shot image priming (vision models)
  cli.py, config.py, results.py
  workspace_template/gamelog.py   # agent-side log reader (copied per run)
scripts/             # setup_containment_venv.sh, replay_runs.py, cerebras_smoke.py, ...
configs/             # models.yaml, budgets.yaml, containment.yaml, human_baselines.yaml
runs/                # one dir per run (gitignored): workspace/, transcript.jsonl,
                     # actions.jsonl, usage.jsonl, metrics.json, containment.json
docs/                # results.md, disclosures.md
tests/               # 60 unit + integration tests (mock game end-to-end)
```

## Setup

Requires Python >= 3.10 (>= 3.12 for `--mode local`, a requirement of the
official `arc-agi` toolkit).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'            # harness + tests
pip install -e '.[local]'          # optional: local game execution (py3.12+)
pip install -e '.[vision]'         # optional: image priming (pillow)
bash scripts/setup_containment_venv.sh   # builds .containment-venv + containment.json
cp .env.example .env               # then fill in the two keys
```

### API keys and tiers

- `CEREBRAS_API_KEY` — from [cloud.cerebras.ai](https://cloud.cerebras.ai).
  **A paid (Developer) tier key is required**: free-trial keys cap context at
  65k tokens and 5 requests/minute, below this harness's 90k-token reset
  threshold and interactive cadence. The runner verifies the served context
  limit at startup and fails fast with a tier explanation.
- `ARC_API_KEY` — register at [arcprize.org/platform](https://arcprize.org/platform)
  (Google/GitHub login → profile → API Keys). Needed for `--mode online` and
  for replaying runs onto official scorecards. Rate limit: 600 requests/min.

Keys live only in `.env` (gitignored) or the environment — never in YAML, CLI
args, or git. CI runs `scripts/check_secrets.py`; a pre-commit hook is
provided (`pre-commit install`).

## Usage

```bash
arc3cb models                        # live served catalog (never trust docs over this)
arc3cb probe --model gpt-oss-120b    # context verification + image capability probe
arc3cb run ls20 --model gpt-oss-120b --mode local --cost-cap 10
arc3cb run ls20 --model gemma-4-31b --mode online --cost-cap 10 --tags pilot
arc3cb results runs/                 # tables regenerated from run artifacts
```

Budgets (`configs/budgets.yaml`, all overridable per run): max actions per
level, max actions per game, max tokens, max dollars, wall clock. The run is
killed cleanly at any cap and partial results are recorded.

### Modes

- `--mode local` (default): the official toolkit downloads the game's real
  source and runs it in-process (~2,000 FPS, no rate limits, no scorecard).
  This is the cheap iteration mode.
- `--mode online`: live API play onto a scorecard (`--competition` for a
  competition-mode card).
- `--mode mock`: a built-in deterministic test game; used by CI and the
  offline smoke test. Never scored for reporting.

### Producing the official scorecard (campaign)

Local runs are replayed action-by-action through the live API onto ONE
competition-mode scorecard, with every frame verified against the recording
(`levels_completed` + `state`; add `--strict-boards` for full grids). This is
the accepted pattern (Retrodict precedent) for harnesses whose thinking time
could exceed the server's 15-minute scorecard-inactivity timeout — with
Cerebras throughput it rarely applies, but the fallback is built:

```bash
python scripts/replay_runs.py --competition --open-only          # prints card_id
python scripts/replay_runs.py --card-id <id> --keep-open --runs runs/<r1> runs/<r2> ...
python scripts/replay_runs.py --close-card <id>                  # saves scorecard JSON + URL
```

Replays that diverge abort that game rather than submit an unfaithful
scorecard. Always disclose replays and any restarts in `docs/disclosures.md`.

## Model matrix (live reality vs. plan)

`arc3cb models` enumerates what is actually served; as of 2026-08-22 the
Cerebras shared tier serves exactly two models:

| model | ctx (paid) | max out | $/Mtok in/out | speed | notes |
|---|---|---|---|---|---|
| `gpt-oss-120b` | 131,072 | 40,960 | 0.35 / 0.75 | ~3000 tok/s | reasoning_effort low/med/high |
| `gemma-4-31b` | 131,072 | 40,960 | 0.99 / 1.49 | ~1850 tok/s | vision; reasoning off by default |

`zai-glm-4.7`, `qwen-3-235b-a22b-instruct-2507`, and `llama3.1-8b` from the
original plan were deprecated by Cerebras (2026-05-27 / 2026-08-17) and now
404. **Qwen 3.8 27B is announced for the shared tier on 2026-09-03**
(replacing gemma-4-31b); `arc3cb models` and the CI smoke probe for a
qwen3.8-style id so it can be promoted to the top of the matrix the day it
appears — a placeholder entry exists in `configs/models.yaml`.

## Experiment plan

1. **Smoke:** `ls20`, `gpt-oss-120b`, local mode. Success = loop runs end to
   end, logs frames, executes a plan queue, survives a context reset, produces
   a score. (The loop itself is already exercised offline: the mock-game
   integration tests plus a scripted-model run against the real ls20 engine.)
2. **Pilot:** the three former preview games (`ls20`, `ft09`, `vc33`) x the
   served matrix (gpt-oss-120b, gemma-4-31b, and qwen3.8-27b once served),
   one run per game per model, fixed budgets. Compare RHAE, actions, tokens,
   dollars, wall clock.
3. **Campaign:** best one or two models across all 25 public games, single
   run per game — no cherry-picking; infrastructure restarts are disclosed.
   One official competition-mode scorecard per campaign.
4. **Report:** `docs/results.md` (regenerated via `arc3cb results`).

## Scoring

RHAE (Relative Human Action Efficiency), implementing exactly what official
scorecards compute (the shipped `arc-agi` v0.9.9 calculator; verified against
docs.arcprize.org/methodology):

- per completed level: `min((human_baseline / ai_actions)^2 * 100, 115)` — the
  1.15x cap applied post-square;
- game score: weighted mean over ALL levels (weight = 1-indexed level number;
  unreached levels score 0 but keep weight), additionally capped at the
  completed-weight fraction;
- campaign: plain mean over games. In-play RESETs count as actions; only the
  play-creating reset is free.

Human baselines come from `GET /api/games` (`baseline_actions`, the official
upper-median first-time human); `configs/human_baselines.yaml` is a dated
snapshot, refreshed live in online mode. Local RHAE is instant feedback only —
**official competition-mode scorecards are the source of truth for every
reported number.**

## Validity

The community-leaderboard bar is a general-purpose system: no per-game
hardcoding, prompts, or lookup tables. Grep-able proof that the prompts are
game-agnostic (the full public game list; expect zero matches in `src/`):

```bash
grep -rIiEn 'ls20|ft09|vc33|re86|m0r0|ka59|tu93|cd82|wa30|dc22|sk48|g50t|tr87|sc25|ar25|lp85|sp80|cn04|tn36|lf52|s5i5|sb26|r11l|bp35|su15|locksmith' src/
```

Complete traces are preserved per run — `transcript.jsonl` (every prompt and
reply), `usage.jsonl` (every model call's tokens and cost), `log.txt` (every
frame), `actions.jsonl` (the replayable action record), `containment.json` —
so results are independently verifiable, and `arc3cb results` regenerates all
tables from artifacts alone.

### Community leaderboard submission checklist

Per [ARC-AGI-Community-Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard)
CONTRIBUTING + validator, a submission needs:

1. `submissions/<lowercase-hyphenated-id>/submission.yaml` with `name`
   (unique), `authors` (each with ≥1 http link), `description`, `code_url`
   (public repo, must resolve), `versions[]`.
2. Each version: `version`, `date` (YYYY-MM-DD), `models[].name` (e.g.
   "GPT-OSS-120B (high) via Cerebras"), `scores[]`.
3. arc-agi-3 score entries: `benchmark: arc-agi-3`, `set: "public"`,
   **`scorecard_url`** (competition-mode card on arcprize.org — scores are
   derived from it; a self-reported `score:` field is rejected), optional
   `cost` in USD.
4. The scorecard must come from Competition Mode; label everything
   self-reported until ARC Prize verifies it.
5. Publish token counts and full traces in this repo (they are not schema
   fields).

## CI

- `ci.yml` — secret scan, ruff, the 60-test suite, containment venv build +
  verification. No keys needed.
- `cerebras-smoke.yml` — manual dispatch; uses the repository secret **`C7`**
  (a Cerebras API key) to enumerate the live served catalog, report
  experiment-matrix availability (including the qwen3.8 probe), and round-trip
  one tiny completion through the retrying transport with usage accounting.

## Credits and licenses

This repo is MIT (see LICENSE, © 2026 MIT Critical Data).

- **Retrodict** (ryanbbrown/retrodict) has **no license**; its architecture is
  documented publicly in its README and is reimplemented here from scratch —
  no code or prompt text was copied.
- **duck-harness** (Tufalabs) declares MIT via classifier only; its context
  eviction and world-model-carryover lessons informed the playbook design —
  no code copied.
- **Tycho** (NIMI-research, Apache-2.0) informed the scoring semantics and
  transport/sandbox patterns — reimplemented, not vendored.
- **ARC-AGI-3-Agents / arc-agi toolkit** (MIT, ARC Prize) is used as-is for
  local game execution via the `arc-agi` PyPI package.
