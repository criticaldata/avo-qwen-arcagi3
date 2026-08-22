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
  with numpy/scipy/networkx and no game-engine packages: `containment.json`
  records that `arc_agi`/`arcengine` imports fail (and pip is removed from the
  venv), and the run aborts if they do not. Scope is honest: this contains
  accidental engine use and runaway code (isolated mode, process-group kill,
  rlimits) — it is not an adversarial security boundary; filesystem reads and
  network are not blocked, and the full audit trail is what makes runs
  reviewable.
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
tests/               # 62 unit + integration tests (mock game end-to-end)
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
  threshold and interactive cadence. Key-tier caps are not visible in any
  catalog, so the runner verifies the model's catalog context at startup and
  handles a runtime context-limit rejection with an emergency fresh session —
  if it persists, the run ends with a clear tier explanation.
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
| `gemma-4-31b` | 131,072 | 40,960 | 0.99 / 1.49 | ~1850 tok/s | vision; reasoning off by default; **leaves the shared tier ~2026-09-03** |
| qwen3.8-27b (announced) | TBD | TBD | TBD (est. $0.35–1.49 band) | TBD | multimodal per the announcement; arrives ~2026-09-03; placeholder config ready |

`zai-glm-4.7`, `qwen-3-235b-a22b-instruct-2507`, and `llama3.1-8b` from the
original plan were deprecated by Cerebras (2026-05-27 / 2026-08-17) and now
404. **Qwen 3.8 27B was announced for the shared tier on 2026-09-03**
(replacing gemma-4-31b) in Cerebras' customer email of 2026-08-21 — it does
not yet appear in the public docs or the live catalog, so treat the date as
tentative; `arc3cb models` and the CI smoke probe for a qwen3.8-style id so it
can be promoted to the top of the matrix the day it appears — a placeholder
entry exists in `configs/models.yaml`.

## Experiment plan

Two scheduling windows, forced by Cerebras' announced shared-tier swap
(gemma-4-31b out, Qwen 3.8 27B in, ~2026-09-03, per their customer email of
2026-08-21; the date is tentative until the catalog changes):

**Window 1 — now until ~2026-09-02 (gemma-4-31b still served):**

1. **Smoke:** `ls20`, `gpt-oss-120b`, local mode. Success = loop runs end to
   end, logs frames, executes a plan queue, survives a context reset, produces
   a score. (The loop itself is already exercised offline: the mock-game
   integration tests plus a scripted-model run against the real ls20 engine;
   the live transport smoke is green in CI.)
2. **Pilot:** the three former preview games (`ls20`, `ft09`, `vc33`) x
   gpt-oss-120b and gemma-4-31b, one run per game per model, fixed budgets
   (~$60–150 total). Compare RHAE, actions, tokens, dollars, wall clock.
3. **Gemma campaign (conditional, deadline-bound):** if gemma's pilot numbers
   justify it, its 25-game campaign must finish before the swap date —
   $400–625 at default caps, ≥ ~22 h serial under its 500K TPM limit, so the
   go/no-go decision is needed no later than ~2026-09-01. After the swap this
   experiment stops being possible on the shared tier.

**Window 2 — from the day qwen3.8-27b first appears in `arc3cb models`
(~2026-09-03):**

4. **Qwen 3.8 27B onboarding + pilot (~1 day, ~$10–75):** rename the
   placeholder key in `configs/models.yaml` to the served slug, fill prices
   from the public catalog (they feed the cost meter automatically), run
   `arc3cb probe` (context limit, reasoning controls, image support — the
   announcement says multimodal), then the same 3-game pilot.
5. **Headline campaign:** qwen3.8-27b across all 25 public games — the
   marquee run of this project: the first Qwen-on-Cerebras world-model-harness
   data point, against Duck's and Polyphony's self-hosted Qwen 3.6 results.
   Estimated $250–650 at default caps (price-dependent, see the cost model);
   the $625 hard ceiling holds regardless of the unknown price.
6. **Report:** `docs/results.md` (regenerated via `arc3cb results`), one
   official competition-mode scorecard per campaign.

Steps 1–2 and 4 are also the fallback path if either window is missed: the
harness is model-agnostic within the OpenAI-compatible catalog, so the plan
re-anchors on whatever `arc3cb models` actually lists.

## Measurement strategy and cost model

### Why the experiment is shaped this way

The strategy above exists to make one number — mean RHAE on the public 25 with
an official competition-mode scorecard — as cheap, bounded, and defensible as
possible:

- **Local-first play, replay to score.** Games are played against the official
  local engine (free, ~2,000 FPS, no idle timeouts), and the recorded actions
  are replayed through the live API onto ONE competition-mode scorecard with
  per-step verification. Model tokens are spent exactly once; the scorecard is
  a verified re-execution, not a second attempt (the Retrodict-established
  pattern, disclosed in `docs/disclosures.md`).
- **Budget caps are the cost-control primitive.** Every run carries hard caps
  (dollars, tokens, actions per level and per game, wall clock) and dies
  cleanly at any of them with partial results recorded. Whatever a weak model
  does — loop, stall, wander — the worst case per game is the cap, so the
  campaign's worst case is `25 x cost cap`, known before the first token is
  spent.
- **Pilot before campaign.** Three games x the served matrix (~$10–25/game)
  buy the per-model cost-per-level curve before committing to 25 games, and
  decide which one or two models get the full campaign.
- **Single run per game, everything disclosed.** No cherry-picking; restarts
  and replays are disclosed. One scorecard per campaign.

### Cost model (estimates until the pilot replaces them)

Anchor: no world-model harness has published open-weight token counts, and
NVIDIA AVO published no token/cost data at all — so the reference profile is
Retrodict, the closest architecture with full accounting (652.5M input +
7.4M output tokens, 7,703 actions, 25/25 games with GPT-5.6 Sol; AVO's action
count, 6,624, is within ~15% of Retrodict's, suggesting comparable interaction
volume). Two structural corrections when moving that profile to Cerebras:

1. **No cache discount.** Retrodict's $654 leaned on 96% of input billed at
   10% price. Cerebras prompt caching helps latency and the uncached-TPM
   bucket only — every input token bills at full rate, and a
   conversation-growing harness re-pays its history every call (~45–55k input
   per invocation as the conversation approaches the 90k reset threshold, so
   roughly $0.05/invocation on gemma-4-31b).
2. **A 31B open-weight model is not GPT-5.6 Sol.** It needs more invocations
   per unit progress and will fail levels the closed models finished — which
   raises tokens per attempted level but lowers totals, because runs end at
   budget caps instead of playing 78M-token marathons to a win.

| Scenario | Assumption | Estimate |
|---|---|---|
| Retrodict's exact token profile on gemma-4-31b ($0.99/$1.49 per Mtok) | 652.5M in + 7.4M out, uncached | ~$657 campaign (~$26/game) |
| **Campaign at default caps, gemma-4-31b** | $25/game cap binds after ~460–500 invocations (~23M input tok/game) | **$400–625, hard ceiling $625** |
| Campaign at default caps, gpt-oss-120b ($0.35/$0.75) | same profile, ~1/3 the token price | ~$230 naive-transfer; same $625 ceiling, expected well under |
| **Campaign at default caps, qwen3.8-27b** (price unknown until served) | 27B on the shared tier: gemma-band pricing gives gemma's numbers, gpt-oss-band gives ~$230 | **$250–650 expected; the $625 ceiling holds at any price** |
| Pilot (3 games x 2 models) | default caps | **$60–150, ceiling $150** |
| Qwen 3.8 pilot (3 games, once served) | default caps | ~$10–75, exact the day catalog prices land |
| Uncapped, weak-model pessimistic | 2–3x Sol's interactions, no caps | $1.3k–2k — what the caps exist to prevent |

Non-dollar constraints that bind first:

- **Throughput:** paid-tier TPM is 500K (gemma) / 1M (gpt-oss), so a
  Retrodict-scale campaign is ≥ ~22h/≥ ~11h of wall clock minimum if run
  serially — per-token latency (~0.13s round-trips measured in the live smoke)
  is not the bottleneck, the rate limiter is.
- **Timing:** per Cerebras' customer email of 2026-08-21, gemma-4-31b leaves
  the shared tier ~2026-09-03 (replaced by Qwen 3.8 27B). A gemma campaign
  must therefore finish before then (go/no-go by ~2026-09-01 given its ≥ ~22 h
  wall-clock floor); the qwen3.8 pilot starts the day `arc3cb models` first
  lists the id, and its unknowns (price, context, TPM tier, reasoning
  controls, image support) all resolve from the public catalog and one
  `arc3cb probe` — the cost brackets above collapse to real numbers the same
  day, before any campaign tokens are spent.
- Free-trial keys (65k context, 5 rpm, 30K TPM) cannot run this harness;
  see [API keys and tiers](#api-keys-and-tiers).

### How measurement replaces estimation

Every model call is metered at the source: `usage.jsonl` records per-call
prompt/completion/cached/reasoning tokens, dollars (from `configs/models.yaml`
prices), latency, and retry attempts; `metrics.json` rolls a run up (tokens,
cost, actions, levels, stop reason, provisional RHAE). `arc3cb results`
regenerates every table in `docs/results.md` from those artifacts alone —
nothing is hand-entered, so the first pilot run starts overwriting this
section's estimates with measured numbers, and the estimates above are
falsifiable against preserved traces.

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
- `cerebras-smoke.yml` — manual dispatch, plus an automatic run on pushes that
  touch the transport, the smoke script, or the workflow itself; uses the
  repository secret **`CEREBRAS_API_KEY`** (or `C7`, both hold a Cerebras key)
  to enumerate the live served catalog, report experiment-matrix availability
  (including the qwen3.8 probe), and round-trip one tiny completion through
  the retrying transport with usage accounting. Verified green on 2026-08-22:
  auth, catalog (gpt-oss-120b + gemma-4-31b), and a 0.13s completion probe.

## Credits and licenses

This repo is MIT (see LICENSE, © 2026 MIT Critical Data).

- First inspiration: **NVIDIA AVO reaches 100 on ARC-AGI-3** — <https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents>
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
