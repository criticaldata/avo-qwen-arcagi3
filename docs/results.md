# Results

Status: **harness complete; campaign not yet run.** Every number below either
comes from an official competition-mode scorecard (linked) or is explicitly
labeled self-reported/provisional. All tables in this file are regenerated
from run artifacts with `arc3cb results runs/` (or `scripts/make_results.py`)
— nothing is hand-entered.

## Campaign results (to be filled by the campaign)

| model | mean RHAE (public 25) | levels solved /183 | total actions | total tokens | total cost USD | scorecard |
|---|---|---|---|---|---|---|
| gpt-oss-120b (high) via Cerebras | — | — | — | — | — | — |
| gemma-4-31b (high) via Cerebras | — | — | — | — | — | — |
| qwen3.8-27b via Cerebras (once served, ETA 2026-09-03) | — | — | — | — | — | — |

Per-game breakdowns and per-game costs land here from `arc3cb results` once
runs exist (metrics.json + usage.jsonl per run under `runs/`).

## Pilot (former preview games ls20, ft09, vc33)

| game | model | RHAE | levels | actions | tokens | cost USD | wall clock |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

## Comparison context (as of 2026-08-22)

Sources: community-leaderboard repo submissions + arcprize.org community page,
project READMEs, vendor blogs. "Self-reported" = no arcprize.org scorecard.

| system | model | RHAE (public 25) | cost | provenance |
|---|---|---|---|---|
| Tycho | Claude Opus 5 | 100.0 | $2,986 | community leaderboard, competition-mode scorecard |
| Retrodict v2.0 | GPT-5.6 Sol (max) | 99.86 | $654 | community leaderboard, competition-mode scorecard |
| NVIDIA AVO | Claude Opus 5 | 100.0 | n/a | **self-reported** (vendor blog, own task interface; not on any ARC Prize leaderboard) |
| baseline1 v1.6 | GPT-5.6 Sol (xhigh) | 98.97 | $400 | community leaderboard |
| Claude Opus 5 (High), bare model | — | 30.16 (semi-private) | n/a | ARC Prize verified base-LLM evaluation |
| **Polyphony Agent** | **Qwen3.6 (open-weight, self-hosted)** | **19.8** | $115 | community leaderboard — best open-weight entry |
| **Tufa Labs Duck** | **Qwen 3.6 27B FP8 (open-weight)** | **~1.6 self-measured** (1.21 Kaggle Milestone-1 score) | n/a | Kaggle Milestone 1 winner, single-GPU constraint; no arcprize.org scorecard published |

Notes on the open-weight rows: Duck ran under Kaggle's single-GPU, 45-min-per-
game constraints — its 1.6 is a mean over 20 passes x 25 games measured by its
own framework, not an official scorecard. Polyphony is the only open-weight
entry on the community leaderboard. This project's contribution is the
unconstrained-token version of that experiment: a Retrodict-style world-model
harness with an open-weight model at Cerebras speed.

## Reproduction

```bash
# per-game pilot run (local mode, then replay to a scorecard):
arc3cb run <game> --model gpt-oss-120b --mode local --cost-cap 25
# campaign scorecard:
python scripts/replay_runs.py --competition --open-only
python scripts/replay_runs.py --card-id <id> --keep-open --runs runs/<...>
python scripts/replay_runs.py --close-card <id>
# tables:
arc3cb results runs/
```

See README for setup and docs/disclosures.md for the disclosure policy.
