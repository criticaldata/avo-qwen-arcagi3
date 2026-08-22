# Disclosures

Policy: single run per game in the campaign — no cherry-picking. Anything that
deviates gets a dated entry here before results are shared anywhere.

Must be disclosed:

1. **Restarts.** Any run restarted for infrastructure reasons (API outage,
   container loss, harness crash): game, date, reason, and whether any
   harness/prompt/config change happened between attempts.
2. **Replays.** Which games' scorecard entries were produced by replaying
   locally recorded actions through the live API (scripts/replay_runs.py),
   and that every replayed step was verified against the recording
   (levels_completed + state; --strict-boards where used). A replay that
   diverged and was aborted is also listed.
3. **Version drift.** The game versions (full game_ids) actually played, as
   recorded in each run's run_config.json / metrics.json — ARC Prize bumps
   game versions over time.
4. **Config changes mid-campaign.** None are allowed without restarting the
   campaign; if it ever happens, it is recorded here.
5. **Self-reported status.** All numbers are labeled self-reported until ARC
   Prize independently verifies them; official competition-mode scorecard
   URLs are the only score source for arc-agi-3.

## Log

- 2026-08-22: repository created; no campaign runs yet. The offline smoke
  evidence in this repo (mock-game integration tests; a scripted-model run
  against the real downloaded ls20 engine) exercised the loop, sandbox, and
  scoring — these are engineering tests, not results, and are never reported
  as RHAE.
