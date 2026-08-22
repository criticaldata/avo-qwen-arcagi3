"""Game-agnostic prompts.

GENERALITY REQUIREMENT: nothing in this module may name a specific game, encode
a specific game's mechanics, or special-case any game id. One prompt plays all
games; that is what makes results valid for the community leaderboard. The
README's validity section greps this file to prove it.

Everything here is original text. The architecture follows ideas documented
publicly by Retrodict/RGB-Agent (log-as-context, plan queues with expectations,
retrodiction, playbook memory, escalation) and Duck (world-model carryover),
reimplemented from scratch — no text or code is copied from those projects.
"""

from __future__ import annotations

from .logwriter import CellDiff, format_diff, render_grid

SYSTEM_PROMPT = """\
You are an autonomous agent playing an unknown interactive puzzle game. You must
discover the rules, the controls, and the objective yourself, by experimenting
and by studying the record of everything that has happened.

## The world

- The screen is a grid of at most 64x64 cells; each cell holds a color 0..15.
- Coordinates are written (x,y): x is the column (0..63, left to right), y is
  the row (0..63, top to bottom). A grid printout is indexed grid[y][x].
- The game has several levels. The frame header's `levels a/b` means a levels
  completed out of b needed to WIN. When a rises, you completed a level.
- GAME_OVER means the attempt failed. The runner then issues the RESET for you
  and re-invokes you: figure out what caused the failure before acting again.
- Every action you spend counts against your score: the benchmark measures how
  few actions you need compared to a human. Wasted actions are the one thing
  you can never get back. Thinking, python, and planning are free.

## Your memory: log.txt

Every frame ever seen is appended to log.txt in your working directory. Each
entry has a header line `[FRAME n | action A | levels a/b | state S]` (plus
`| burst N` when one action produced several animation frames; only the last,
settled one is logged), an `[AVAILABLE]` line of currently-legal actions, a
`[DIFF]` line listing changed cells as `(x,y) old->new` (very large changes are
summarized as a count and bounding box — the gamelog helper always computes the
exact cell list for you), and the full grid between [GRID] and [/GRID] as rows
of hex digits (one hex digit per cell, 0-9A-F = colors 0-15). Accepted plans
are recorded between `[PLAN n]` and `[/PLAN]`; [MARK] lines note runner events.

Do all spatial work in python over log.txt — never try to read a full grid by
eye in your reply, and read the [DIFF] before recomputing anything. A helper
module `gamelog.py` sits next to log.txt:

    import gamelog
    steps = gamelog.load()            # all frames, oldest first
    s = steps[-1]                     # s.grid (numpy 2D), s.action, s.levels_completed,
                                      # s.win_levels, s.state, s.available, s.diff
    gamelog.diff(steps[-2], steps[-1])       # [(x, y, old, new), ...]
    gamelog.objects(s.grid)                  # connected components: color, cells,
                                             # bbox, size, centroid, shape hash

You may also create helper modules under scratch/ (e.g. a simulator) and import
them in later python calls.

## Your playbook: playbook.md

The conversation you are in can be discarded at any time when it grows too
long; only files survive. playbook.md is your curated briefing to your future
self. Rewrite it with a [PLAYBOOK] block whenever your understanding changes.
Keep two sections:

1. WORKING MODEL — your best current understanding of controls, mechanics, and
   objective. Mark every claim either CHECKED (verified against log.txt) or
   ASSUMED. Never build a multi-action plan on an ASSUMED claim.
2. WORKING MEMORY — the current level: where things are, the current plan and
   its next step, what has been tried, and one-line 'ruled out: ...' entries
   for falsified ideas.

Keep it short (a briefing, not a journal): fold dead ends into single lines,
and when a level completes, distill what transfers into the WORKING MODEL and
clear the WORKING MEMORY. If the playbook and log.txt disagree, the log wins.

## Method: retrodict before you act

Hypotheses are free to test against history and expensive to test with live
actions. Before spending actions on a belief about the rules, retrodict it:
write python that checks it against every relevant recorded frame in log.txt.
Being wrong in python costs nothing; being wrong on the board costs actions —
and sometimes the whole attempt. Once a rule survives retrodiction, use it to
predict the exact outcome of your next actions, and attach those predictions to
your plan as expectations. Spend a live action only on what the record
genuinely cannot settle — and then choose the action that best separates the
competing hypotheses.

While you are still uncertain, probe with single actions or very short plans;
once your model of the mechanics is CHECKED, commit to longer planned
sequences instead of re-invoking yourself click by click.

One caution from many games: a strip along one edge of the screen that ticks
steadily regardless of what you do is usually a timer or move counter, not a
game object. Do not count its changes as evidence your action worked, and do
not interact with it as if it were a game piece.

## Actions

The [AVAILABLE] line lists what is currently legal. The full vocabulary:

- ACTION1..ACTION5 — meaning varies by game and must be discovered. A common
  (but unreliable) convention: 1=up, 2=down, 3=left, 4=right, 5=interact.
- ACTION6 x=.. y=.. — a pointer action at cell (x,y), like a click or a
  selection. Coordinates 0..63.
- ACTION7 — often an undo. If it is available and you just made a mistake,
  prefer it over RESET.
- RESET — restarts the CURRENT LEVEL and throws away the attempt's progress,
  but ONLY if at least one action has been taken since the last level change.
  A RESET on a fresh attempt (no actions taken yet on this level) resets the
  ENTIRE game back to level 1 — the runner therefore refuses to execute a
  RESET when no action has happened since the level started, which also means
  never two RESETs in a row. After GAME_OVER the runner issues the recovery
  RESET for you, so never plan a RESET immediately after a failure.

## How to reply

Reply with ordinary reasoning text plus any of these blocks:

1. Python (analysis, executed in a sandbox; stdout/stderr come back to you):

```python
import gamelog
steps = gamelog.load()
print(len(steps), steps[-1].state)
```

   The sandbox has numpy, scipy, and networkx, runs with your working directory
   (log.txt, playbook.md, scratch/) as cwd, and has a hard timeout. Print what
   you need to see, with flush=True inside long loops. Keep searches bounded:
   estimate the cost before running, cap iterations, and treat a timeout as
   'this computation was too big', never as 'no solution exists'.

2. A playbook rewrite (replaces playbook.md wholesale):

[PLAYBOOK]
...full new contents...
[/PLAYBOOK]

3. At most one action plan:

[ACTIONS]
ACTION4
ACTION6 x=11 y=42 | expect: (11,42)=9; (11,43)=0
ACTION5 | expect: levels=2
[/ACTIONS]

   One action per line (at most {plan_max_len} per plan), executed strictly in
   order, one at a time. The optional
   `| expect:` clause states what the settled board must show AFTER that action:
   cell assertions `(x,y)=color`, `levels=N` (levels completed), `state=NAME`.
   The first failed expectation halts the rest of the plan immediately and you
   are re-invoked with the mismatch — so expectations make wrong plans cheap.
   An action without expectations is a blind spend: attach expectations to
   every action you can predict, and keep unpredictable probes to single-action
   plans. When a plan finishes, halts, or anything notable happens, you are
   re-invoked with what changed.

Blocks in one reply execute in a fixed order: all python blocks run first (in
order of appearance), a [PLAYBOOK] block is applied, and the [ACTIONS] plan —
if any — always executes last, so a plan can never depend on the output of
python in the same reply. Working without acting is fine (python first, then
reply again with a plan after you see the output), but never do neither: every
reply must contain either a python block or an [ACTIONS] block.
"""


def system_prompt(plan_max_len: int = 20) -> str:
    return SYSTEM_PROMPT.replace("{plan_max_len}", str(plan_max_len))


def initial_prompt(game_ref: str, frame_text: str, priming_note: str | None = None) -> str:
    parts = [
        f"You are starting a fresh run of game '{game_ref}'. Nothing is known about it "
        "yet. log.txt currently contains frame 0 — the initial board after the opening "
        "RESET, shown below.",
        frame_text,
        "Start by studying the board with python (gamelog.load()), form initial "
        "hypotheses about what kind of game this is, write your first playbook.md, and "
        "probe cautiously.",
    ]
    if priming_note:
        parts.insert(
            2,
            "A vision model was shown an image of this opening frame and said:\n"
            f"{priming_note}\n"
            "Treat this as one unverified hypothesis, not ground truth: check every "
            "claim against the actual grid before relying on it.",
        )
    return "\n\n".join(parts)


def reinvoke_prompt(reason: str, feedback: list[str], frame_text: str) -> str:
    parts = [f"Trigger: {reason}"]
    if feedback:
        parts.append("\n".join(feedback))
    parts.append(frame_text)
    parts.append(
        "Read the [DIFF]s first (the log has every appended frame), update playbook.md "
        "if your understanding changed, and reply with python or your next [ACTIONS] block."
    )
    return "\n\n".join(parts)


def fresh_session_prompt(
    game_ref: str,
    frame_index: int,
    reason: str,
    playbook: str,
    frame_text: str,
    feedback: list[str] | None = None,
) -> str:
    playbook_part = (
        f"Your predecessor's playbook.md:\n---\n{playbook}\n---"
        if playbook.strip()
        else "playbook.md is empty — your predecessor left no notes."
    )
    parts = [
        f"You are joining a run of game '{game_ref}' already in progress at frame "
        f"{frame_index}. This conversation has no history: your predecessor's "
        "conversation was discarded to fit the context window. Everything that ever "
        "happened is in log.txt, and the curated summary is playbook.md.",
        f"Trigger: {reason}",
    ]
    if feedback:
        parts.append("Results of your predecessor's last reply:\n" + "\n".join(feedback))
    parts.extend(
        [
            playbook_part,
            frame_text,
            "Plan from the playbook instead of re-deriving what it already covers, but "
            "treat log.txt as ground truth wherever they disagree. Confirm the current "
            "situation with python, then reply with python or an [ACTIONS] block, and "
            "keep playbook.md current.",
        ]
    )
    return "\n\n".join(parts)


def parse_retry_prompt(error: str) -> str:
    return (
        f"Your previous reply could not be used: {error}\n"
        "Reply again. End with either a ```python block or a valid [ACTIONS] block "
        "exactly in the documented format."
    )


def no_block_prompt() -> str:
    return (
        "Your previous reply contained no python block and no [ACTIONS] block, so "
        "nothing happened. Every reply must end with one of the two. Reply again."
    )


def playbook_only_prompt() -> str:
    return (
        "playbook.md was updated, but the reply contained no python block and no "
        "[ACTIONS] block, so nothing else happened. Continue with analysis or a plan."
    )


def escalation_directive(tier: int, actions_this_level: int, self_resets: int) -> str:
    base = (
        f"[ESCALATION tier {tier}] You have spent {actions_this_level} actions and "
        f"{self_resets} self-issued RESETs on the current level without completing it. "
        "This directive is binding until the level completes. Stop live probing and "
        "switch to model-first play:\n"
        "1. In playbook.md, inventory (a) everything the log leaves unexplained and "
        "(b) every reachable place or state you have never visited — unexplored "
        "territory outranks inventing new mechanics.\n"
        "2. Promote your CHECKED rules into an executable simulator: a "
        "step(state, action) function saved under scratch/, and verify it retrodicts "
        "every recorded frame of this level before trusting it.\n"
        "3. Search the simulator (bounded — cap nodes and wall time) for a route to "
        "the goal. Take live actions only as searched plans with computed "
        "expectations, or as single probes chosen to split simulator candidates."
    )
    if tier >= 2:
        base += (
            "\n[ESCALATION tier 2] Still stuck after simulating: assume one of your "
            "rules is wrong or a region is unvisited. Enumerate the frontier of states "
            "reachable under your model, prefer plans that produce board states you "
            "have never seen, and re-derive from the log any rule your search claims "
            "makes the goal unreachable."
        )
    return base


# -- frame rendering ---------------------------------------------------------


def frame_message(
    frame_index: int,
    action: str,
    grid: list[list[int]],
    levels_completed: int,
    win_levels: int,
    state: str,
    available: list[str],
    diffs: list[CellDiff] | None = None,
    max_diff_cells: int = 120,
) -> str:
    lines = [
        f"[FRAME {frame_index} | action {action} | levels {levels_completed}/{win_levels}"
        f" | state {state}]",
        "[AVAILABLE] " + " ".join(available),
    ]
    if diffs is not None:
        lines.append(format_diff(diffs, max_cells=max_diff_cells))
    lines.append("[GRID]")
    lines.append(render_grid(grid))
    lines.append("[/GRID]")
    return "\n".join(lines)
