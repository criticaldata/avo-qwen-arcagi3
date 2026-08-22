"""Sandboxed execution of agent-authored Python.

Agent code runs in a dedicated *containment* virtualenv that has numpy, scipy
and networkx installed but — provably — no game-engine packages. The proof is
``containment.json``: at setup (and again at the start of every local run) we
attempt to import each forbidden engine module inside the venv and record that
every one of them fails. If any succeeds, the run aborts: a local run whose
agent code could import the engine would not be a credible result.

The subprocess gets a minimal environment, a CPU/address-space rlimit, a wall
clock timeout, and its stdout/stderr truncated before being fed back to the
model. This is containment against accidental cheating and runaway code, not a
security boundary against a hostile adversary; runs execute on infrastructure
you control.
"""

from __future__ import annotations

import argparse
import json
import resource
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Engine/package names that must NOT be importable from agent code. The list is
# extended by configs/containment.yaml (forbidden_modules) at run time.
DEFAULT_FORBIDDEN_MODULES = [
    "arc_agi",
    "arc_agi_3",
    "arcengine",
    "arcprize",
    "arc3",
    "agents",  # the official ARC-AGI-3-Agents package name
]

TRUNCATION_NOTICE = "\n...[output truncated at {limit} characters]"


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    def render(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("[python tool] TIMED OUT")
        parts.append(f"[python tool] exit code {self.exit_code}")
        if self.stdout:
            parts.append("--- stdout ---\n" + self.stdout)
        if self.stderr:
            parts.append("--- stderr ---\n" + self.stderr)
        if not self.stdout and not self.stderr:
            parts.append("(no output — use print() to see results)")
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_NOTICE.format(limit=limit)


class Sandbox:
    def __init__(
        self,
        venv_dir: str | Path,
        workdir: str | Path,
        log_path: str | Path,
        timeout_s: int = 60,
        max_output_chars: int = 20000,
        memory_mb: int = 2048,
    ):
        # .absolute(), not .resolve(): resolving the venv's python symlink would
        # escape the venv and lose its site-packages.
        self.python = Path(venv_dir).absolute() / "bin" / "python"
        self.workdir = Path(workdir).absolute()
        self.log_path = Path(log_path).resolve()
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars
        self.memory_mb = memory_mb
        self._n = 0
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not self.python.exists():
            raise FileNotFoundError(
                f"containment venv python not found at {self.python}; "
                "run scripts/setup_containment_venv.sh first"
            )

    def _limits(self) -> None:
        mem = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = self.timeout_s
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))

    def run(self, code: str) -> SandboxResult:
        self._n += 1
        script = self.workdir / f"tool_{self._n:04d}.py"
        script.write_text(code)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.workdir),
            "LOG_PATH": str(self.log_path),
            "PYTHONUNBUFFERED": "1",
        }
        try:
            proc = subprocess.run(
                [str(self.python), str(script)],
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                preexec_fn=self._limits,
            )
            # A CPU-bound loop trips RLIMIT_CPU (SIGXCPU) before the wall-clock
            # timeout; report both paths as a timeout to the model.
            cpu_killed = proc.returncode == -signal.SIGXCPU
            return SandboxResult(
                stdout=_truncate(proc.stdout, self.max_output_chars),
                stderr=_truncate(proc.stderr, self.max_output_chars),
                exit_code=proc.returncode,
                timed_out=cpu_killed,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                stdout=_truncate((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), self.max_output_chars),
                stderr=_truncate((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""), self.max_output_chars),
                exit_code=-1,
                timed_out=True,
            )


class ContainmentError(RuntimeError):
    pass


def verify_containment(
    venv_dir: str | Path,
    forbidden_modules: list[str] | None = None,
    out_path: str | Path | None = None,
) -> dict:
    """Prove that engine imports fail inside the containment venv.

    Returns the containment report and writes it to ``out_path`` (JSON). Raises
    ContainmentError if any forbidden module imports successfully.
    """
    python = Path(venv_dir).absolute() / "bin" / "python"
    if not python.exists():
        raise ContainmentError(f"containment venv python not found at {python}")
    modules = list(forbidden_modules or DEFAULT_FORBIDDEN_MODULES)
    report: dict = {"venv": str(Path(venv_dir).resolve()), "python": str(python), "modules": {}}
    breached = []
    for mod in modules:
        proc = subprocess.run(
            [str(python), "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin"},
        )
        ok = proc.returncode != 0
        report["modules"][mod] = {
            "import_failed": ok,
            "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
        }
        if not ok:
            breached.append(mod)
    for mod in ("numpy", "scipy", "networkx"):
        proc = subprocess.run(
            [str(python), "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            timeout=120,
            env={"PATH": "/usr/bin:/bin"},
        )
        report["modules"][mod] = {"import_failed": proc.returncode != 0, "required": True}
    report["contained"] = not breached
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    if breached:
        raise ContainmentError(
            f"containment breach: forbidden module(s) importable in venv: {', '.join(breached)}"
        )
    missing = [
        m
        for m in ("numpy", "scipy", "networkx")
        if report["modules"][m]["import_failed"]
    ]
    if missing:
        raise ContainmentError(
            f"containment venv is missing required packages: {', '.join(missing)}; "
            "re-run scripts/setup_containment_venv.sh"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify sandbox containment")
    p.add_argument("--venv", default=".containment-venv")
    p.add_argument("--out", default="containment.json")
    p.add_argument("--extra-forbidden", nargs="*", default=[])
    args = p.parse_args(argv)
    try:
        verify_containment(
            args.venv,
            DEFAULT_FORBIDDEN_MODULES + list(args.extra_forbidden),
            args.out,
        )
    except ContainmentError as e:
        print(f"CONTAINMENT FAILED: {e}", file=sys.stderr)
        return 1
    print(f"containment verified; report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
