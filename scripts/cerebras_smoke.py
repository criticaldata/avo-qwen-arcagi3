#!/usr/bin/env python3
"""Live smoke test of the Cerebras shared-inference endpoint.

Run locally (CEREBRAS_API_KEY in .env) or in CI, where the key is supplied by
the `C7` repository secret. Verifies:
  1. GET /v1/models responds and enumerates the served catalog;
  2. which models from our experiment matrix are actually served right now
     (including probing for a qwen3.8-27b style id);
  3. a tiny chat completion round-trips with usage accounting.

Prints served model ids and usage numbers. Never prints the key.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arc3cb.config import load_env_file  # noqa: E402
from arc3cb.transport import CerebrasTransport, ModelConfig, UsageMeter  # noqa: E402

# The experiment matrix from the project brief; live /v1/models wins over this.
MATRIX_CANDIDATES = [
    "gemma-4-31b",
    "gpt-oss-120b",
    "zai-glm-4.7",
    "qwen-3-235b-a22b-instruct-2507",
    "llama3.1-8b",
]
QWEN38_PATTERN = re.compile(r"qwen[-_.]?3\.?8.*27b|qwen3\.8", re.IGNORECASE)

# Cheap models preferred for the completion probe.
PROBE_PREFERENCE = ["llama3.1-8b", "llama-3.3-70b", "gpt-oss-120b", "qwen-3-32b"]


def main() -> int:
    load_env_file()
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("CEREBRAS_API_KEY is not set (locally: .env; CI: the C7 secret)", file=sys.stderr)
        return 2

    meter = UsageMeter()
    boot = CerebrasTransport(ModelConfig(id="bootstrap"), meter)
    models = boot.list_models()
    ids = sorted(m.get("id", "?") for m in models)
    print(f"served models ({len(ids)}):")
    for m in sorted(models, key=lambda m: m.get("id", "")):
        extras = {
            k: v
            for k, v in m.items()
            if k not in ("id", "object", "created", "owned_by")
        }
        print(f"  - {m.get('id')}" + (f"  {extras}" if extras else ""))

    print("\nexperiment matrix availability:")
    for cand in MATRIX_CANDIDATES:
        mark = "SERVED" if cand in ids else "not served"
        print(f"  {cand:40s} {mark}")
    qwen38 = [i for i in ids if QWEN38_PATTERN.search(i)]
    if qwen38:
        print(f"  qwen3.8-27b-style id DETECTED: {', '.join(qwen38)} — promote to top of matrix")
    else:
        print("  qwen3.8-27b-style id                     not served yet")

    probe_id = os.environ.get("SMOKE_MODEL") or next(
        (m for m in PROBE_PREFERENCE if m in ids), ids[0] if ids else None
    )
    if not probe_id:
        print("no models served — nothing to probe", file=sys.stderr)
        return 1
    print(f"\ncompletion probe on {probe_id}:")
    t = CerebrasTransport(
        ModelConfig(id=probe_id, temperature=0.0, max_output_tokens=2048), meter
    )
    result = t.chat(
        [{"role": "user", "content": "Reply with exactly: SMOKE_OK"}], purpose="smoke"
    )
    print(f"  reply: {result.text.strip()[:200]!r}")
    print(f"  usage: prompt={result.prompt_tokens} completion={result.completion_tokens}")
    print(f"  latency: {result.latency_s:.2f}s  finish_reason: {result.finish_reason}")
    if "SMOKE_OK" not in result.text:
        print("  note: model did not echo the token verbatim; transport still verified OK")
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
