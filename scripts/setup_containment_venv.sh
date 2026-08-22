#!/usr/bin/env bash
# Create the containment venv that agent-authored Python runs in, then prove
# containment (engine packages must NOT be importable) into containment.json.
#
# Usage: scripts/setup_containment_venv.sh [VENV_DIR]
set -euo pipefail

VENV_DIR="${1:-.containment-venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet numpy scipy networkx
# Remove the installer itself so agent code cannot fetch the engine at runtime.
"$VENV_DIR/bin/python" -m pip uninstall --quiet -y pip setuptools wheel || true

# Verify containment with the project venv's interpreter if available, else
# whatever python3 has arc3cb installed.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
"$PY" -m arc3cb.tools --venv "$VENV_DIR" --out containment.json
echo "containment venv ready at $VENV_DIR"
