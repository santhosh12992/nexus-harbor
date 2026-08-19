#!/bin/bash
set -euo pipefail

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Locate harbor CLI binary
if command -v harbor >/dev/null 2>&1; then
  HARBOR_BIN="harbor"
elif [ -x "$REPO_ROOT/.venv/bin/harbor" ]; then
  HARBOR_BIN="$REPO_ROOT/.venv/bin/harbor"
elif [ -x "$HOME/.gemini/antigravity-ide/brain/4b74de2c-d9af-4441-8bba-801ddb49fdec/scratch/test_venv/bin/harbor" ]; then
  HARBOR_BIN="$HOME/.gemini/antigravity-ide/brain/4b74de2c-d9af-4441-8bba-801ddb49fdec/scratch/test_venv/bin/harbor"
else
  echo "Error: 'harbor' CLI not found. Please install Harbor via 'pip install -r requirements.txt' or 'pip install harbor'." >&2
  exit 1
fi

echo "=== Running Harbor with Custom Skill (distinctive-output) ==="
"$HARBOR_BIN" run -c configs/antigravity-with-skill.yaml "$@"
