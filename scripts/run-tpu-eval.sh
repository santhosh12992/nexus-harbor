#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

if command -v harbor >/dev/null 2>&1; then
  HARBOR_BIN="harbor"
elif [ -x "$REPO_ROOT/.venv/bin/harbor" ]; then
  HARBOR_BIN="$REPO_ROOT/.venv/bin/harbor"
else
  echo "Error: 'harbor' CLI not found. Please install Harbor via 'pip install -r requirements.txt'." >&2
  exit 1
fi

echo "=== Running Harbor TPU Hardware Speedup Benchmark (MaxKernel on TPU v6e) ==="
"$HARBOR_BIN" run -c configs/maxkernel-tpu-eval.yaml "$@"
