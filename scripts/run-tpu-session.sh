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

if [ -d "$REPO_ROOT/.venv" ]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
  export VIRTUAL_ENV="$REPO_ROOT/.venv"
fi

PYTHON_BIN="python3"
if [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
fi

MODE="${1:-session}"

case "$MODE" in
  session)
    echo "=== Running MaxKernel Live Multi-Iteration TPU Hardware Session ==="
    USE_REAL_TPU="${USE_REAL_TPU:-1}" \
    WORKSPACE_DIR="$REPO_ROOT/datasets/tpu-kernel-speedup/task-001/environment" \
    LOGS_DIR="/tmp/verifier_logs" \
    "$PYTHON_BIN" datasets/tpu-kernel-speedup/task-001/tests/maxkernel_live_session.py
    ;;
  agent)
    echo "=== Running Gemini 2.5 LLM Autonomous Agent on TPU Hardware ==="
    USE_REAL_TPU="${USE_REAL_TPU:-1}" \
    "$PYTHON_BIN" datasets/tpu-kernel-speedup/task-001/tests/gemini_maxkernel_live_agent.py
    ;;
  test)
    echo "=== Running Direct TPU Hardware Verifier ==="
    USE_REAL_TPU="${USE_REAL_TPU:-1}" \
    WORKSPACE_DIR="$REPO_ROOT/datasets/tpu-kernel-speedup/task-001/solution" \
    LOGS_DIR="/tmp/verifier_logs" \
    bash datasets/tpu-kernel-speedup/task-001/tests/test.sh
    ;;
  sim)
    echo "=== Running Local Verifier in CPU Simulation Mode ==="
    USE_REAL_TPU=0 \
    WORKSPACE_DIR="$REPO_ROOT/datasets/tpu-kernel-speedup/task-001/solution" \
    LOGS_DIR="/tmp/verifier_logs" \
    "$PYTHON_BIN" datasets/tpu-kernel-speedup/task-001/tests/profiler_runner.py
    ;;
  *)
    echo "Usage: $0 [session|agent|test|sim]"
    exit 1
    ;;
esac
