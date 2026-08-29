#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"

# --- ALWAYS CREATE REWARD.TXT FIRST ---
mkdir -p "$LOGS_DIR"
echo "0" > "${LOGS_DIR}/reward.txt"

# If agent didn't produce the kernel, finish cleanly with score 0
if [ ! -f "${WORKSPACE_DIR}/optimized_kernel.py" ]; then
  echo "FAIL: Expected optimized kernel file missing"
  exit 0
fi

export WORKSPACE_DIR="$WORKSPACE_DIR"
export LOGS_DIR="$LOGS_DIR"

PYTHON_BIN="python3"
if [ -x "/usr/local/google/home/smuralik/nexus_hands_on_ws/nexus-harbor/.venv/bin/python3" ]; then
  PYTHON_BIN="/usr/local/google/home/smuralik/nexus_hands_on_ws/nexus-harbor/.venv/bin/python3"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"
fi

if [ "${USE_REAL_TPU:-1}" = "1" ]; then
  echo "[Mode] Running on LIVE Physical TPU Hardware (${TPU_NAME:-maxkernel-v6e-1})..."
  "$PYTHON_BIN" "${SCRIPT_DIR}/tpu_remote_profiler.py" || true
else
  echo "[Mode] Running in Local Evaluation Simulation Mode..."
  "$PYTHON_BIN" "${SCRIPT_DIR}/profiler_runner.py" || true
fi

# Export all artifacts to /logs/artifacts for Harbor UI Artifacts Tab display
mkdir -p /logs/artifacts
cp "${WORKSPACE_DIR}/optimized_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/baseline_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/profiler_history.json" /logs/artifacts/ 2>/dev/null || true
cp "${LOGS_DIR}/xprof_telemetry.json" /logs/artifacts/ 2>/dev/null || true

REWARD=$(cat "${LOGS_DIR}/reward.txt" 2>/dev/null || echo "0")
echo "Verifier completed with final reward: ${REWARD}"
exit 0
