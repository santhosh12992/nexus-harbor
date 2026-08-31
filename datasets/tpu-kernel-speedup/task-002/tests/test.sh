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

echo "=== Running TPU Kernel Profiling & Verification ==="
if [ "${USE_REAL_TPU:-0}" = "1" ] && [ -f "${SCRIPT_DIR}/tpu_remote_profiler.py" ]; then
  echo "[Mode] Running on LIVE Physical TPU Hardware (${TPU_NAME:-maxkernel-v6e-1})..."
  "$PYTHON_BIN" "${SCRIPT_DIR}/tpu_remote_profiler.py" || true
else
  echo "[Mode] Running in Local JAX CPU Profiler Simulation Mode..."
  if ! python3 /workspace/profile_kernel.py; then
    echo "FAIL: Optimized kernel failed verification or profiling"
    exit 0
  fi
  echo "SUCCESS: Optimized kernel verified and profiled successfully"
  echo "1" > "${LOGS_DIR}/reward.txt"
fi

# Export all artifacts to /logs/artifacts for Harbor UI Artifacts Tab display
mkdir -p /logs/artifacts
cp "${WORKSPACE_DIR}"/iteration_*_kernel.py /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/optimized_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/baseline_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/profiler_history.json" /logs/artifacts/ 2>/dev/null || true
cp "${SCRIPT_DIR}/../instruction.md" /logs/artifacts/ 2>/dev/null || true
cp "${LOGS_DIR}/xprof_telemetry.json" /logs/artifacts/ 2>/dev/null || true

REWARD=$(cat "${LOGS_DIR}/reward.txt" 2>/dev/null || echo "0")
echo "Verifier completed with final reward: ${REWARD}"
exit 0
