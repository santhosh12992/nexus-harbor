#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"

# --- ALWAYS CREATE REWARD.TXT FIRST ---
mkdir -p "$LOGS_DIR"
echo "0" > "${LOGS_DIR}/reward.txt"

# Locate the best optimized kernel produced by MaxKernel multi-agent workflow
if [ ! -f "${WORKSPACE_DIR}/optimized_kernel.py" ]; then
  if [ -f "${WORKSPACE_DIR}/state.json" ]; then
    BEST_PATH=$(python3 -c "import json; s=json.load(open('${WORKSPACE_DIR}/state.json')); print(s.get('best_code_path', ''))" 2>/dev/null || true)
    if [ -n "$BEST_PATH" ] && [ -f "$BEST_PATH" ]; then
      cp "$BEST_PATH" "${WORKSPACE_DIR}/optimized_kernel.py"
    elif [ -n "$BEST_PATH" ] && [ -f "${WORKSPACE_DIR}/${BEST_PATH}" ]; then
      cp "${WORKSPACE_DIR}/${BEST_PATH}" "${WORKSPACE_DIR}/optimized_kernel.py"
    fi
  fi
  if [ ! -f "${WORKSPACE_DIR}/optimized_kernel.py" ]; then
    LATEST_ITER=$(find "${WORKSPACE_DIR}" -name "optimized.py" -o -name "optimized_kernel.py" 2>/dev/null | sort | tail -n 1)
    if [ -n "$LATEST_ITER" ] && [ -f "$LATEST_ITER" ]; then
      cp "$LATEST_ITER" "${WORKSPACE_DIR}/optimized_kernel.py"
    fi
  fi
fi

# If agent didn't produce any optimized kernel, finish cleanly with score 0
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
cp -r "${WORKSPACE_DIR}"/iter* /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/state.json" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}"/iteration_*_kernel.py /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/optimized_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/baseline_kernel.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/base.py" /logs/artifacts/ 2>/dev/null || true
cp "${WORKSPACE_DIR}/profiler_history.json" /logs/artifacts/ 2>/dev/null || true
cp "${SCRIPT_DIR}/../instruction.md" /logs/artifacts/ 2>/dev/null || true
cp "${LOGS_DIR}/xprof_telemetry.json" /logs/artifacts/ 2>/dev/null || true

REWARD=$(cat "${LOGS_DIR}/reward.txt" 2>/dev/null || echo "0")
echo "Verifier completed with final reward: ${REWARD}"
exit 0
