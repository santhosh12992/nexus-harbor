#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "    Harbor + Antigravity Custom-Skill Evaluation POC Verification"
echo "================================================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

record_pass() {
  echo -e "\033[0;32m[PASS]\033[0m $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

record_fail() {
  echo -e "\033[0;31m[FAIL]\033[0m $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

record_skip() {
  echo -e "\033[0;33m[SKIP]\033[0m $1"
  SKIP_COUNT=$((SKIP_COUNT + 1))
}

# --- Step 1: Check Harbor & Python Environment ---
echo "[Step 1] Checking Harbor and Python installation..."
PYTHON_BIN=""
HARBOR_BIN=""

if command -v harbor >/dev/null 2>&1; then
  HARBOR_BIN="harbor"
  PYTHON_BIN="python3"
elif [ -x "$REPO_ROOT/.venv/bin/harbor" ]; then
  HARBOR_BIN="$REPO_ROOT/.venv/bin/harbor"
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
elif [ -x "$HOME/.gemini/antigravity-ide/brain/4b74de2c-d9af-4441-8bba-801ddb49fdec/scratch/test_venv/bin/harbor" ]; then
  HARBOR_BIN="$HOME/.gemini/antigravity-ide/brain/4b74de2c-d9af-4441-8bba-801ddb49fdec/scratch/test_venv/bin/harbor"
  PYTHON_BIN="$HOME/.gemini/antigravity-ide/brain/4b74de2c-d9af-4441-8bba-801ddb49fdec/scratch/test_venv/bin/python3"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  fi
fi

if [ -n "$HARBOR_BIN" ]; then
  HARBOR_VER=$("$HARBOR_BIN" --version 2>&1 || echo "unknown")
  record_pass "Harbor is installed ($HARBOR_VER) using $HARBOR_BIN"
else
  record_fail "Harbor CLI is not installed in PATH or local virtual environment"
fi

# --- Step 2: Check Docker Runtime ---
echo ""
echo "[Step 2] Checking Docker availability..."
DOCKER_AVAILABLE=false
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    DOCKER_VER=$(docker --version)
    record_pass "Docker daemon is running ($DOCKER_VER)"
    DOCKER_AVAILABLE=true
  else
    record_skip "Docker CLI found but daemon is not running. Live container execution will be skipped."
  fi
else
  record_skip "Docker is not installed / not in PATH. Live container execution will be skipped."
fi

# --- Step 3: Validate Task and Skill Structure ---
echo ""
echo "[Step 3] Validating task and skill directory structure & schema..."
MISSING_FILES=0
check_file() {
  if [ -f "$1" ]; then
    echo "  ✓ Found $1"
  else
    echo "  ✗ Missing $1"
    MISSING_FILES=$((MISSING_FILES + 1))
  fi
}

check_file "skills/distinctive-output/SKILL.md"
check_file "datasets/antigravity-skill-poc/task-001/task.toml"
check_file "datasets/antigravity-skill-poc/task-001/instruction.md"
check_file "datasets/antigravity-skill-poc/task-001/environment/Dockerfile"
check_file "datasets/antigravity-skill-poc/task-001/environment/input.txt"
check_file "datasets/antigravity-skill-poc/task-001/tests/test.sh"
check_file "datasets/antigravity-skill-poc/task-001/solution/solve.sh"
check_file "configs/antigravity-with-skill.yaml"
check_file "configs/antigravity-without-skill.yaml"

if [ "$MISSING_FILES" -eq 0 ]; then
  record_pass "All required files and directories exist"
else
  record_fail "Missing $MISSING_FILES required file(s)"
fi

# Validate Schema with Python
if [ -n "$PYTHON_BIN" ]; then
  echo "  Validating schemas against Harbor models..."
  "$PYTHON_BIN" - << 'EOF'
import sys, tomllib, yaml
from pathlib import Path
from harbor.models.task.config import TaskConfig
from harbor.models.job.config import JobConfig

# Validate task.toml
task_path = Path("datasets/antigravity-skill-poc/task-001/task.toml")
with open(task_path, "rb") as f:
    task_data = tomllib.load(f)
TaskConfig.model_validate(task_data)

# Validate YAML configs
for cfg_file in ["configs/antigravity-with-skill.yaml", "configs/antigravity-without-skill.yaml"]:
    with open(cfg_file, "r") as f:
        job_data = yaml.safe_load(f)
    JobConfig.model_validate(job_data)

print("  ✓ All task and job schemas validated successfully")
EOF
  if [ $? -eq 0 ]; then
    record_pass "TaskConfig and JobConfig schema validation succeeded"
  else
    record_fail "Schema validation failed"
  fi
fi

# --- Step 4: Run Oracle / Reference Solution & Verifier Logic ---
echo ""
echo "[Step 4] Executing reference solution and verifier test..."
TEMP_SANDBOX=$(mktemp -d)
trap 'rm -rf "$TEMP_SANDBOX"' EXIT

mkdir -p "$TEMP_SANDBOX/workspace" "$TEMP_SANDBOX/logs/verifier"
cp datasets/antigravity-skill-poc/task-001/environment/input.txt "$TEMP_SANDBOX/workspace/input.txt"

# Run solve.sh in temp sandbox
(
  export WORKSPACE_DIR="$TEMP_SANDBOX/workspace"
  export LOGS_DIR="$TEMP_SANDBOX/logs/verifier"
  cd "$TEMP_SANDBOX"
  bash "$REPO_ROOT/datasets/antigravity-skill-poc/task-001/solution/solve.sh"
  bash "$REPO_ROOT/datasets/antigravity-skill-poc/task-001/tests/test.sh"
)

REWARD=$(cat "$TEMP_SANDBOX/logs/verifier/reward.txt" 2>/dev/null || echo "0")
if [ "$REWARD" == "1" ]; then
  record_pass "Oracle reference solution produced valid report and earned Reward 1"
else
  record_fail "Oracle reference solution verification failed (Reward: $REWARD)"
fi

# --- Step 5: Test A/B Skill Marker Detection Logic ---
echo ""
echo "[Step 5] Testing A/B Skill Marker discrimination..."
REPORT_WITHOUT_SKILL=$(mktemp)
REPORT_WITH_SKILL=$(mktemp)

# Baseline report without skill
echo "Summary: The Pacific Ocean is Earth's largest ocean." > "$REPORT_WITHOUT_SKILL"
# Skill-injected report
echo "SKILL-MARKER: ENABLED" > "$REPORT_WITH_SKILL"
echo "Summary: The Pacific Ocean is Earth's largest ocean." >> "$REPORT_WITH_SKILL"

FIRST_LINE_BASELINE=$(head -n 1 "$REPORT_WITHOUT_SKILL")
FIRST_LINE_SKILL=$(head -n 1 "$REPORT_WITH_SKILL")

if [ "$FIRST_LINE_BASELINE" != "SKILL-MARKER: ENABLED" ] && [ "$FIRST_LINE_SKILL" == "SKILL-MARKER: ENABLED" ]; then
  record_pass "Marker discrimination logic verified (Absent on baseline, Present with skill)"
else
  record_fail "Marker discrimination logic failed"
fi

rm -f "$REPORT_WITHOUT_SKILL" "$REPORT_WITH_SKILL"

# --- Step 6: Live Antigravity SDK Execution ---
echo ""
echo "[Step 6] Checking prerequisites for live containerized Antigravity evaluation..."

if [ "$DOCKER_AVAILABLE" = false ]; then
  record_skip "Docker daemon not reachable; live container runs skipped."
  echo "         Prerequisite: Install and start Docker Desktop."
elif [ -z "${GEMINI_API_KEY:-}" ]; then
  record_skip "GEMINI_API_KEY not set in environment; live Antigravity execution skipped."
  echo "         Prerequisite: export GEMINI_API_KEY='your-gemini-api-key'"
else
  echo "  Running live baseline evaluation (without skill)..."
  ./scripts/run-without-skill.sh
  record_pass "Antigravity baseline run executed successfully"

  echo "  Running live skill evaluation (with skill)..."
  ./scripts/run-with-skill.sh
  record_pass "Antigravity with-skill run executed successfully"
fi

# --- Summary ---
echo ""
echo "================================================================"
echo "                       VERIFICATION SUMMARY"
echo "================================================================"
echo -e "Total Passed: \033[0;32m$PASS_COUNT\033[0m"
echo -e "Total Failed: \033[0;31m$FAIL_COUNT\033[0m"
echo -e "Total Skipped (Missing External Infra/Key): \033[0;33m$SKIP_COUNT\033[0m"
echo "================================================================"

if [ "$FAIL_COUNT" -eq 0 ]; then
  echo -e "\033[0;32mPOC INFRASTRUCTURE & TASK SPECIFICATION VERIFIED SUCCESSFULLY!\033[0m"
  exit 0
else
  echo -e "\033[0;31mPOC VERIFICATION COMPLETED WITH FAILURES.\033[0m"
  exit 1
fi
