#!/bin/bash
set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
WORKSPACE="${WORKSPACE_DIR:-/workspace}"
REPORT_FILE="$WORKSPACE/report.txt"

# Ensure verifier logs directory exists
mkdir -p "$LOGS"

# Default to reward 0 on failure
echo "0" > "$LOGS/reward.txt"

echo "=== Running Task Verifier ==="

# Check 1: Verify report.txt exists
if [ ! -f "$REPORT_FILE" ]; then
  echo "FAIL: Expected report file does not exist at $REPORT_FILE"
  exit 1
fi

# Check 2: Verify report.txt is non-empty
if [ ! -s "$REPORT_FILE" ]; then
  echo "FAIL: Report file $REPORT_FILE is empty"
  exit 1
fi

echo "--- Report Contents ---"
cat "$REPORT_FILE"
echo ""
echo "-----------------------"

# Check 3: Verify content summarizes input accurately (case-insensitive keyword check)
CONTENT=$(cat "$REPORT_FILE" | tr '[:upper:]' '[:lower:]')

if [[ "$CONTENT" != *"pacific"* ]] || [[ "$CONTENT" != *"ocean"* ]]; then
  echo "FAIL: Report does not contain relevant summary terms ('Pacific', 'ocean')"
  exit 1
fi

# All criteria passed
echo "1" > "$LOGS/reward.txt"
echo "SUCCESS: Report file verified successfully with valid summary content."
exit 0
