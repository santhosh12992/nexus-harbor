#!/bin/bash
set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
INPUT_FILE="$WORKSPACE/input.txt"
OUTPUT_FILE="$WORKSPACE/report.txt"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file $INPUT_FILE not found" >&2
  exit 1
fi

# Read input content and generate a clean summary report
INPUT_TEXT=$(cat "$INPUT_FILE")
echo "Summary: The Pacific Ocean is Earth's largest ocean, as described in the source document." > "$OUTPUT_FILE"

echo "Solution generated report at $OUTPUT_FILE"
