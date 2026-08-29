#!/bin/bash
set -euo pipefail
TARGET_DIR="${1:-${WORKSPACE_DIR:-/workspace}}"
mkdir -p "$TARGET_DIR"
cp "$(dirname "$0")/optimized_kernel.py" "${TARGET_DIR}/optimized_kernel.py"
