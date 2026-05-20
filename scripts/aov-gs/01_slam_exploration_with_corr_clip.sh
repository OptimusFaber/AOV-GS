#!/usr/bin/env bash
# Wrapper: same as 01_slam_exploration.sh with MASK_COLLECTOR=corrclip
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/01_slam_exploration.sh" "$1" "$2" "$3" "$4" "$5" "${6:-run_0}" corrclip
