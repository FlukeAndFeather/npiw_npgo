#!/usr/bin/env bash
# Smoke-test the pipeline on a single week of data.
# Usage: ./scripts/test_single_week.sh [year] [start_date] [end_date]
set -euo pipefail

YEAR=${1:-2010}
START=${2:-"$YEAR-06-01"}
END=${3:-"$YEAR-06-07"}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROC_DIR="$ROOT/outputs/test_chunks"
OUTPUT="$ROOT/outputs/npiw_test_week_${START}_${END}.nc"

echo "=== PROCESS: $START to $END ==="
python "$ROOT/scripts/process_year.py" \
    --year "$YEAR" \
    --processed-dir "$PROC_DIR" \
    --start "$START" \
    --end "$END"

echo "=== COMBINE ==="
python "$ROOT/scripts/combine.py" \
    --processed-dir "$PROC_DIR" \
    --output "$OUTPUT" \
    --years "$YEAR" "$YEAR"

echo ""
echo "Done: $OUTPUT"
