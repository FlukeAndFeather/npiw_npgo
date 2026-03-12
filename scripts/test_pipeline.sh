#!/usr/bin/env bash
# Test the full pipeline locally, running process jobs in parallel (mimics SLURM array).
# Usage: ./scripts/test_pipeline.sh [start_year] [end_year]
set -euo pipefail

START_YEAR=${1:-2004}
END_YEAR=${2:-2006}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROC_DIR="$ROOT/outputs/processed_chunks"
OUTPUT="$ROOT/outputs/npiw_${START_YEAR}_${END_YEAR}.nc"

years=()
for (( y=START_YEAR; y<=END_YEAR; y++ )); do
    years+=("$y")
done

# ── Process (parallel) ────────────────────────────────────────────────────────
echo "=== PROCESS: ${years[*]} ==="
pids=()
for year in "${years[@]}"; do
    python "$ROOT/scripts/process_year.py" --year "$year" --processed-dir "$PROC_DIR" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# ── Combine ───────────────────────────────────────────────────────────────────
echo "=== COMBINE: $START_YEAR–$END_YEAR ==="
python "$ROOT/scripts/combine.py" \
    --processed-dir "$PROC_DIR" \
    --output "$OUTPUT" \
    --years "$START_YEAR" "$END_YEAR"

echo ""
echo "Done: $OUTPUT"
