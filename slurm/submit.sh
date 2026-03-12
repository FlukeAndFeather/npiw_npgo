#!/usr/bin/env bash
# Submit the full NPIW processing pipeline as chained SLURM jobs.
# Usage: ./slurm/submit.sh [start_year] [end_year]
#
# Dependency chain:
#   process array  →  combine
set -euo pipefail

START_YEAR=${1:-2004}
END_YEAR=${2:-2020}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Edit these for your HPC ───────────────────────────────────────────────────
PARTITION="compute"
PROC_DIR="/scratch/$USER/npiw_processed"
OUTPUT="/scratch/$USER/npiw_${START_YEAR}_${END_YEAR}.nc"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$ROOT/logs"

# ── Step 1: process array (download + compute + interpolate per year) ─────────
PROCESS_JID=$(sbatch --parsable \
    --job-name=npiw_process \
    --partition="$PARTITION" \
    --array="${START_YEAR}-${END_YEAR}%4" \
    --time=04:00:00 \
    --mem=32G \
    --cpus-per-task=1 \
    --output="$ROOT/logs/process_%a.out" \
    --error="$ROOT/logs/process_%a.err" \
    --wrap="python $ROOT/scripts/process_year.py \
        --year \$SLURM_ARRAY_TASK_ID \
        --processed-dir $PROC_DIR"
)
echo "Submitted process array: job $PROCESS_JID (years $START_YEAR–$END_YEAR)"

# ── Step 2: combine (after all process jobs succeed) ──────────────────────────
COMBINE_JID=$(sbatch --parsable \
    --job-name=npiw_combine \
    --partition="$PARTITION" \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=1 \
    --output="$ROOT/logs/combine.out" \
    --error="$ROOT/logs/combine.err" \
    --dependency=afterok:"$PROCESS_JID" \
    --wrap="python $ROOT/scripts/combine.py \
        --processed-dir $PROC_DIR \
        --output $OUTPUT \
        --years $START_YEAR $END_YEAR"
)
echo "Submitted combine:       job $COMBINE_JID (depends on $PROCESS_JID)"

echo ""
echo "Monitor with: squeue -u $USER"
echo "Output will be written to: $OUTPUT"
