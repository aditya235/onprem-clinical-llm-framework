#!/usr/bin/env bash
# ------------------------------------------------------------------
# Start / resume the pipeline in the background on the on-premise server.
# Safe to run any time: finished steps are skipped, unfinished ones resume
# from their checkpoints.
#
#   bash scripts/run_pipeline.sh                      # resume, incl. LoRA training
#   PIPELINE_ARGS="" bash scripts/run_pipeline.sh     # resume WITHOUT training
#   PIPELINE_ARGS="--train --tag v2 --max-length 8192 --gen-tokens 8192 --batch-size 1" bash scripts/run_pipeline.sh
#
# Environment variables:
#   PY              python interpreter (default: python3)
#   IE_OUT_DIR      output folder (default: ~/mimic_project/ie_pipeline)
#   MIMIC_DB        DuckDB file with MIMIC-III (default: ~/mimic_project/mimic3.duckdb)
#   VLLM_URL        OpenAI-compatible vLLM endpoint (default: http://localhost:8001/v1)
#
# Progress:  tail -f "$IE_OUT_DIR/logs/pipeline_$(date +%Y%m%d).log"
# Status:    $PY src/mimic_pipeline.py --status
# Stop:      pkill -f '[m]imic_pipeline.py'    (checkpoints are kept; just run again)
# ------------------------------------------------------------------
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO/src/mimic_pipeline.py"
PY="${PY:-python3}"
ARGS="${PIPELINE_ARGS---train}"
OUT_DIR="${IE_OUT_DIR:-$HOME/mimic_project/ie_pipeline}"
LOGDIR="$OUT_DIR/logs"; mkdir -p "$LOGDIR"
OUT="$LOGDIR/console_$(date +%Y%m%d_%H%M%S).log"

# The [m] trick stops pgrep from matching this check itself
if pgrep -f "[m]imic_pipeline.py" >/dev/null; then
  echo "Pipeline is already running - not starting a second copy."; exit 0
fi

# Live console log, faster CUDA memory reuse, offline model loading once downloaded
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cd "$REPO" || exit 1
nohup "$PY" "$SCRIPT" $ARGS >> "$OUT" 2>&1 &
echo "Pipeline started (pid $!) with args: $ARGS"
echo "Console log: $OUT"
