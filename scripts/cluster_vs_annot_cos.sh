#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM batch script: run the clusterAnalysis pipeline
#
# Usage:
#   sbatch clusterAnalysis/scripts/run_analysis.sh
#   sbatch --export=CONFIG=my_config.yaml,RUN_NAME=exp01 clusterAnalysis/scripts/run_analysis.sh
#
# Environment variables (override defaults):
#   CONFIG    — path to YAML config file
#   RUN_NAME  — run name (overrides config's output.run_name)
#   LOG_LEVEL — DEBUG | INFO | WARNING (default: INFO)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cluster_analysis
#SBATCH --output=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/clusterAnalysis/logs/run_%j.out
#SBATCH --error=/srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/clusterAnalysis/logs/run_%j.out
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=shared-gpu
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate

# ── Working directory ─────────────────────────────────────────────────────────
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

# mkdir -p /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper/clusterAnalysis/logs

# ── Parameters ────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-clusterAnalysis/configs/cluster_vs_annot_cos.yaml}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "========================================"
echo "  clusterAnalysis pipeline"
echo "  SLURM Job ID : $SLURM_JOB_ID"
echo "  Config       : $CONFIG"
echo "  Run name     : $RUN_NAME"
echo "  Log level    : $LOG_LEVEL"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ── Run ───────────────────────────────────────────────────────────────────────
python -m clusterAnalysis.src.run_analysis \
    --config   "$CONFIG" \
    --run-name "$RUN_NAME" \
    --log-level "$LOG_LEVEL"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Pipeline finished successfully"
else
    echo "  Pipeline finished with errors (exit code $EXIT_CODE)"
    echo "  Check clusterAnalysis/results/${RUN_NAME}/run_summary.json for details"
fi
echo "========================================"

exit $EXIT_CODE
