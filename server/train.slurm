#!/usr/bin/env bash
#SBATCH --job-name=poncan-yolo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm-%j.log

set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

export PYTHON_COMMAND="${PYTHON_COMMAND:-python3.11}"
export PIPELINE_VENV="${PIPELINE_VENV:-${project_dir}/.venv}"

./run_pipeline.sh train --device 0

