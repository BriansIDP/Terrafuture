#!/bin/bash
#SBATCH --job-name=app-terra1-launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=app_terra1-%j.log

set -euo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate qwen35

# cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-9B}"
# export QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.8-27B}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export PDF_CHAT_WORKDIR=/projects/u6zu/Terrafuture/Terra1/pdfworkdir
export LORA_PATH=/projects/u6zu/Terrafuture/Terra1/results/qwen35_9b_lora128_phase1/final/
export PDF_INDEX_DIR=data/phase1_vector_db
export PDF_DATABASE_ROOT=data/Phase1PDFBase
export PDF_TOP_N=3
export PDF_CANDIDATES=100
export PDF_RERANK_TOP=50
export GRADIO_SHARE=True
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False

export HF_HOME=/projects/u6zu/Terrafuture/ckpt

# python app.py
cd /projects/u6zu/Terrafuture/Terra1

python -u app_with_retrieval.py