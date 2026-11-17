#!/bin/bash
#SBATCH --job-name=terra1-chainlit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=app_terra1_chainlit-%j.log

set -euo pipefail
source "$HOME/miniforge3/bin/activate"
conda activate qwen35
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-9B}"
export LORA_PATH="${LORA_PATH:-/projects/u6zu/Terrafuture/Terra1/results/qwen35_9b_lora128_phase1/final/}"
export PDF_CHAT_WORKDIR="${PDF_CHAT_WORKDIR:-/projects/u6zu/Terrafuture/Terra1/pdfworkdir}"
export PDF_INDEX_DIR="${PDF_INDEX_DIR:-data/phase1_vector_db}"
export PDF_DATABASE_ROOT="${PDF_DATABASE_ROOT:-data/Phase1PDFBase}"
export STANDARDS_INDEX_DIR="${STANDARDS_INDEX_DIR:-data/phase1_standards_db}"
export STANDARDS_DATABASE_ROOT="${STANDARDS_DATABASE_ROOT:-data/Phase1Standards}"
export PDF_MAX_HOPS="${PDF_MAX_HOPS:-4}"
export PDF_TOP_N="${PDF_TOP_N:-3}"
export HF_HOME="${HF_HOME:-/projects/u6zu/Terrafuture/ckpt}"
export PYTHONUNBUFFERED=1
python -m chainlit run app_chainlit.py --host "${CHAINLIT_HOST:-127.0.0.1}" --port "${CHAINLIT_PORT:-8000}"
