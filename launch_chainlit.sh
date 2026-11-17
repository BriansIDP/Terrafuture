#!/bin/bash
#SBATCH --job-name=terra1-chainlit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
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
export PDF_MAX_HOPS="${PDF_MAX_HOPS:-10}"
export PDF_TOP_N="${PDF_TOP_N:-3}"
export HF_HOME="${HF_HOME:-/projects/u6zu/Terrafuture/ckpt}"
export PYTHONUNBUFFERED=1
if ! command -v cloudflared >/dev/null 2>&1; then
    echo 'cloudflared is required to create a public link.' >&2
    exit 1
fi

chainlit_pid=''
tunnel_pid=''
cleanup() {
    [[ -z "$tunnel_pid" ]] || kill "$tunnel_pid" 2>/dev/null || true
    [[ -z "$chainlit_pid" ]] || kill "$chainlit_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python -m chainlit run app_chainlit.py --host "${CHAINLIT_HOST:-127.0.0.1}" --port "${CHAINLIT_PORT:-8000}" &
chainlit_pid=$!

echo 'Starting public tunnel; the trycloudflare.com link will appear below.'
cloudflared tunnel --url "http://127.0.0.1:${CHAINLIT_PORT:-8000}" &
tunnel_pid=$!

# Stop the other process if either the app or the tunnel exits.
wait -n "$chainlit_pid" "$tunnel_pid"
