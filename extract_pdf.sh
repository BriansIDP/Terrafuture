
# set -euo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate qwen35

mkdir -p data/pdf_database

for pdf in /projects/u6zu/Terrafuture/Terra1/data/Phase1Dataset/S*.pdf; do
    [ -f "$pdf" ] || continue

    pdf_name=$(basename "$pdf" .pdf)

    python prepare_pdf.py \
        --pdf "$pdf" \
        --out-dir "data/Phase1Standards/$pdf_name" \
        --extract-content
done