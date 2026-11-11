#!/bin/bash
#SBATCH --job-name=qwen35-videomme
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --output=videomme_highres-%j.log

set -euo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate qwen35

# python prepare_pdf.py \
#   --pdf /projects/u6zu/Terrafuture/Terra1/data/icml2026_vsalmonnS_cameraready.pdf \
#   --out-dir data/vss_rendered \
#   --extract-content

export HF_HOME=/projects/u6zu/Terrafuture/ckpt
# python retriever.py build \
#     --database data/Phase1Standards \
#     --index-dir data/phase1_standards_db \
#     --embedding-model Qwen/Qwen3-Embedding-4B \
#     --max-words 350 \
#     --overlap 80 \
#     --parent-pages 1 \
#     --batch-size 4

# ckpt=/projects/u6zu/Terrafuture/Terra1/results/qwen35_9b_lora128_phase1/final
ckpt=no
output=results/outputs/qwen35_pdfqa_with_retriever_max5_standardtest.json
# testdata=data/preprocess/phase1_questions_modified.json
testdata=data/preprocess/phase1_standard_test.json
# # python run_qwen35_pdfqa_retrieval.py \
# python run_qwen35_pdfqa_multiturn_retrieval.py \
#     --questions $testdata \
#     --output $output \
#     --page-mode selected \
#     --content-mode hybrid \
#     --inline-images \
#     --num-workers 2 \
#     --prefetch-factor 1 \
#     --gh200-patch-optimization \
#     --max-new-tokens 2048 \
#     --index-dir data/phase1_vector_db \
#     --database-root data/Phase1PDFBase \
#     --disable_pdf \
#     --ckpt $ckpt \

python run_qwen35_multihop_inference.py \
    --questions $testdata \
    --output $output \
    --index-dir data/phase1_standards_db \
    --database-root data/Phase1Standards \
    --content-mode hybrid \
    --num-workers 2 \
    --prefetch-factor 1 \
    --max-hops 3 \
    --top-n 2 \
    --max-evidence-pages 18 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --ckpt $ckpt \
    # --disable_pdf \

python LLM_as_a_judge.py --resultpath $output
