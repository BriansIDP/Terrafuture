#!/bin/bash
#SBATCH --job-name=qwen35-videomme
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --output=videomme_highres-%j.log

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
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
# output=results/outputs/qwen35_pdfqa_with_retriever_max1_top8_page16_standardtest.json
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

# python run_qwen35_multihop_inference.py \
#     --questions $testdata \
#     --output $output \
#     --index-dir data/phase1_standards_db_16pages \
#     --database-root data/Phase1Standards \
#     --content-mode hybrid \
#     --num-workers 2 \
#     --prefetch-factor 1 \
#     --max-hops 1 \
#     --top-n 8 \
#     --max-evidence-pages 256 \
#     --gh200-patch-optimization \
#     --max-new-tokens 2048 \
#     --ckpt $ckpt \
#     # --disable_pdf \

# ckpt=/projects/u6zu/Terrafuture/Terra1/exp/qwen35_trajectory_dpo_max3_top3/final
ckpt=no
testdata=data/preprocess/phase1_questions_modified.json
# testdata=data/phase1_stage3_case_a_questions.json
# output=rollouts/qwen35_pdfqa_with_retriever_max1_top8_seed${1}.json
output=rollouts/qwen35_pdfqa_stage3_with_retriever_max3_top3.json
python run_qwen35_multihop_inference.py \
    --questions $testdata \
    --output $output \
    --index-dir data/phase1_vector_db \
    --database-root data/Phase1PDFBase \
    --content-mode hybrid \
    --num-workers 2 \
    --prefetch-factor 1 \
    --max-hops 3 \
    --top-n 3 \
    --max-evidence-pages 128 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --ckpt $ckpt \
    # --do-sample \
    # --seed ${1} \
    # --disable_pdf \

python LLM_as_a_judge.py --resultpath $output

# # Dual-source inference on the same test set.
# testdata=data/phase1_stage3_case_a_questions.json
# output=rollouts/qwen35_pdfqa_dual_max2_top6.json
# python run_qwen35_multihop_inference_dual.py \
#     --questions "$testdata" \
#     --output "$output" \
#     --paper-index-dir data/phase1_vector_db \
#     --standards-index-dir data/phase1_standards_db \
#     --paper-database-root data/Phase1PDFBase \
#     --standards-database-root data/Phase1Standards \
#     --content-mode hybrid \
#     --max-hops 2 \
#     --top-n 6 \
#     --max-evidence-pages 128 \
#     --gh200-patch-optimization \
#     --max-new-tokens 2048 \
#     --do-sample \
#     --seed "${1}"

# python LLM_as_a_judge.py --resultpath "$output"
