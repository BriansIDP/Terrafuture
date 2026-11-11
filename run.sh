#!/bin/bash
#SBATCH --job-name=qwen35-videomme
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=videomme_highres-%j.log

set -euo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate qwen35

export HF_HOME=/projects/u6zu/Terrafuture/ckpt

# python prepare_pdf.py \
#   --pdf /projects/u6zu/Terrafuture/Terra1/data/icml2026_vsalmonnS_cameraready.pdf \
#   --out-dir data/vss_rendered \
#   --extract-content

ckpt=/projects/u6zu/Terrafuture/Terra1/results/qwen35_9b_lora512_phase1/final
# ckpt=no
output=results/outputs/qwen35_pdfqa_lora512.json

python run_qwen35_pdfqa.py \
    --questions data/preprocess/phase1_questions_modified.json \
    --output $output \
    --page-mode all \
    --content-mode hybrid \
    --inline-images \
    --num-workers 0 \
    --prefetch-factor 1 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --ckpt $ckpt \
    # --disable_pdf \

python LLM_as_a_judge.py --resultpath results/outputs/qwen35_pdfqa_lora512.json


ckpt=/projects/u6zu/Terrafuture/Terra1/results/qwen35_9b_lora128_phase1/final
# ckpt=no
output=results/outputs/qwen35_pdfqa_lora128.json

python run_qwen35_pdfqa.py \
    --questions data/preprocess/phase1_questions_modified.json \
    --output $output \
    --page-mode all \
    --content-mode hybrid \
    --inline-images \
    --num-workers 0 \
    --prefetch-factor 1 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --ckpt $ckpt \
    # --disable_pdf \

python LLM_as_a_judge.py --resultpath results/outputs/qwen35_pdfqa_lora128.json

ckpt=/projects/u6zu/Terrafuture/Terra1/results/qwen35_lora_phase1/final
# ckpt=no
output=results/outputs/qwen35_pdfqa.json

python run_qwen35_pdfqa.py \
    --questions data/preprocess/phase1_questions_modified.json \
    --output $output \
    --page-mode all \
    --content-mode hybrid \
    --inline-images \
    --num-workers 0 \
    --prefetch-factor 1 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --do-sample \
    --ckpt $ckpt \
    # --disable_pdf \

python LLM_as_a_judge.py --resultpath results/outputs/qwen35_pdfqa.json