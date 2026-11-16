#!/bin/bash
#SBATCH --job-name=qwen35_train_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=qwen35_train_phase1-%j.log

set -euo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate qwen35

model=Qwen/Qwen3.5-9B
# model=Qwen/Qwen3.5-27B
export HF_HOME=/projects/u6zu/Terrafuture/ckpt

# python train_qwen35.py \
#     --model $model \
#     --questions data/phase1_questions.json \
#     --output-dir results/qwen35_9b_lora128_phase1_longer \
#     --page-mode all \
#     --content-mode hybrid \
#     --epochs 10 \
#     --learning-rate 5e-5 \
#     --no-inline-images \
#     --gradient-accumulation-steps 1 \
#     --lora-r 128 \
#     --lora-alpha 256 \

traindata=data/preprocess/phase1_questions_modified.json

python train_qwen35_dpo.py \
    --rollouts rollouts/qwen35_pdfqa_with_retriever_max3_top3.json \
    --questions $traindata \
    --database-root data/Phase1PDFBase \
    --output-dir results/qwen35_trajectory_dpo_max3_top3 \
    --min-score-gap 0.5 \
    --max-pairs-per-qid 10 \
    --beta 0.1 \
    --epochs 3 \
    --learning-rate 1e-5 \
    --warmup 10 \
    --lora-r 128 \
    --lora-alpha 256 \
