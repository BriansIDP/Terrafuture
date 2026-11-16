#!/bin/bash
#SBATCH --job-name=qwen35-dpo-stage3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=qwen35-dpo-stage3-%j.log

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate qwen35
export HF_HOME=/projects/u6zu/Terrafuture/ckpt

python train_qwen35_dpo_stage3.py \
    --model Qwen/Qwen3.5-9B \
    --rollouts rollouts/qwen35_dual_max2_papers6_standards6.json \
    --questions data/phase1_stage3_case_a_questions.json \
    --paper-database-root data/Phase1PDFBase \
    --standards-database-root data/Phase1Standards \
    --output-dir results/qwen35_trajectory_dpo_stage3_max2_papers6_standards6 \
    --page-mode selected \
    --content-mode hybrid \
    --inline-images \
    --min-score-gap 0.5 \
    --max-pairs-per-qid 10 \
    --beta 0.1 \
    --epochs 3 \
    --learning-rate 1e-5 \
    --warmup 10 \
    --lora-r 128 \
    --lora-alpha 256
