#!/bin/bash
#SBATCH --job-name=qwen35_grpo_stage3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=qwen35_grpo_stage3-%j.log

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

traindata=/projects/u6zu/Terrafuture/Terra1/data/phase1_stage3_type_a_questions.json

python train_qwen35_grpo_stage3.py \
    --model "$model" \
    --questions "$traindata" \
    --paper-database-root data/Phase1PDFBase \
    --standards-database-root data/Phase1Standards \
    --output-dir exp/qwen35_trajectory_grpo_stage3 \
    --judge-device cuda:0 \
    --beta 0.1 \
    --epochs 3 \
    --learning-rate 1e-5 \
    --group-size 4 \
    --max-hops 3 \
    --paper-top-n 3 \
    --standards-top-n 3 \
    --verbosity 1 \
    --lora-r 128 \
    --lora-alpha 256
