#!/bin/bash
#SBATCH --job-name=qwen35-dual-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --array=0-127
#SBATCH --output=dual-seeds-%A_%a.log

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate qwen35
export HF_HOME=/projects/u6zu/Terrafuture/ckpt

counts=(6 9)
hops=(2 3 4 5)
combination=$((SLURM_ARRAY_TASK_ID / 8 % 4))
max_hops=${hops[$((SLURM_ARRAY_TASK_ID / 32))]}
paper_top_n=${counts[$((combination / 2))]}
standards_top_n=${counts[$((combination % 2))]}
seed=$((SLURM_ARRAY_TASK_ID % 8 + 1))
output="auto_rollouts/qwen35_dual_auto_${max_hops}_papers${paper_top_n}_standards${standards_top_n}_seed${seed}.json"

python run_qwen35_multihop_inference_dual.py \
    --questions /projects/u6zu/Terrafuture/Terra1/data/phase1_stage3_type_a_questions.json \
    --output "$output" \
    --paper-index-dir data/phase1_vector_db \
    --standards-index-dir data/phase1_standards_db \
    --paper-database-root data/Phase1PDFBase \
    --standards-database-root data/Phase1Standards \
    --content-mode hybrid \
    --max-hops "$max_hops" \
    --paper-top-n "$paper_top_n" \
    --standards-top-n "$standards_top_n" \
    --max-evidence-pages 128 \
    --gh200-patch-optimization \
    --max-new-tokens 2048 \
    --do-sample \
    --seed "$seed"

python LLM_as_a_judge.py --resultpath "$output"
