#!/bin/bash
#SBATCH --job-name=planner-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --array=0-23
#SBATCH --output=auto_rollouts/planner-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"

# Paired comparison: 2 planners x 2 hop budgets x 2 evidence-page budgets
# x 3 sampling seeds. Each pair uses the same questions and retrieval settings.
# The planner can stop early, so these are maximums, not required searches.
task=${SLURM_ARRAY_TASK_ID:-${1:-}}
if [[ ! $task =~ ^[0-9]+$ ]] || (( task > 23 )); then
    echo 'Use sbatch run_planner_comparison.sh or bash run_planner_comparison.sh TASK_ID (0-23)' >&2
    exit 2
fi

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-qwen35}"
export HF_HOME=${HF_HOME:-/projects/u6zu/Terrafuture/ckpt}

methods=(text pdf)
standards_hops=(2 4)
paper_hops=(1 2)
page_budgets=(12 24)
method=${methods[$((task / 12))]}
hop_index=$((task / 6 % 2))
standards=${standards_hops[$hop_index]}
papers=${paper_hops[$hop_index]}
pages=${page_budgets[$((task / 3 % 2))]}
seed=$((task % 3 + 1))

mkdir -p auto_rollouts
name="${method}_standards${standards}_papers${papers}_pages${pages}_seed${seed}"
output="auto_rollouts/${name}.json"
if [[ -e $output || -e auto_rollouts/${name}_scored.json ]]; then
    echo "Output already exists for $name; move it before rerunning." >&2
    exit 1
fi

script=run_qwen35_multihop_inference_dual.py
extra=()
if [[ $method == pdf ]]; then
    script=run_qwen35_multihop_inference_dual_pdf_planner.py
    extra+=(--planner-max-pages 6)
fi

echo "Running $name"
/usr/bin/time -v -o "auto_rollouts/${name}.time.txt" python "$script" \
    --questions "${QUESTIONS:-data/phase1_stage3_type_a_questions.json}" \
    --output "$output" \
    --paper-index-dir data/phase1_vector_db \
    --standards-index-dir data/phase1_standards_db \
    --paper-database-root data/Phase1PDFBase \
    --standards-database-root data/Phase1Standards \
    --content-mode hybrid \
    --max-standards-hops "$standards" --max-paper-hops "$papers" \
    --paper-top-n 2 --standards-top-n 2 --max-evidence-pages "$pages" \
    --planner-max-passages 3 --planner-passage-chars 2400 --planner-parent-chars 3600 \
    --gh200-patch-optimization --max-new-tokens 2048 --do-sample --seed "$seed" \
    "${extra[@]}"

python LLM_as_a_judge.py --resultpath "$output"
