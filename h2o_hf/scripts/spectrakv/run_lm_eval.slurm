#!/bin/bash
#SBATCH --job-name=spectrakv-eval
#SBATCH --account=aip-lenck
#SBATCH --partition=gpubase_h100_b1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=592730248@qq.com

# 真实评测: 跑 H2O lm_eval pipeline, 支持 H2O / SpectraKV 切换.
# 默认模型 type = spectra (走 SpectraLlamaForCausalLM),
# 要和 H2O baseline 对比就把 $4 改为 llama.
#
# 提交 (默认 SpectraKV + 共享 Llama-2-7b):
#   sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa
#
# 完整参数:
#   $1 task        e.g. openbookqa, copa, mathqa, ...
#   $2 model       HF id 或本地路径 (默认: 共享 Llama-2-7b)
#   $3 model_arch  llama / opt        (默认: llama)
#   $4 model_type  spectra / llama / opt / gpt_neox  (默认: spectra)
#
# 默认 partition = H100 80GB. 内存够想省卡可以换 gpubase_l40s_b1 (L40S 48GB).

set -euo pipefail

TASK="${1:?usage: sbatch run_lm_eval.slurm <task> [<model> <model_arch> <model_type>]}"
MODEL="${2:-/project/aip-lenck/shared/models/Llama-2-7b-hf}"
MODEL_ARCH="${3:-llama}"
MODEL_TYPE="${4:-spectra}"
SHOTS=5

echo "==> SLURM job ${SLURM_JOB_ID} on $(hostname)"
echo "==> task=${TASK} model=${MODEL} arch=${MODEL_ARCH} type=${MODEL_TYPE} shots=${SHOTS}"
echo "==> CWD: $(pwd)"
nvidia-smi || echo "nvidia-smi 不可用 (CPU 节点?)"

VENV_DIR="/home/w1996246/projects/aip-lenck/w1996246/spectrakv_env"

module --force purge
module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11.5 cuda/12.6
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spectrakv_runs/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# 软链一份代码到工作目录 (评测脚本会写一堆中间 jsonl)
ln -sf "${SLURM_SUBMIT_DIR}/generate_task_data.py" .
ln -sf "${SLURM_SUBMIT_DIR}/run_lm_eval_harness.py" .
ln -sf "${SLURM_SUBMIT_DIR}/evaluate_task_result.py" .
ln -sf "${SLURM_SUBMIT_DIR}/utils_real_drop" .
ln -sf "${SLURM_SUBMIT_DIR}/utils_spectra" .
ln -sf "${SLURM_SUBMIT_DIR}/tasks" .

INPUT="${TASK}-${SHOTS}.jsonl"
OUTPUT="${TASK}-${SHOTS}-${MODEL_ARCH}-${MODEL_TYPE}.jsonl"

echo "==> Step 1: prepare task data"
python -u generate_task_data.py \
    --output-file "${INPUT}" \
    --task-name "${TASK}" \
    --num-fewshot "${SHOTS}"

echo "==> Step 2: inference (model_type=${MODEL_TYPE}, h_ratio=0.1, r_ratio=0.1)"
python -u run_lm_eval_harness.py \
    --input-path "${INPUT}" \
    --output-path "${OUTPUT}" \
    --model-name "${MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --enable_small_cache \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1

echo "==> Step 3: evaluate"
python -u evaluate_task_result.py \
    --result-file "${OUTPUT}" \
    --task-name "${TASK}" \
    --num-fewshot "${SHOTS}" \
    --model-type "${MODEL_ARCH}"

echo "==> done. Outputs in ${OUT_DIR}"
