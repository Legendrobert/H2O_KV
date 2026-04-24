#!/bin/bash
#SBATCH --job-name=spectrakv-eval
#SBATCH --account=def-YOUR_PI
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 真实评测: 跑 H2O 原 lm_eval pipeline.
# 等 SpectraKV 接好顶层模型 (与 transformers 4.36 兼容那一版) 后,
# 把这里 run_lm_eval_harness.py 的 --model-type 换成 spectra 即可.
#
# 提交:
#   sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa huggyllama/llama-7b llama
#
# 参数:
#   $1 task        e.g. openbookqa, copa, mathqa, ...
#   $2 model       HF id, 必须事先 download_model.sh 下载过
#   $3 model_arch  llama / opt

set -euo pipefail

TASK="${1:?usage: sbatch run_lm_eval.slurm <task> <model> <model_arch>}"
MODEL="${2:?usage: sbatch run_lm_eval.slurm <task> <model> <model_arch>}"
MODEL_ARCH="${3:?usage: sbatch run_lm_eval.slurm <task> <model> <model_arch>}"
SHOTS=5

echo "==> SLURM job ${SLURM_JOB_ID} on $(hostname)"
echo "==> task=${TASK} model=${MODEL} arch=${MODEL_ARCH} shots=${SHOTS}"
echo "==> CWD: $(pwd)"
nvidia-smi || echo "nvidia-smi 不可用 (CPU 节点?)"

VENV_DIR="${SCRATCH}/envs/spectrakv"

module --force purge
module load StdEnv/2023 python/3.10 cuda/12.1
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
OUTPUT="${TASK}-${SHOTS}-${MODEL_ARCH}-h2o.jsonl"

echo "==> Step 1: prepare task data"
python -u generate_task_data.py \
    --output-file "${INPUT}" \
    --task-name "${TASK}" \
    --num-fewshot "${SHOTS}"

echo "==> Step 2: H2O inference (h_ratio=0.1, r_ratio=0.1)"
python -u run_lm_eval_harness.py \
    --input-path "${INPUT}" \
    --output-path "${OUTPUT}" \
    --model-name "${MODEL}" \
    --model-type "${MODEL_ARCH}" \
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
