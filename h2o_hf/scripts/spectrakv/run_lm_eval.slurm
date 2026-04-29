#!/bin/bash
#SBATCH --job-name=spectrakv-eval
#SBATCH --account=aip-lenck
#SBATCH --partition=gpubase_h100_b2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=result/%x-%j.out
#SBATCH --error=result/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=592730248@qq.com

# H2O lm_eval pipeline, 支持 6 种 method 直接对照.
# 同 task / 同 ratio / 同 prompts, 唯一差别在 attention 上的 KV 策略.
#
# 用法:
#   sbatch run_lm_eval.slurm <task> <method> [<heavy_ratio> <recent_ratio>] [<model>]
#
# method 取值与默认 ratio (override 可传 $3 $4):
#   full           -> 不启用 small cache (oracle, 不丢任何 KV)
#   h2o            -> heavy=0.1 recent=0.1  (论文标配)
#   local          -> heavy=0   recent=0.2  (纯 sliding window)
#   spectra        -> heavy=0.1 recent=0.1  (SpectraKV 框架 + K-leverage 选择)
#   spectra_v      -> heavy=0.1 recent=0.1  (SpectraKV 框架 + V-leverage 选择, V 对输出贡献)
#   spectra_oracle -> heavy=0.1 recent=0.1  (SpectraKV 框架 + H2O attention-sum, 信号上限 ablation)
#
# 例:
#   sbatch run_lm_eval.slurm openbookqa h2o                 # H2O 论文标配
#   sbatch run_lm_eval.slurm openbookqa spectra             # 默认 SpectraKV
#   sbatch run_lm_eval.slurm openbookqa spectra_v           # V-leverage 变体
#   sbatch run_lm_eval.slurm openbookqa spectra_oracle      # 信号上限对照
#   sbatch run_lm_eval.slurm openbookqa spectra 0.05 0.05   # 低预算 SpectraKV
#   sbatch run_lm_eval.slurm openbookqa full                # 上限 baseline

set -euo pipefail

TASK="${1:?usage: sbatch run_lm_eval.slurm <task> <method> [heavy recent] [model]}"
METHOD="${2:?method: full | h2o | local | spectra | spectra_v | spectra_oracle}"
MODEL="${5:-/project/aip-lenck/shared/models/Llama-2-7b-hf}"
SHOTS=5

case "${METHOD}" in
    full)           DEFAULT_H=0.0 ; DEFAULT_R=0.0 ; MODEL_TYPE=llama   ; SMALL_CACHE=0 ; SEL_MODE="" ;;
    h2o)            DEFAULT_H=0.1 ; DEFAULT_R=0.1 ; MODEL_TYPE=llama   ; SMALL_CACHE=1 ; SEL_MODE="" ;;
    local)          DEFAULT_H=0.0 ; DEFAULT_R=0.2 ; MODEL_TYPE=llama   ; SMALL_CACHE=1 ; SEL_MODE="" ;;
    spectra)        DEFAULT_H=0.1 ; DEFAULT_R=0.1 ; MODEL_TYPE=spectra ; SMALL_CACHE=1 ; SEL_MODE="leverage" ;;
    spectra_v)      DEFAULT_H=0.1 ; DEFAULT_R=0.1 ; MODEL_TYPE=spectra ; SMALL_CACHE=1 ; SEL_MODE="v_leverage" ;;
    spectra_oracle) DEFAULT_H=0.1 ; DEFAULT_R=0.1 ; MODEL_TYPE=spectra ; SMALL_CACHE=1 ; SEL_MODE="attention_sum" ;;
    *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

HEAVY="${3:-${DEFAULT_H}}"
RECENT="${4:-${DEFAULT_R}}"
MODEL_ARCH=llama
TAG="${TASK}-${METHOD}-h${HEAVY}-r${RECENT}"

echo "==> SLURM job ${SLURM_JOB_ID} on $(hostname)"
echo "==> task=${TASK} method=${METHOD} heavy=${HEAVY} recent=${RECENT} model_type=${MODEL_TYPE} small_cache=${SMALL_CACHE} sel_mode=${SEL_MODE:-n/a}"
echo "==> model=${MODEL}"
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

# tasks/eval_harness.py 通过这个环境变量找 tokenizer (默认值 facebook/opt-1.3b 会触发联网下载).
# 必须设到本地模型路径, 否则 generate_task_data.py 在离线节点崩.
export MODEL_NAME="${MODEL}"

cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spectrakv_runs/${TAG}-${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# 软链一份代码到工作目录 (评测脚本会写一堆中间 jsonl)
ln -sf "${SLURM_SUBMIT_DIR}/generate_task_data.py" .
ln -sf "${SLURM_SUBMIT_DIR}/run_lm_eval_harness.py" .
ln -sf "${SLURM_SUBMIT_DIR}/evaluate_task_result.py" .
ln -sf "${SLURM_SUBMIT_DIR}/utils_lm_eval" .
ln -sf "${SLURM_SUBMIT_DIR}/utils_real_drop" .
ln -sf "${SLURM_SUBMIT_DIR}/utils_spectra" .
ln -sf "${SLURM_SUBMIT_DIR}/tasks" .

INPUT="${TASK}-${SHOTS}.jsonl"
OUTPUT="${TAG}.jsonl"

echo "==> Step 1: prepare task data"
python -u generate_task_data.py \
    --output-file "${INPUT}" \
    --task-name "${TASK}" \
    --num-fewshot "${SHOTS}"

echo "==> Step 2: inference (method=${METHOD}, heavy=${HEAVY}, recent=${RECENT}, sel=${SEL_MODE:-n/a})"
INFER_ARGS=(
    --input-path "${INPUT}"
    --output-path "${OUTPUT}"
    --model-name "${MODEL}"
    --model-type "${MODEL_TYPE}"
    --heavy_ratio "${HEAVY}"
    --recent_ratio "${RECENT}"
)
if [[ "${SMALL_CACHE}" == "1" ]]; then
    INFER_ARGS+=( --enable_small_cache )
fi
if [[ -n "${SEL_MODE}" ]]; then
    INFER_ARGS+=( --selection_mode "${SEL_MODE}" )
fi
python -u run_lm_eval_harness.py "${INFER_ARGS[@]}"

echo "==> Step 3: evaluate"
python -u evaluate_task_result.py \
    --result-file "${OUTPUT}" \
    --task-name "${TASK}" \
    --num-fewshot "${SHOTS}" \
    --model-type "${MODEL_ARCH}"

echo "==> done. Outputs in ${OUT_DIR}"
