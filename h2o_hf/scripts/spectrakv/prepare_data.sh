#!/usr/bin/env bash
# 在 Killarney 登录节点跑一次, 预下评测数据集到 $HF_HOME.
# Compute node 没有外网 (run_lm_eval.slurm 里设了 HF_HUB_OFFLINE=1), 必须先在登录节点缓存.
#
# 用法:
#   cd ~/projects/aip-lenck/${USER}/H2O_KV/h2o_hf
#   bash scripts/spectrakv/prepare_data.sh
#
# 默认 task: openbookqa copa piqa winogrande (覆盖 commonsense QA 类的 4 个常用集).
# 自定义: TASKS="copa piqa rte" bash scripts/spectrakv/prepare_data.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/w1996246/projects/aip-lenck/w1996246/spectrakv_env}"
TASKS="${TASKS:-openbookqa copa piqa winogrande}"
MODEL="${MODEL:-/project/aip-lenck/shared/models/Llama-2-7b-hf}"
SHOTS=5

# 登录节点 = 有外网, 把 offline 标志清掉 (slurm 脚本里再开)
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
mkdir -p "${HF_HOME}"

module --force purge
module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11.5 cuda/12.6
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# tasks/eval_harness.py 通过 MODEL_NAME 找 tokenizer, 默认 facebook/opt-1.3b 会触发下载.
export MODEL_NAME="${MODEL}"

TMPDIR_OUT="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_OUT}"' EXIT

cd "$(dirname "$0")/../.."  # h2o_hf/

echo "==> HF_HOME = ${HF_HOME}"
echo "==> tasks = ${TASKS}"
echo

for task in ${TASKS}; do
    echo "==> Downloading ${task} (${SHOTS}-shot)"
    python -u generate_task_data.py \
        --output-file "${TMPDIR_OUT}/${task}-${SHOTS}.jsonl" \
        --task-name "${task}" \
        --num-fewshot "${SHOTS}"
    echo
done

echo "==> 完成. 缓存目录:"
du -sh "${HF_HOME}/datasets" 2>/dev/null || echo "  (空)"
echo
echo "==> 已缓存的 dataset:"
ls -1 "${HF_HOME}/datasets" 2>/dev/null || echo "  (空)"
