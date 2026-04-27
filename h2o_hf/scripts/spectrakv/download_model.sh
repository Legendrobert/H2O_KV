#!/bin/bash
# 在 Killarney 登录节点跑, 把 HuggingFace 模型下到 $SCRATCH/hf_cache.
# 计算节点无外网, 必须事先在登录节点下完整.
#
# 提示:
#   Llama-2-7b 已经有共享副本: ~/projects/aip-lenck/shared/models/Llama-2-7b-hf
#   主线实验直接用它, 不需要跑这个脚本. 本脚本留给非 Llama-2-7b 的模型.
#
# 用法:
#   bash scripts/spectrakv/download_model.sh meta-llama/Llama-2-7b-hf   # 除非你想自己下一份, 否则用共享的
#   bash scripts/spectrakv/download_model.sh huggyllama/llama-7b
#
# 注意:
#   - 7B fp16 ~14GB, 下载 5-15 分钟.
#   - 登录节点不允许跑长任务, 但下载属于 IO 任务一般不会被 kill.
#     真要 timeout, 用 Killarney 的 datatransfer node.

set -euo pipefail

MODEL="${1:-meta-llama/Llama-2-7b-hf}"
HF_CACHE="${SCRATCH}/hf_cache"
VENV_DIR="${SCRATCH}/envs/spectrakv"

# 共享 Llama-2-7b 提醒
if [[ "${MODEL}" == "meta-llama/Llama-2-7b-hf" ]]; then
    echo "提示: Llama-2-7b 已经有共享副本:"
    echo "  ~/projects/aip-lenck/shared/models/Llama-2-7b-hf"
    echo "评测脚本直接指向该路径即可, 省配额."
    echo ""
fi

echo "==> 模型: ${MODEL}"
echo "==> 缓存目录: ${HF_CACHE}"

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_DOWNLOAD_TIMEOUT=120

# 加载 venv (依赖 huggingface_hub)
module --force purge
module load StdEnv/2023 python/3.11.5
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<PY
import os
from huggingface_hub import snapshot_download

model_id = "${MODEL}"
cache_dir = os.environ["HF_HOME"]

print(f"snapshot_download({model_id!r}, cache_dir={cache_dir!r}) ...")
local_dir = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    # 只下推理需要的, 跳过 .bin 改用 safetensors (省一半空间)
    allow_patterns=["*.json", "*.model", "*.safetensors", "tokenizer*", "*.txt"],
    resume_download=True,
)
print(f"完成: {local_dir}")
PY

echo ""
echo "下载完成. 评测脚本里用相同的 model_name 即可: ${MODEL}"
echo "  HF_HOME=${HF_CACHE}"
echo "  TRANSFORMERS_CACHE=${HF_CACHE}"
