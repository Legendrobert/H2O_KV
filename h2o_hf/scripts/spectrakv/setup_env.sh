#!/bin/bash
# 在 CC 登录节点跑一次, 建 venv 并装 SpectraKV 依赖.
# venv 放在 $SCRATCH 下避免吃 $HOME 配额.
#
# 用法:
#   cd $PROJECT/$USER/H2O_KV/h2o_hf
#   bash scripts/spectrakv/setup_env.sh

set -euo pipefail

VENV_DIR="${SCRATCH}/envs/spectrakv"
REQ_FILE="$(dirname "$0")/../../requirements_spectrakv.txt"

echo "==> 加载 CC 模块"
# CC 上 python 3.10 + cuda 12.1 是当前 H2O / transformers 4.36 兼容性最好的组合
module --force purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.1
module load arrow/14.0.1   # datasets / pyarrow 依赖, CC 必须用模块版

echo "==> 创建 venv: ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -d "${VENV_DIR}" ]]; then
    virtualenv --no-download "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> 升级 pip (用 CC 内网 wheelhouse)"
pip install --no-index --upgrade pip

echo "==> 安装 SpectraKV 依赖"
# --no-index: 强制走 CC 内网 wheelhouse, 不连外网
# 大部分包 CC wheelhouse 都有; 如果某个版本没有, 改回普通 pip install 即可
pip install --no-index -r "${REQ_FILE}"

echo "==> 验证"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); print('transformers', transformers.__version__)"

echo ""
echo "完成. 平时激活: source ${VENV_DIR}/bin/activate"
