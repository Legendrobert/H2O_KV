#!/bin/bash
# 在 Killarney 登录节点跑一次, 建 venv 并装 SpectraKV 依赖.
# venv 放项目空间 (有备份); 不放 $SCRATCH 因为 60 天清理会把它删掉.
#
# 用法:
#   cd $PROJECT/$USER/H2O_KV/h2o_hf
#   bash scripts/spectrakv/setup_env.sh

set -euo pipefail

# venv 放项目空间, 不放 $SCRATCH (后者 60 天清理会把 venv 删掉)
VENV_DIR="${VENV_DIR:-/home/w1996246/projects/aip-lenck/w1996246/spectrakv_env}"
REQ_FILE="$(dirname "$0")/../../requirements_spectrakv.txt"

echo "==> 加载 Killarney 模块"
# 和 run_lm_eval.slurm 严格一致, 否则 venv 装时和运行时 ABI 会错位
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11.5 cuda/12.6

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
