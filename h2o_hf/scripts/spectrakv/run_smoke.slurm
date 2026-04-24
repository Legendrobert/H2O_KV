#!/bin/bash
#SBATCH --job-name=spectrakv-smoke
#SBATCH --account=def-YOUR_PI
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 算法级 smoke test, 只验证 SpectraKV 压缩算法本身的正确性.
# 不需要 GPU, 不加载真实模型, 几十秒跑完.
#
# 提交:
#   sbatch scripts/spectrakv/run_smoke.slurm

set -euo pipefail

echo "==> SLURM job ${SLURM_JOB_ID} on $(hostname)"
echo "==> CWD: $(pwd)"

VENV_DIR="${SCRATCH}/envs/spectrakv"

module --force purge
module load StdEnv/2023 python/3.10
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 这俩开关确保即使脚本里有 from_pretrained 也不会去联网
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

cd "${SLURM_SUBMIT_DIR}"
python -u test_spectra_smoke.py --verbose

echo "==> done"
