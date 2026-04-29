#!/usr/bin/env bash
# 批量提交 H2O vs SpectraKV 对照矩阵.
# 在 h2o_hf/ 下跑:
#   bash scripts/spectrakv/run_sweep.sh
#
# 默认矩阵:
#   tasks         = openbookqa copa
#   methods_fixed = full                                          (1 point/task, 不扫 ratio)
#   methods_ratio = h2o spectra spectra_v spectra_oracle          (跟 RATIOS 笛卡尔积)
#   ratios        = 0.01 0.02 0.05 0.1                            (heavy = recent = ratio)
# = 2 tasks * (1 + 4*4) = 34 个 sbatch.
#
# spectra_v      = SpectraKV 框架 + V-leverage (V 对输出贡献的选择信号)
# spectra_oracle = SpectraKV 框架 + H2O attention-sum (信号上限对照)
#
# 可用环境变量覆盖, 例:
#   TASKS="openbookqa" RATIOS="0.02 0.05" bash scripts/spectrakv/run_sweep.sh
#   METHODS_RATIO="spectra" bash scripts/spectrakv/run_sweep.sh   # 只跑 SpectraKV
#   METHODS_RATIO="spectra spectra_v spectra_oracle" METHODS_FIXED="" \
#       bash scripts/spectrakv/run_sweep.sh                       # 只跑信号 ablation

set -euo pipefail

TASKS="${TASKS:-openbookqa copa}"
RATIOS="${RATIOS:-0.01 0.02 0.05 0.1}"
METHODS_RATIO="${METHODS_RATIO:-h2o spectra spectra_v spectra_oracle}"
METHODS_FIXED="${METHODS_FIXED:-full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_lm_eval.slurm"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "找不到 ${SLURM_SCRIPT}" >&2
    exit 1
fi

n=0
for task in ${TASKS}; do
    for m in ${METHODS_FIXED}; do
        echo "+ sbatch ${SLURM_SCRIPT} ${task} ${m}"
        sbatch "${SLURM_SCRIPT}" "${task}" "${m}"
        n=$((n+1))
    done
    for m in ${METHODS_RATIO}; do
        for r in ${RATIOS}; do
            echo "+ sbatch ${SLURM_SCRIPT} ${task} ${m} ${r} ${r}"
            sbatch "${SLURM_SCRIPT}" "${task}" "${m}" "${r}" "${r}"
            n=$((n+1))
        done
    done
done

echo "==> submitted ${n} jobs. squeue -u \$USER 可查."
