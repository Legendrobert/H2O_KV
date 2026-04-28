# 在 Killarney 上跑 SpectraKV

Killarney (Compute Canada) 上跑这套代码的注意点和操作步骤.

## 一次性: 在登录节点准备环境

```bash
# 1. 拉代码 (假设你 fork 的 origin 已经 push 好了 spectrakv 分支)
cd ~/projects/aip-lenck/${USER}
git clone -b spectrakv git@github.com:Legendrobert/H2O_KV.git
cd H2O_KV/h2o_hf

# 2. 建 venv 装依赖 (只需做一次)
bash scripts/spectrakv/setup_env.sh

# 3. (可选) Llama-2-7b 已经在共享目录, 不需要下载:
#      ~/projects/aip-lenck/shared/models/Llama-2-7b-hf
#    只有要换别的模型才跑这个:
# bash scripts/spectrakv/download_model.sh huggyllama/llama-7b
```

## 平时: 提交作业

```bash
# 算法级 smoke (CPU 节点, ~10 秒跑完, 不耗 GPU 配额)
sbatch scripts/spectrakv/run_smoke.slurm

# 真实评测: 单点 (task + method, 可选 heavy/recent ratio)
#   method = full | h2o | local | spectra
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa h2o            # H2O 论文标配 (10/10)
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa spectra        # SpectraKV 同预算
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa spectra 0.05 0.05
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa full           # 上限 baseline

# 批量扫: 默认 2 tasks * (1 full + 2 methods * 4 ratios) = 18 个 sbatch
bash scripts/spectrakv/run_sweep.sh

# 自定义批量: 只跑 SpectraKV, 任务限制为 openbookqa
TASKS="openbookqa" METHODS_RATIO="spectra" METHODS_FIXED="" \
    bash scripts/spectrakv/run_sweep.sh
```

## 关键路径 (Killarney)

| 变量 | 含义 | 值 |
|---|---|---|
| 代码 | 项目空间 (有备份) | `~/projects/aip-lenck/${USER}/H2O_KV` |
| venv | 项目空间 (有备份, 不放 $SCRATCH 因为 60 天清理) | `/home/w1996246/projects/aip-lenck/w1996246/spectrakv_env` |
| 数据 / HF cache / 日志 | $SCRATCH (无备份, 60 天清理) | Killarney 自带 |
| 共享 Llama-2-7b | 已下载, 直接指 | `/project/aip-lenck/shared/models/Llama-2-7b-hf` |
| `SLURM --account` | RAS account | `aip-lenck` |
| `SLURM --partition` | GPU partition | 默认 `gpubase_h100_b1` (H100 80GB) / 备选 `gpubase_l40s_b1..b5` (L40S 48GB) |
| modules | venv 装时和运行时必须一致 | `StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11.5 cuda/12.6` |

## 几个必须知道的坑

1. **不要在登录节点跑训练/评测**. CC 会 kill 长任务.
2. **HF 模型放 $SCRATCH, 不放 $HOME**. $HOME 50GB, 一个 7B 就爆.
3. **计算节点无外网**. 评测脚本里加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
   `setup_env.sh` 和 `run_*.slurm` 里都设好了.
4. **transformers 必须钉 4.36.2**. 新版接口不兼容 H2O 原 attention 改写,
   `requirements_spectrakv.txt` 里已固定.
5. **不要用 conda**. CC 推荐 `module load python/3.11.5` + `virtualenv`.
6. **wheelhouse**: CC 内网有 PyPI 镜像 (`--no-index`). `setup_env.sh` 已处理好.

## 调试: 申请交互式节点

```bash
# CPU 交互式 (跑 smoke, 排查代码 bug)
salloc --account=aip-lenck --time=1:00:00 --mem=8G --cpus-per-task=2

# GPU 交互式 (跑模型, 排查 CUDA / OOM)
salloc --account=aip-lenck --partition=gpubase_l40s_b1 \
       --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1
```

进去之后:
```bash
cd ~/projects/aip-lenck/${USER}/H2O_KV/h2o_hf
source /home/w1996246/projects/aip-lenck/w1996246/spectrakv_env/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
python test_spectra_smoke.py --verbose
```
