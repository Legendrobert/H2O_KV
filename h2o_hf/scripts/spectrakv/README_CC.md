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

# 真实评测 (1 L40S, 默认跑 SpectraKV + 共享 Llama-2-7b)
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa

# 对比: 同一 task 跑 H2O baseline
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa \
    ~/projects/aip-lenck/shared/models/Llama-2-7b-hf llama llama
```

## 关键路径 (Killarney)

| 变量 | 含义 | 值 |
|---|---|---|
| `~/projects/aip-lenck/${USER}` | 代码 + venv (有备份) | 项目空间 |
| `$SCRATCH` | 数据集 + HF cache + 日志 (无备份, 60 天清理) | Killarney 自带 |
| `~/projects/aip-lenck/shared/models/Llama-2-7b-hf` | 共享 Llama-2-7b, 省配额 | 共享模型目录 |
| `SLURM --account` | RAS account | `aip-lenck` |
| `SLURM --partition` | GPU partition | `gpubase_l40s_b1..b5` (L40S 48GB, 默认) / `gpubase_h100_b1..b5` (H100 80GB) |

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
source $SCRATCH/envs/spectrakv/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
python test_spectra_smoke.py --verbose
```
