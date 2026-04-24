# 在 Compute Canada 上跑 SpectraKV

CC (Cedar/Graham/Beluga/Narval) 上跑这套代码的注意点和操作步骤.

## 一次性: 在登录节点准备环境和模型

```bash
# 1. 拉代码 (假设你 fork 的 origin 已经 push 好了 spectrakv 分支)
cd $PROJECT/$USER
git clone -b spectrakv git@github.com:Legendrobert/H2O_KV.git
cd H2O_KV/h2o_hf

# 2. 建 venv 装依赖 (只需做一次)
bash scripts/spectrakv/setup_env.sh

# 3. 把 Llama-7b 抓到 $SCRATCH (~14GB, fp16)
#    计算节点没外网, 必须先在登录节点下载
bash scripts/spectrakv/download_model.sh huggyllama/llama-7b
```

## 平时: 提交作业

```bash
# 算法级 smoke (CPU 节点, ~5 分钟队列, ~10 秒跑完, 不耗 GPU 配额)
sbatch scripts/spectrakv/run_smoke.slurm

# 真实评测 (1 GPU, 几小时)
sbatch scripts/spectrakv/run_lm_eval.slurm openbookqa huggyllama/llama-7b llama
```

## 关键路径约定 (改你自己的)

| 变量 | 含义 | 默认值 |
|---|---|---|
| `$PROJECT/$USER` | 代码 + venv (有备份, 1TB) | CC 自带 |
| `$SCRATCH` | 数据集 + HF 模型 cache + 日志 (无备份, 60 天清理, 几 TB) | CC 自带 |
| `SLURM --account` | 你的 PI / RAS account | **占位符 `def-YOUR_PI`, SLURM 脚本里改** |

## 几个必须知道的坑

1. **不要在登录节点跑训练/评测**.  CC 会 kill 长任务.
2. **HF 模型必须放 $SCRATCH 不是 $HOME**.  $HOME 只有 50GB, 一个 7B 就爆.
3. **计算节点无外网**.  评测脚本里加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
   `setup_env.sh` 和 `run_*.slurm` 里都设好了.
4. **transformers 必须钉 4.36.2**.  新版接口不兼容 H2O 原 attention 改写,
   `requirements_spectrakv.txt` 里已固定.
5. **不要用 conda**.  CC 推荐 `module load python` + `virtualenv`, 文件数少 + 性能好.
6. **wheelhouse**:  CC 内网有 PyPI 镜像 (`--no-index`),  装包优先走它.
   `setup_env.sh` 已经处理好.

## 调试: 申请交互式节点

```bash
# CPU 交互式 (跑 smoke, 排查代码 bug)
salloc --account=def-YOUR_PI --time=1:00:00 --mem=8G --cpus-per-task=2

# GPU 交互式 (跑模型, 排查 CUDA / OOM)
salloc --account=def-YOUR_PI --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1
```

进去之后:
```bash
cd $PROJECT/$USER/H2O_KV/h2o_hf
source $SCRATCH/envs/spectrakv/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
python test_spectra_smoke.py --verbose
```
