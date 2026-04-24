"""
SpectraKV 算法级 smoke test.

设计理念:
  - 这是算法单元测试, 不跑任何真实 LLM forward, 不依赖 transformers 的 Llama 架构.
  - 只构造随机 K/V tensor 喂给 SpectraKVCache_LayerWise, 验证谱压缩算法的正确性.
  - CPU 秒跑, 几十 MB 内存, 不受 transformers 版本影响, 适合本地高频迭代.

覆盖的性质:
  1. 短 N (<= cache_size): 不压缩, shape 原样返回, compressed=False
  2. 长 N (>  cache_size): 一次性压缩到 cache_size, compressed=True
  3. sink_size 开头强制保留
  4. recent_size 尾部强制保留
  5. one-shot 语义: 压缩后再次调用 __call__ 为 no-op
  6. _clean_scores 后可重新压缩
  7. leverage score 数学性质: 对低秩 K, 能有效识别出"骨架行"
  8. 压缩后 K/V 行顺序和原始位置序保持一致 (保 RoPE 语义)
  9. K 的 gather 索引对 V 同样适用 (两边对齐, 否则 attention 就错位)

用法:
    cd h2o_hf
    python test_spectra_smoke.py
    python test_spectra_smoke.py --seed 42 --verbose
"""
import argparse
import math
import sys

import torch

from utils_spectra.modify_llama import SpectraKVCache_LayerWise


# ============================================================================
# 公共工具: 造一个 (1, H, N, d) 的随机 K/V. V 用 K 加扰动, 方便检查 V 跟着 K 走.
# ============================================================================
def make_random_kv(bsz=1, num_heads=8, N=512, d=64, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    K = torch.randn(bsz, num_heads, N, d, generator=g, device=device)
    # V 用独立分布, 但给每个位置打上"可识别标记": V[..., i, 0] = i
    # 这样压缩后可以反查哪些 token 位置被保留
    V = torch.randn(bsz, num_heads, N, d, generator=g, device=device)
    pos_stamp = torch.arange(N, device=device, dtype=V.dtype).view(1, 1, N, 1)
    V = V.clone()
    V[..., :1] = pos_stamp  # V 的第 0 个通道被我们覆盖成位置索引
    return K, V


def recover_kept_positions(V_compressed):
    """
    V[..., :1] 存了原始位置索引, 压缩后从 V_compressed 反推每个头保留了哪些位置.
    返回 (H, L) 的 LongTensor.
    """
    return V_compressed[0, :, :, 0].round().long()


# ============================================================================
# 测试 1: 短序列不压缩
# ============================================================================
def test_short_no_compression(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)
    budget = cache.cache_size  # 44

    N = budget - 1  # 低于预算
    K, V = make_random_kv(N=N, device=device)
    K_out, V_out = cache((K, V))

    assert K_out.shape == K.shape, f"K shape 被改动: {K.shape} -> {K_out.shape}"
    assert V_out.shape == V.shape, f"V shape 被改动: {V.shape} -> {V_out.shape}"
    assert torch.equal(K_out, K), "K 内容被改动, 短序列应原样返回"
    assert torch.equal(V_out, V), "V 内容被改动, 短序列应原样返回"
    assert cache.compressed is False, "短序列不应标记 compressed"

    if verbose:
        print(f"    N={N} <= cache_size={budget}, K/V 原样返回, compressed={cache.compressed}")
    print("  [OK] 短序列 (N <= cache_size) 不触发压缩")


# ============================================================================
# 测试 2: 长序列触发一次性压缩 + 形状/长度正确
# ============================================================================
def test_long_triggers_compression(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)
    budget = cache.cache_size

    N = 512
    K, V = make_random_kv(N=N, device=device)
    K_out, V_out = cache((K, V))

    assert K_out.shape[-2] == budget, f"K 压缩后长度应为 {budget}, 实际 {K_out.shape[-2]}"
    assert V_out.shape[-2] == budget, f"V 压缩后长度应为 {budget}, 实际 {V_out.shape[-2]}"
    assert cache.compressed is True, "长序列压缩后 compressed 应为 True"

    if verbose:
        print(f"    N={N} > budget={budget}, 压缩到 {K_out.shape[-2]}, compressed={cache.compressed}")
    print(f"  [OK] 长序列 (N={N}) 压缩到 cache_size={budget}")


# ============================================================================
# 测试 3: sink + recent 强制保留
# ============================================================================
def test_sink_and_recent_preserved(device, verbose):
    sink_size, recent_size, hh_size = 4, 8, 32
    cache = SpectraKVCache_LayerWise(hh_size=hh_size, recent_size=recent_size, sink_size=sink_size)

    N = 512
    K, V = make_random_kv(N=N, device=device)
    _, V_out = cache((K, V))

    kept = recover_kept_positions(V_out)  # (H, L)

    # sink 段 [0, sink_size) 必须整段在
    sink_expected = set(range(sink_size))
    for h in range(kept.shape[0]):
        head_kept = set(kept[h].tolist())
        missing = sink_expected - head_kept
        assert not missing, f"Head {h} 缺失 sink 位置 {missing}"

    # recent 段 [N-recent_size, N) 必须整段在
    recent_expected = set(range(N - recent_size, N))
    for h in range(kept.shape[0]):
        head_kept = set(kept[h].tolist())
        missing = recent_expected - head_kept
        assert not missing, f"Head {h} 缺失 recent 位置 {missing}"

    if verbose:
        print(f"    sink={list(range(sink_size))} 全保留")
        print(f"    recent={list(range(N-recent_size, N))} 全保留")
    print("  [OK] sink / recent 段强制保留")


# ============================================================================
# 测试 4: one-shot 语义, 第二次调用 __call__ 应该 no-op
# ============================================================================
def test_one_shot_semantics(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)

    K, V = make_random_kv(N=512, device=device)
    K1, V1 = cache((K, V))
    assert cache.compressed, "首次应该压缩"

    # 模拟 decode 追加了 5 个新 token
    K_new = torch.randn(1, 8, 5, 64, device=device)
    V_new = torch.randn(1, 8, 5, 64, device=device)
    K_appended = torch.cat([K1, K_new], dim=2)
    V_appended = torch.cat([V1, V_new], dim=2)

    K2, V2 = cache((K_appended, V_appended))
    assert torch.equal(K2, K_appended), "压缩后再次调用应 no-op, K 应原样返回"
    assert torch.equal(V2, V_appended), "压缩后再次调用应 no-op, V 应原样返回"
    assert cache.compressed, "状态保持 compressed=True"

    if verbose:
        print(f"    压缩后 cache_size={K1.shape[-2]}, decode 追加 5 后 = {K2.shape[-2]}")
        print(f"    第二次 __call__ 未再次压缩 (one-shot 正确)")
    print("  [OK] one-shot: 压缩后再次调用 no-op")


# ============================================================================
# 测试 5: _clean_scores 后可重新压缩
# ============================================================================
def test_clean_and_recompress(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)

    K, V = make_random_kv(N=512, seed=0, device=device)
    cache((K, V))
    assert cache.compressed

    cache._clean_scores()
    assert cache.compressed is False, "_clean_scores 后 compressed 应重置为 False"

    K2, V2 = make_random_kv(N=512, seed=1, device=device)
    K_out, V_out = cache((K2, V2))
    assert K_out.shape[-2] == cache.cache_size, "重置后再压一次, 长度应为 cache_size"
    assert cache.compressed is True

    if verbose:
        print(f"    第一次压缩 -> _clean_scores -> 第二次压缩, 均到达 cache_size={cache.cache_size}")
    print("  [OK] _clean_scores 后可重新压缩")


# ============================================================================
# 测试 6: 位置序保持单调递增 (压缩后 keep_idx 是 sort 过的)
# ============================================================================
def test_position_order_preserved(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)

    N = 512
    K, V = make_random_kv(N=N, device=device)
    _, V_out = cache((K, V))

    kept = recover_kept_positions(V_out)  # (H, L)
    for h in range(kept.shape[0]):
        diffs = kept[h][1:] - kept[h][:-1]
        assert (diffs > 0).all(), f"Head {h} 保留位置不是严格递增: {kept[h].tolist()}"

    if verbose:
        print(f"    head 0 保留位置序前 10: {kept[0][:10].tolist()}")
    print("  [OK] 保留位置严格递增 (RoPE 顺序保真)")


# ============================================================================
# 测试 7: K/V 在相同索引上被 gather (关键: 否则 attention 错位)
# ============================================================================
def test_kv_indices_aligned(device, verbose):
    cache = SpectraKVCache_LayerWise(hh_size=32, recent_size=8, sink_size=4)

    # 构造 K 和 V, 在每个位置 i 上的内容都可识别且 K[i] 和 V[i] 一一对应
    N = 512
    bsz, H, d = 1, 8, 64
    K = torch.zeros(bsz, H, N, d, device=device)
    V = torch.zeros(bsz, H, N, d, device=device)
    # K[..., i, 0] = i, V[..., i, 0] = -i, 这样 K_out[..., j, 0] 和 V_out[..., j, 0]
    # 满足 V_out[..., j, 0] == -K_out[..., j, 0] 当且仅当两者从同一原始位置 gather
    K[..., 0] = torch.arange(N, device=device, dtype=K.dtype).view(1, 1, N)
    V[..., 0] = -torch.arange(N, device=device, dtype=V.dtype).view(1, 1, N)
    # 其他维度随机, 让 leverage score 的选择不是平凡的
    g = torch.Generator(device=device).manual_seed(0)
    K[..., 1:] = torch.randn(bsz, H, N, d - 1, generator=g, device=device)
    V[..., 1:] = torch.randn(bsz, H, N, d - 1, generator=g, device=device)

    K_out, V_out = cache((K, V))
    # 相加应全为 0 (每个 head 的每个保留位置 K[...,0] + V[...,0] == i + (-i) == 0)
    mismatch = (K_out[..., 0] + V_out[..., 0]).abs().max().item()
    assert mismatch < 1e-4, f"K 和 V gather 索引不对齐, 最大偏差 {mismatch}"

    if verbose:
        print(f"    K/V 位置标记 sum 最大偏差 = {mismatch:.2e}")
    print("  [OK] K 与 V 的 gather 索引完全对齐")


# ============================================================================
# 测试 8: 低秩 K 下 leverage score 的数学性质
# ============================================================================
def test_leverage_score_on_low_rank_K(device, verbose):
    """
    构造一个 rank=r 的 K, 明确知道"骨架行"是哪 r 行.
    leverage score 应该把大部分分数集中在这些行上.

    设计:
      K ∈ R^{N×d}, 其中 N > d.
      选定 r_target 个"骨架行", 给它们大的系数 (norm).
      其余行 = 骨架行的小随机线性组合 + tiny 噪声.
      理论上 leverage score 应该在骨架行上显著更高.
    """
    torch.manual_seed(0)
    bsz, H, d = 1, 1, 16
    N = 128
    r_target = 8
    sink_size, recent_size, hh_size = 2, 2, r_target + 2  # 给中间段留 r_target 的预算 + 冗余

    # 骨架行 (放在中间段里, 避开 sink/recent 的强制保留范围)
    # 中间段是 [sink_size, N - recent_size) = [2, 126)
    middle_start, middle_end = sink_size, N - recent_size
    middle_len = middle_end - middle_start
    skeleton_local = torch.randperm(middle_len)[:r_target]
    skeleton_global = (skeleton_local + middle_start).sort().values

    # 骨架向量: 大 norm
    skeleton_vecs = torch.randn(r_target, d) * 5.0

    # 构造完整 K: 骨架行 = skeleton_vecs, 其余行 = 骨架行的小系数组合 + 小扰动
    K_single = torch.zeros(N, d)
    # 先填骨架行
    for i, pos in enumerate(skeleton_global.tolist()):
        K_single[pos] = skeleton_vecs[i]
    # 其余行 (包括 sink 和 recent 内的): 骨架行的小系数线性组合
    skeleton_set = set(skeleton_global.tolist())
    for i in range(N):
        if i in skeleton_set:
            continue
        coef = torch.randn(r_target) * 0.1  # 小系数 -> row norm 小, leverage 小
        K_single[i] = coef @ skeleton_vecs + 0.01 * torch.randn(d)

    K = K_single.view(1, 1, N, d).to(device)
    V = torch.randn(1, 1, N, d, device=device)
    # 把位置标记塞进 V 便于回溯
    V = V.clone()
    V[..., 0] = torch.arange(N, device=device, dtype=V.dtype).view(1, 1, N)

    cache = SpectraKVCache_LayerWise(
        hh_size=hh_size, recent_size=recent_size, sink_size=sink_size,
        jl_dim_multiplier=4,
    )
    _, V_out = cache((K, V))
    kept = recover_kept_positions(V_out)[0].tolist()  # head 0
    kept_middle = [p for p in kept if sink_size <= p < N - recent_size]

    # 骨架行被选中的比例
    hit = len(set(kept_middle) & set(skeleton_global.tolist()))
    hit_ratio = hit / r_target
    assert hit_ratio >= 0.75, (
        f"leverage score 应优先选骨架行, 但命中率只有 {hit_ratio:.2%} "
        f"(命中 {hit}/{r_target}). 中间段选中: {sorted(kept_middle)}, "
        f"骨架位置: {skeleton_global.tolist()}"
    )

    if verbose:
        print(f"    骨架行 (真 rank-{r_target}): {skeleton_global.tolist()}")
        print(f"    中间段选中:                   {sorted(kept_middle)}")
        print(f"    命中率: {hit}/{r_target} = {hit_ratio:.2%}")
    print(f"  [OK] 低秩 K 下 leverage score 识别出 {hit}/{r_target} 个骨架行 ({hit_ratio:.0%})")


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu",
                        help="默认 CPU, 显存紧张的机器不碰 GPU")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tests = [
        ("短序列不压缩",                        test_short_no_compression),
        ("长序列触发压缩",                      test_long_triggers_compression),
        ("sink + recent 强制保留",              test_sink_and_recent_preserved),
        ("one-shot 语义",                       test_one_shot_semantics),
        ("_clean_scores 后可重压",              test_clean_and_recompress),
        ("保留位置序单调",                      test_position_order_preserved),
        ("K/V 索引对齐",                        test_kv_indices_aligned),
        ("低秩 K 下 leverage 命中骨架行",       test_leverage_score_on_low_rank_K),
    ]

    print(f"设备: {device}")
    print(f"种子: {args.seed}")
    print(f"共 {len(tests)} 个测试\n")

    failed = []
    for i, (name, fn) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {name}")
        try:
            fn(device=device, verbose=args.verbose)
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed.append(name)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed.append(name)
        print()

    print("=" * 60)
    if failed:
        print(f"失败 {len(failed)}/{len(tests)}: {failed}")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"全部通过 {len(tests)}/{len(tests)} ✅")
        print("=" * 60)


if __name__ == "__main__":
    main()
