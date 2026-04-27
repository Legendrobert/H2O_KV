"""
SpectraKV: 基于统计杠杆分数 (Statistical Leverage Scores) 的一次性 KV Cache 压缩

===================================================================
算法核心思路
===================================================================
H2O 通过累加 attention 概率评估 token 重要性 —— query-dependent, 无谱近似保证.
SpectraKV 换视角: 只看 K 矩阵本身的几何结构, 挑出张成 K 列空间的"骨架" token.

对 K ∈ R^{N×d}, 第 i 行的 leverage score 定义为:
    l_i = ||U_{i,:}||^2,    其中 K = U Σ W^T 为 K 的 SVD
等价形式 (PSD 下):
    l_i = K_i (K^T K)^{-1} K_i^T

按 l_i 取 top-r 行, 对 *任意* query x 都满足谱近似:
    (1 - ε) ||K x||^2  ≤  ||K̃ x||^2  ≤  (1 + ε) ||K x||^2

这是 H2O 的 heuristic 拿不到的数学保证, 也是 SpectraKV 的理论根基.

===================================================================
实现要点
===================================================================
1. 不直接做 SVD (开销 O(N d^2)), 用 JL sketch 近似:
       R = (S K)^T (S K) ≈ K^T K,      S ∈ R^{m×N}, m = O(d log d)
       l̃_i = K_i R^{-1} K_i^T
   复杂度 O(N d log d + d^3), 远小于 prefill 自身.

2. 一次性 (one-shot) 压缩: prefill 结束后压一次, decode 阶段不再介入.
   这是和 H2O 最核心的工程区别 —— H2O 每步都要 evict, SpectraKV 不再动.

3. 强制保留:
   - sink token (前 sink_size 个): 处理 attention sink 现象, 纯谱视角看不到
   - recent window (最后 recent_size 个): 保 RoPE 近邻相关性
   - 中间段按 leverage score 取 top hh_size 个

4. Top-r 确定性选择 (不做加权采样 + reweight):
   reweight 会改 K 的行范数, 破坏 QK^T 的量纲 -> softmax 温度崩坏.
   放弃严格无偏估计, 换数值稳定和可复现性.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)

# 复用 H2O 里已经写好的工具函数和 LlamaConfig (避免重复定义, 保证行为一致)
from utils_real_drop.modify_llama import (
    LlamaConfig,
    repeat_kv,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
)


__all__ = [
    "SpectraKVCache_LayerWise",
    "SpectraLlamaAttention",
    "SpectraLlamaForCausalLM",
    "convert_kvcache_llama_spectra",
]


# ==========================================================================
# 一次性 leverage-score 压缩器
# ==========================================================================
class SpectraKVCache_LayerWise:
    """
    单层 KV Cache 的一次性谱压缩器.

    和 H2OKVCache_LayerWise 接口基本对齐, 方便 run_summarization / lm_eval 等
    评测脚本最小改动就能切换. 但语义不同:
      - H2O: 每个 forward 调一次, 持续 evict, 需要 attn_score_cache
      - SpectraKV: 第一次 cache 超过预算时压一次, 之后 no-op, 不需要 attn 分数

    预算构成:
        cache_size = sink_size + hh_size + recent_size
        ├── sink_size   : 强制保留的开头 token, 处理 attention sink
        ├── hh_size     : 中间段按 leverage score 取 top-r (算法核心)
        └── recent_size : 强制保留的尾部窗口
    """

    def __init__(
        self,
        hh_size: int = 128,
        recent_size: int = 32,
        sink_size: int = 4,
        jl_dim_multiplier: int = 4,
        reg_lambda: float = 1e-3,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        """
        Args:
            hh_size: leverage-score 选出的中间段 token 数
            recent_size: 尾部强制保留窗口大小
            sink_size: 开头强制保留的 sink token 数 (StreamingLLM 发现的现象, 一般 4)
            jl_dim_multiplier: JL sketch 维度 m = jl_dim_multiplier * head_dim
                               理论上 m = Θ(d log d) 足够, 4*d 在实际中稳健
            reg_lambda: R = (SK)^T(SK) 求逆前加的 λI 正则, 防病态
            k_seq_dim / v_seq_dim: K/V 张量里 "序列长度" 所在的维度
        """
        print(
            f"SpectraKVCache-LayerWise: sink={sink_size}, hh(top-r)={hh_size}, recent={recent_size}"
        )
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.sink_size = sink_size
        self.cache_size = sink_size + hh_size + recent_size

        self.jl_dim_multiplier = jl_dim_multiplier
        self.reg_lambda = reg_lambda

        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        # 标志: 是否已经做过一次性压缩. 一旦为 True 后续 forward 不再介入.
        self.compressed = False
        # 缓存当前样本算出来的 keep 索引. 一份 keep_idx 同时供:
        #   1) attention forward 里把非 keep 位置 mask 掉 (lm_eval 单次 forward 看得见压缩效果)
        #   2) 末尾 gather-based KV cache 压缩 (decode 复用)
        # 两路共用同一份索引保证 prefill 和 decode 一致.
        self._keep_idx = None

    # ----------------------------------------------------------------------
    # 对外接口: 接受 past_key_values, 返回压缩/未压缩 past_key_values
    # ----------------------------------------------------------------------
    def __call__(self, past_key_values, attn_score_cache=None):
        """
        Args:
            past_key_values: tuple (K, V), 每个 shape (bsz, num_kv_heads, seq_len, head_dim)
            attn_score_cache: 占位, 为了接口和 H2OKVCache_LayerWise 对齐. 这里不用.
        Returns:
            压缩后的 (K, V) 或原样返回.
        """
        if past_key_values is None:
            return None

        # 已经压缩过: 直接返回 (one-shot 语义)
        if self.compressed:
            return past_key_values

        seq_len = past_key_values[0].size(self.k_seq_dim)

        # 还没超预算: 不做事 (对应短 prefill 的情况)
        if seq_len <= self.cache_size:
            return past_key_values

        # 执行一次性压缩
        k_compressed, v_compressed = self._compress(past_key_values[0], past_key_values[1])
        self.compressed = True
        return (k_compressed, v_compressed)

    def evict_for_space(self, past_key_values, num_coming):
        """
        H2O streaming 接口, 这里语义是"腾空间".
        one-shot 模式下和 __call__ 语义相同, 直接复用.
        """
        return self.__call__(past_key_values, None)

    def _clean_scores(self):
        """每个样本评测完调用, 重置压缩状态和缓存的 keep 索引."""
        self.compressed = False
        self._keep_idx = None

    # ----------------------------------------------------------------------
    # JL sketch 近似 leverage score
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _compute_leverage_scores(self, K: torch.Tensor) -> torch.Tensor:
        """
        对 K 每一行计算近似 leverage score.

        数学:  l_i = K_i (K^T K)^{-1} K_i^T
        JL 近似: 用 R = (SK)^T(SK) + λI 替代 K^T K, 其中 S 是随机投影矩阵.
        复杂度 O(N d log d + d^3) 每头.

        Args:
            K: (bsz=1, num_heads, N, d), 可能是 fp16
        Returns:
            leverage: (num_heads, N), fp32
        """
        bsz, num_heads, N, d = K.shape
        assert bsz == 1, "SpectraKV 目前只支持 bsz=1 (和 H2O 保持一致)"

        # 数值计算统一在 fp32, 避免 fp16 下 R 的求逆不稳
        K_f32 = K.squeeze(0).to(torch.float32)              # (H, N, d)
        device = K.device

        # JL 维度 m. 理论上 m = Θ(d log d) 就足够 subspace embedding.
        # 实际 4*d 比较稳健, 同时不超过 N 否则浪费.
        m = min(self.jl_dim_multiplier * d, N)

        # 随机投影矩阵 S ~ N(0, 1/m). 所有 head 共享同一个 S:
        # leverage 计算本身是 per-head 的 (R 和 l_i 都是 per-head), 共享 S 不影响
        # 单头的估计质量, 同时显著省显存 (省 num_heads 倍).
        S = torch.randn(m, N, device=device, dtype=torch.float32) / math.sqrt(m)

        # SK: (1, m, N) @ (H, N, d) -> (H, m, d). 广播省去显式复制.
        SK = torch.matmul(S.unsqueeze(0), K_f32)

        # R = SK^T SK + λI: (H, d, d)
        R = torch.matmul(SK.transpose(1, 2), SK)
        eye = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0)

        # 自适应 λ: 跟 R 的对角平均同量级, 避免 K 数值大时 reg_lambda 失效.
        # 某些 Llama head (attention sink) K 行范数极大, R 量级到 1e6+, 固定的
        # reg_lambda=1e-3 起不到正则化作用, fp32 下噪声就能让 Cholesky 挂.
        # 这里把 λ 钉到 R 自己的 scale 的某个比例, 不管 K 多大都有效.
        diag_mean = R.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # (H, 1, 1)
        lam = (self.reg_lambda * diag_mean).clamp(min=1e-6)
        R = R + lam * eye

        # 一般情况 Cholesky 即可. 极端病态的 head fallback 到 pinv.
        try:
            L = torch.linalg.cholesky(R)                    # (H, d, d)
            R_inv = torch.cholesky_inverse(L)               # (H, d, d)
        except torch._C._LinAlgError:
            R_inv = torch.linalg.pinv(R)

        # leverage_i = K_i R^{-1} K_i^T
        # 先 K R^{-1}: (H, N, d)
        KR_inv = torch.matmul(K_f32, R_inv)
        # 然后逐行内积: (H, N, d) * (H, N, d) 按最后一维求和 -> (H, N)
        leverage = (K_f32 * KR_inv).sum(dim=-1)

        return leverage

    # ----------------------------------------------------------------------
    # 选 keep 行 (sink + top-r + recent), 缓存索引供 mask 和 gather 共用
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _compute_keep_idx(self, K: torch.Tensor) -> torch.Tensor:
        """
        算 leverage score, 选 sink + top-r(中间段) + recent, 返回排好序的索引.
        缓存到 self._keep_idx, 同一份索引被 attention mask 和 cache gather 共享.

        K shape: (1, num_kv_heads, N, head_dim)
        return : (num_kv_heads, L) long, L 通常 = cache_size
        """
        if self._keep_idx is not None:
            return self._keep_idx

        bsz, num_heads, N, d = K.shape
        device = K.device

        leverage = self._compute_leverage_scores(K)              # (H, N)

        middle_start = self.sink_size
        middle_end = N - self.recent_size
        middle_len = max(middle_end - middle_start, 0)
        k_middle = min(self.hh_size, middle_len)

        if k_middle > 0:
            middle_leverage = leverage[:, middle_start:middle_end]
            _, topk_local = torch.topk(middle_leverage, k=k_middle, dim=-1, largest=True)
            topk_global = topk_local + middle_start              # (H, k_middle)
        else:
            topk_global = torch.empty(num_heads, 0, dtype=torch.long, device=device)

        sink_idx = (
            torch.arange(self.sink_size, device=device)
            .unsqueeze(0)
            .expand(num_heads, -1)
        )
        recent_idx = (
            torch.arange(N - self.recent_size, N, device=device)
            .unsqueeze(0)
            .expand(num_heads, -1)
        )

        # 合并并按原位置排序: RoPE 已经烤在 K 里, 顺序保原样最安全
        keep_idx = torch.cat([sink_idx, topk_global, recent_idx], dim=-1)  # (H, L)
        keep_idx, _ = keep_idx.sort(dim=-1)

        self._keep_idx = keep_idx
        return keep_idx

    # ----------------------------------------------------------------------
    # 用缓存好的 keep_idx gather 出压缩的 K/V (decode 复用用)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _compress(self, K: torch.Tensor, V: torch.Tensor):
        """
        一次性压缩:
          1. _compute_keep_idx (若已缓存直接复用, 跟 attention mask 同源)
          2. 按原位置排序后 gather K/V (保留 RoPE 内嵌的位置语义)

        K, V shape: (1, num_kv_heads, N, head_dim)
        返回     : (1, num_kv_heads, cache_size, head_dim)
        """
        bsz, num_heads, N, d = K.shape

        keep_idx = self._compute_keep_idx(K)                     # (H, L)
        L = keep_idx.size(-1)

        idx_expand = keep_idx.unsqueeze(0).unsqueeze(-1).expand(bsz, num_heads, L, d)
        K_compressed = torch.gather(K, dim=2, index=idx_expand)
        V_compressed = torch.gather(V, dim=2, index=idx_expand)

        print(f"[SpectraKV] Triggered compression! Compressed from length {N} to {L}.", flush=True)

        return K_compressed, V_compressed


# ==========================================================================
# SpectraLlamaAttention: 替换 LlamaAttention
# ==========================================================================
class SpectraLlamaAttention(nn.Module):
    """
    Llama attention 的 SpectraKV 版本.

    结构上和 H2OLlamaAttention 几乎一致 (RoPE, GQA, causal mask 等都照搬),
    关键区别只在最后一步: 把 H2OKVCache_LayerWise 换成 SpectraKVCache_LayerWise,
    且不再需要传 attn_weights (query-agnostic).
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        # 从 config 读 SpectraKV 专属参数, 带缺省值方便和老 config 兼容
        self.kv_cache = SpectraKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            sink_size=getattr(config, "sink_size", 4),
            jl_dim_multiplier=getattr(config, "jl_dim_multiplier", 4),
            reg_lambda=getattr(config, "reg_lambda", 1e-3),
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _init_rope(self):
        # transformers >= 4.46 把 LlamaRotaryEmbedding 改为只接受 config 参数.
        # 旧版接受 (dim, max_position_embeddings, base), 此处做版本兼容.
        import inspect
        sig = inspect.signature(LlamaRotaryEmbedding.__init__)
        params = list(sig.parameters.keys())
        if "config" in params and len(params) <= 3:
            # 新版 API: LlamaRotaryEmbedding(config)
            if self.config.rope_scaling is not None:
                raise NotImplementedError(
                    "SpectraKV 暂未接入 rope_scaling, 如需长上下文扩展请补充 LinearScaling / DynamicNTK"
                )
            self.rotary_emb = LlamaRotaryEmbedding(self.config)
        else:
            # 旧版 API: LlamaRotaryEmbedding(dim, max_position_embeddings, base)
            if self.config.rope_scaling is None:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                raise NotImplementedError(
                    "SpectraKV 暂未接入 rope_scaling, 如需长上下文扩展请补充 LinearScaling / DynamicNTK"
                )

    def _clean_cache(self):
        """每条样本评测完调用, 重置压缩状态. (评测脚本里的 for loop 需要)"""
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ---- 新样本起点重置 cache 状态 ----
        # lm_eval harness 每个样本各调一次 forward (past_key_value=None 起步).
        # 不重置的话 self.compressed 会从上一个样本残留为 True, 导致这个样本的 attention
        # mask 不会激活, logprobs 还是无压缩的结果. 真 decode 场景 past_key_value 不为 None,
        # 不会进这个分支, one-shot 语义保留.
        if past_key_value is None:
            self.kv_cache._clean_scores()

        # ---- QKV 投影 ----
        # 注: pretraining_tp>1 的情况不进入我们的使用场景 (Llama-2-7b 单卡), 省略分片路径
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # ---- 重建 causal mask ----
        # past cache 长度是压缩后的 cache_size (或未压缩的原长), 每次 forward 重建最稳
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 处理 decode 单 token 情况下 position_ids 可能是标量的 corner case
        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1

        # ---- RoPE: 在 cat 到 past 之前烤到当前 q/k 上 ----
        # 这样 past_key_value 里存的一直是"带旋转的" K, 压缩时直接 gather 子集即可,
        # 不需要重新计算 RoPE (也不能重算, 因为被选中的 token 的绝对位置要保持).
        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # ---- 合并 past cache ----
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # use_cache=True 时, 这是"当前步看到的完整 KV", 稍后喂给 SpectraKV 压缩器
        past_key_value = (key_states, value_states) if use_cache else None

        # ---- 计算 spectral keep mask, 让压缩对当前 forward 的 logprobs 也生效 ----
        # 关键: 单次 forward 评测 (lm_eval) 没有 decode 步, 单纯把 past_key_value 压缩
        # 不会影响这一步 attention 的输出. 必须在 softmax 之前把非 keep 位置 mask 掉,
        # 才能让 SpectraKV 真的影响 logprobs (跟 H2O utils_lm_eval 的 mask 路径一致).
        # 同一份 keep_idx 后面也供 gather-based cache 压缩用, 保证两路一致.
        spec_mask = None
        if not self.kv_cache.compressed and kv_seq_len > self.kv_cache.cache_size:
            keep_idx = self.kv_cache._compute_keep_idx(key_states)              # (num_kv_heads, L)
            keep_bool = torch.zeros(
                self.num_key_value_heads, kv_seq_len,
                dtype=torch.bool, device=key_states.device,
            )
            keep_bool.scatter_(1, keep_idx, True)
            # GQA 展开成 num_heads
            keep_bool = keep_bool.repeat_interleave(self.num_key_value_groups, dim=0)  # (num_heads, kv_seq_len)
            mask_min = torch.finfo(query_states.dtype).min
            spec_mask = torch.zeros(
                bsz, self.num_heads, 1, kv_seq_len,
                dtype=query_states.dtype, device=key_states.device,
            )
            spec_mask.masked_fill_(~keep_bool.unsqueeze(0).unsqueeze(2), mask_min)

        # ---- GQA: 把 kv_head 重复到 num_heads ----
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        # ---- 标准 scaled-dot-product attention ----
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights size mismatch: "
                f"expected {(bsz, self.num_heads, q_len, kv_seq_len)}, got {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask size mismatch: "
                    f"expected {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # spectral keep mask: 把不被保留的 token 推到 -inf, softmax 后权重为 0
        if spec_mask is not None:
            attn_weights = attn_weights + spec_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # ---- SpectraKV 一次性 gather 压缩 (供 decode 复用; lm_eval 单 forward 用不到) ----
        # 此处会调用同一份缓存的 keep_idx, 不会重新算 leverage. 一致性由 _keep_idx 保证.
        past_key_value = self.kv_cache(past_key_value, None)

        # ---- 输出投影 ----
        attn_output = torch.matmul(attn_weights, value_states_rep)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output size mismatch: "
                f"expected {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ==========================================================================
# 顶层模型: 把每层的 self_attn 换成 SpectraLlamaAttention
# ==========================================================================
class SpectraLlamaForCausalLM(LlamaForCausalLM):
    """
    对齐 H2OLlamaForCausalLM 的用法:
        config.hh_size = ...
        config.recent_size = ...
        (可选) config.sink_size, config.jl_dim_multiplier, config.reg_lambda
        model = SpectraLlamaForCausalLM.from_pretrained(model_name, config=config)
    """

    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = SpectraLlamaAttention(config)


# ==========================================================================
# In-place patcher: 给 run_lm_eval_harness.py 用的, 接口对齐
# convert_kvcache_llama_heavy_recent
# ==========================================================================
def _resolve_spectra_budgets(config):
    """
    统一 "ratio vs. absolute size" 两种配置来源:
      - 优先用 config.hh_size / config.recent_size (如果显式设置了)
      - 否则用 config.heavy_ratio / config.recent_ratio × max_position_embeddings
    sink_size 取 config.sink_size, 默认 4.
    把结果写回 config, 方便 SpectraLlamaAttention 直接读.
    """
    base_len = getattr(config, "max_position_embeddings", 4096)

    if not hasattr(config, "hh_size") or config.hh_size is None:
        ratio = getattr(config, "heavy_ratio", 0.1)
        config.hh_size = max(int(ratio * base_len), 1)

    if not hasattr(config, "recent_size") or config.recent_size is None:
        ratio = getattr(config, "recent_ratio", 0.1)
        config.recent_size = max(int(ratio * base_len), 1)

    if not hasattr(config, "sink_size") or config.sink_size is None:
        config.sink_size = 4

    return config


def convert_kvcache_llama_spectra(model, config):
    """
    递归把 model 里所有 LlamaAttention 替换成 SpectraLlamaAttention.

    用法同 utils_lm_eval.convert_kvcache_llama_heavy_recent:
        model = AutoModelForCausalLM.from_pretrained(...)
        ckpt = copy.deepcopy(model.state_dict())
        model = convert_kvcache_llama_spectra(model, config)
        model.load_state_dict(ckpt)   # 权重名对齐, 可无损回填

    config 上会被自动补上 hh_size / recent_size / sink_size (见 _resolve_spectra_budgets).
    """
    config = _resolve_spectra_budgets(config)
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_spectra(module, config)
        if isinstance(module, LlamaAttention):
            model._modules[name] = SpectraLlamaAttention(config)
    return model
