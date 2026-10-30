# Sol-Attn 一阶泰勒（anchor）indexer

> 对应 `docs/Indexer k compress探索.md` 中 HSA 那条线的实现记录。代码在
> `techniques/sparse_backends/sol_attn/anchor.py`，kernel 改动在
> `techniques/sparse_backends/sol_attn/triton_ref/fwd.py`。

## 1. 它替换的是什么

Sol-Attn 对每个 64-token 键块的代理分数是 `s·q·k̄_j`（k̄ 是块内均值键），近似分支用
`64·exp(s·q·k̄_j)` 作分母、`exp(s·q·k̄_j)·ΣV_j` 作分子。这恰好是块的 log-partition

```
f_j(q) = log Σ_{t∈block j} exp(s·q·k_t)
```

在原点 `q'=0` 的一阶泰勒展开：`f_j(q) ≈ s·q·k̄_j + log 64`。

anchor indexer 把展开点挪到每个 head 的若干个 anchor 查询 `q'` 上：

```
p_t   = softmax_t( s·(q'/T)·k_t )          块内 softmax，T 是温度
k'_j  = Σ_t p_t k_t                         anchor 加权键
v'_j  = Σ_t p_t v_t                         anchor 加权值（均值，不是求和）
b'_j  = −Σ_t p_t log p_t                    熵 = log(有效键数)
f_j(q) ≈ s·q·k'_j + b'_j
```

性质（都有测试）：
- `q = q'` 时近似精确（切点）；对任意 q 近似值 ≤ 真值（log-sum-exp 凸）。
- `softmax(s·(q'/T)·k) = softmax(s·(q'/T)·k)`：温度等价于把展开点沿直线挪向原点，
  `T→∞` 严格退化为 mean pooling。所以现有方案是这个参数族的一个端点。

## 2. 对 Sol-Attn 各部分的改动

| 部件 | mean pooling | anchor |
|---|---|---|
| kc | 块均值键 `[B,N,H,D]` | `k'` 按 `[B,N,H·A,D]` 排布，伪 head = `h·A + a` |
| vc | 块求和值 | `v'` 加权均值，同样排布 |
| bias | 无（隐含 `log 64`） | `b'/ln2`，`[B,N,H·A]`，log2 单位，kernel 内按列加到分数 tile 上 |
| assign | 无 | `[B,N,H]` int32，每个查询块用哪个 anchor |
| 近似分母 | `Σ exp2(score)·len_j` | `Σ exp2(score)`，块长已在 bias 里 |
| 路由阈值 | `μ_i + τσ_i`，由 kc 一阶/二阶矩闭式得到 | 同样闭式：分数线性于 `[s·k'_j ; b'_j]`，查询特征 `[Q̄_i ; 1]`，维度 128→129 |

kernel（Triton 参考实现）新增编译期开关 `HAS_ANCHOR`：
1. 启动时按 `assign` 取伪 head 索引，kc/vc 读该 anchor 的切片；
2. 分数 tile 先对 64 个查询行取列均值再加 bias（避免尾块零填充行稀释 bias），再比阈值；
3. 近似分支的 `lengths` 乘子去掉。
`HAS_ANCHOR=False` 时代码路径与原来逐字相同。CuTe（SM89/90/100/120）与 Metal
后端未实现，`num_anchors>0` 时 CUDA 设备自动走 Triton。

## 3. anchor 怎么选

- `position`（默认）：把 N 个查询块按 token 顺序切成 A 个连续区间，每个区间的平均
  查询做 anchor。Morton/raster 顺序下一个区间就是一个时空区域，anchor 与它服务的
  查询共享 RoPE 相位。
- `kmeans`：以 position 为初值，对块均值查询做几轮 k-means。
- `A=1` 即全局平均查询。预期（也被合成数据证实）它基本退化为 mean pooling。

## 4. 用法

```python
from sol_attn import sol_attn
out = sol_attn(q, k, v, tau=1.0, num_anchors=8, anchor_temperature=2.0, anchor_mode="position")
```

模型侧环境变量（`models/{hunyuan_video,wan21_t2v_14b}/optimized/gpu_infer.py`）：
`*_SOL_ANCHORS`（默认 0 = 关闭）、`*_SOL_ANCHOR_TEMP`（默认 inf）、`*_SOL_ANCHOR_MODE`。

离线评估（不需要 kernel，CPU 可跑）：

```bash
cd techniques/sparse_backends
python tools/anchor_indexer_eval.py --tokens 4096 --anchors 1,4,8 --temps inf,4,2,1
python tools/anchor_indexer_eval.py --source dump.pt --heads 0,1 --density 0.15   # 真实 Q/K/V
```

指标：匹配预算下真实 top 块召回、代理分数与真实 log Z 的 Spearman、同预算 top-k
路由下 Sol-Attn 输出相对 dense 的 L2 误差（近似分支用各自的 summary）。

## 5. 合成数据上的初步结果（2048 tokens，2 heads，预算 15.6%，含 needle）

| indexer | recall | spearman | rel-L2 |
|---|---|---|---|
| mean pooling | 0.822 | 0.957 | 0.305 |
| A=1, T=1 | 0.819 | 0.958 | 0.305 |
| A=4, T=1 | 0.897 | 0.972 | 0.142 |
| A=8, T=2 | 0.944 | 0.997 | 0.096 |
| A=8, T=1 | 0.959 | 0.998 | 0.007 |

合成数据是按区域构造的，对 anchor 方案偏友好，只能说明实现方向正确，不能外推到
真实模型；下一步要用真实 Q/K/V dump 跑同一个脚本。

## 6. 待办

- 在 GPU 机器上跑 `pytest tests/test_anchor_indexer.py`（含 Triton kernel 对照参考实现的测试）。
- 从 Wan2.1-1.3B / SANA-Video dump post-RoPE Q/K/V，跑离线评估，扫 (A, T) 网格。
- 预处理目前是纯 PyTorch（持有 fp32 K/V 副本），有收益后再写 Triton 版。
- β 的 Gaussian 校准可能需要重新标定（bias 项会让分数分布右偏）。
