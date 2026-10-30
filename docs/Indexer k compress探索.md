# Indexer k compress探索

**背景：**

1. topk模式，Sparseattention主干的计算量后期不随序列长度增加而增加，但是indexer一般都是o\(n^2\)复杂度，长文会耗时剧增成为性能瓶颈

2. 现有的压缩手法一般是对检索任务有损的

目标是做一个最efficient的indexer设计





**已有的研究 or 方案：**已有的 paper 有什么样的相关研究或者观测

1:  [deepseek v4](https://arxiv.org/pdf/2606.19348) 

这里的csa是可以参考的可训练的序列维度压缩的尝试，主要特点是引入了gate来做压缩，

$K = H W_K,\quad Z = H W_Z;\qquad 
$

$
S_{mi:m(i+1)-1} = \mathrm{Softmax}\big(Z_{mi:m(i+1)-1} + B\big),\qquad $

$

K^{\mathrm{Comp}}_i = \sum_{tile_i} S_j \odot K_j$

这个方法设计简单，实践有效

2: ** **[Hierarchical Sparse Attention](https://arxiv.org/abs/2607.02980) \&\& [LogSumExp和Softmax的泰勒展开](https://kexue.fm/archives/11814)

记第 $c$ 个 block 的 keys 为 $\mathbf{K}_c=\{\mathbf{k}_j\mid j\in\mathcal{T}_c\}$，对指定query其重要性为 block 内未归一化 attention score 之和：

$Z_c(\mathbf{q})=\sum_{j\in\mathcal{T}_c}\exp\left(\frac{\mathbf{q}^\top\mathbf{k}_j}{\sqrt d}\right)$

记 $f(\mathbf{q},\mathbf{K}_c)=\log Z_c(\mathbf{q})$。直接计算 $f$ 需要保存并读取整个 $\mathbf{K}_c$，即每个 block 的 $B$ 个 key。

在 $\mathbf{q}'_c$ 处，对 $f$ 关于 $\mathbf{q}$ 做一阶 Taylor 展开：

$\log Z_c(\mathbf{q})\approx\hat f(\mathbf{q};\mathbf{k}'_c,b'_c)=\frac{\mathbf{q}^\top\mathbf{k}'_c}{\sqrt d}+b'_c$

其中，

$\mathbf{k}'_c=\sum_{j\in\mathcal{T}_c}p_j\mathbf{k}_j,\quad b'_c=-\sum_{j\in\mathcal{T}_c}p_j\log p_j$，$s_j=\frac{(\mathbf{q}'_c)^\top\mathbf{k}_j}{\sqrt d},\quad p_j=\frac{\exp(s_j)}{\sum_{l\in\mathcal{T}_c}\exp(s_l)}$。

于是 $f(\mathbf{q},\mathbf{K}_c)$ 变为 $\hat f(\mathbf{q};\mathbf{k}'_c,b'_c)$：每个 block 从 $B$ 个 key 压缩为一个向量和一个标量，查询时只需一次点积，indexer cache 从 `[num blocks, B, d]` 降为 `[num blocks, d] + [num blocks, 1]`。

这个方案给了一个新的分析和改进平均池化压缩方案的视角，**找一个更合适的泰勒展开的起始点（可训练的）或许会有帮助。**

备注：在q=\[0,\.\.,0\]处展开这里的$\mathbf{k}'_c$就是$avg(K_c)$,$b'_c=ln(B)$,对应MoBA的avg pooling方案

3：[COBS: CUMULANT ORDER BLOCK SPARSE ATTENTION](https://arxiv.org/pdf/2607.09052) 

继承2的思路，不过在原点把泰勒展开进行到2阶，

$\log Z_c(\mathbf{q})\approx\hat f(\mathbf{q};\mathbf{k}'_c,b'_c)=\frac{1}{2d} \mathbf q^\top\boldsymbol\Sigma_c\mathbf q +\frac{\mathbf{q}^\top\mathbf{k}'_c}{\sqrt d}+b'_c$

其中

$\boldsymbol\Sigma_c = \frac1B \sum_{j\in\mathcal T_c} (\mathbf k_j-\mathbf k'_c) (\mathbf k_j-\mathbf k'_c)^\top.$

因为是在原点展开，$\mathbf k'_c = \frac1B\sum_{j\in\mathcal T_c}\mathbf k_j, \qquad b'_c=\log B,$

代价是每个block再额外存储一个\[headdim,headdim\]Hessian matrix，计算也更繁琐，\(部分工作把它对角化近似来节省计算存储成本，[Sparse Linear Attention: 当稀疏遇上线性注意力 · Haoyi Zhu](https://www.haoyizhu.site/blog/sparse-linear-attention/)\)，基于这个思路做一系列复杂的优化让他有一定的实际的压缩和加速，如果做更极致的压缩或许可以尝试

4. kmeans方法，

先对block内聚类，然后对簇内token非均匀压缩



**合板：**
kmeans\+lse的1/2阶泰勒展开导出的压缩手段



## 探究的点：



