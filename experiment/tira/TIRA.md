# TIRA: Tiled Rank-1 Adaptation for High-Rank Parameter-Efficient Fine-Tuning

---

## 1. 核心思想

**TIRA** 将权重增量矩阵 $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 划分为 $M \times M$ 的块网格（每块大小 $n_{\text{out}} \times n_{\text{in}}$，其中 $n_{\text{out}} = d_{\text{out}}/M$，$n_{\text{in}} = d_{\text{in}}/M$），设计 $K$ 组参数，每组持有 $M$ 对可训练短向量 $(a_{k,m} \in \mathbb{R}^{n_{\text{in}}},\; b_{k,m} \in \mathbb{R}^{n_{\text{out}}})$，通过外积 $b_{k,m} \cdot a_{k,m}^T$ 构造秩-1 子块，放置在块网格的第 $m$ 行、第 $(m+k) \bmod M$ 列——即相对主对角线错位 $k$ 列。不同组的秩-1 块分布在不同的斜带上互不重叠，当 $K = M$ 时完全覆盖所有 $M^2$ 个块位置。从其等价形式去分析（详见第 2 节），每组参数可视为一个秩不超过 $M$ 的**稀疏 LoRA 专家**（$\mathbf{B}_k \mathbf{A}_k^T$），各专家间参数完全解耦；$K$ 个独立专家的叠加在不增加优化难度的前提下，将总秩从单一 LoRA 的 $r$ 提升至 $K \cdot M$，以 $K(d_{\text{out}} + d_{\text{in}})$ 的参数量实现结构化高秩适配。

以 $M=3$ 为例，各组占据的块位置（$\star$ = 秩-1 块，$\cdot$ = 零块）：

$$
\underbrace{\begin{bmatrix} \star & \cdot & \cdot \\ \cdot & \star & \cdot \\ \cdot & \cdot & \star \end{bmatrix}}_{k=0} \quad \underbrace{\begin{bmatrix} \cdot & \star & \cdot \\ \cdot & \cdot & \star \\ \star & \cdot & \cdot \end{bmatrix}}_{k=1} \quad \underbrace{\begin{bmatrix} \cdot & \cdot & \star \\ \star & \cdot & \cdot \\ \cdot & \star & \cdot \end{bmatrix}}_{k=2} \quad \xrightarrow{\text{叠加}} \quad \begin{bmatrix} \star & \star & \star \\ \star & \star & \star \\ \star & \star & \star \end{bmatrix}
$$

3 条斜带互不重叠，叠加后覆盖全部 $M^2 = 9$ 个块位置。

**符号汇总**：

| 符号 | 说明 |
| :--- | :--- |
| $d_{\text{out}},\; d_{\text{in}}$ | 输出/输入特征维度 |
| $M$ | 分块数，需满足 $M \mid d_{\text{out}}$ 且 $M \mid d_{\text{in}}$ |
| $K$ | 斜带数（参数组数），约束 $K \geq M$ 且为 $M$ 的倍数 |
| $\alpha$ | 全局缩放因子，类比 LoRA 的 $\alpha$，默认值为 $K$ |
| $n_{\text{out}} = d_{\text{out}}/M,\; n_{\text{in}} = d_{\text{in}}/M$ | 每个块的行数/列数 |
| $a_{k,m} \in \mathbb{R}^{n_{\text{in}}}$ | 输入侧短向量，kaiming 初始化（类比 LoRA 的 $A$） |
| $b_{k,m} \in \mathbb{R}^{n_{\text{out}}}$ | 输出侧短向量，零初始化（类比 LoRA 的 $B$） |

---

## 2. 构造形式与等价性

### 2.1 秩-1 块错位叠加构造形式（原文方法）

以 $K=M=3,\; d_{\text{out}}=d_{\text{in}}=9,\; n_{\text{out}}=\frac {d_{\text{out}} }{M} = \frac {d_{\text{in}} }{M} =  n_{\text{in}}=3$ 为例

- 组 $k=0$：$b_{0,m}$ 分量为 $a, b, c$，$a_{0,m}$ 分量为 $m, n, o$
- 组 $k=1$：$b_{1,m}$ 分量为 $e, f, g$，$a_{1,m}$ 分量为 $q, r, s$
- 组 $k=2$：$b_{2,m}$ 分量为 $i, j, k$，$a_{2,m}$ 分量为 $u, v, w$

其中下标 $1,2,3$ 分别对应向量索引 $m=0,1,2$。

每对短向量通过外积生成一个秩-1 块：

$$
C_{k,m} = b_{k,m} \cdot a_{k,m}^T \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}
$$

将 $M$ 个秩-1 块放置在块网格的第 $k$ 条斜带上——第 $m$ 个块放在第 $m$ 行、第 $(m+k) \bmod M$ 列，其余位置为零：

$$
[\Delta W_k]_{i,j} = \begin{cases} C_{k,m} & \text{if } i = m,\; j = (m+k) \bmod M \\ \mathbf{0} & \text{otherwise} \end{cases}
$$

$\Delta W_k \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是仅在第 $k$ 条斜带上有 $M$ 个非零块的稀疏矩阵。

以 $k=0$ 组为例，3 个秩-1 块放置在主斜带上：

$$
C_{0,0} = \begin{pmatrix} a_1 \\ b_1 \\ c_1 \end{pmatrix} \begin{pmatrix} m_1 & n_1 & o_1 \end{pmatrix}, \quad C_{0,1} = \begin{pmatrix} a_2 \\ b_2 \\ c_2 \end{pmatrix} \begin{pmatrix} m_2 & n_2 & o_2 \end{pmatrix}, \quad C_{0,2} = \begin{pmatrix} a_3 \\ b_3 \\ c_3 \end{pmatrix} \begin{pmatrix} m_3 & n_3 & o_3 \end{pmatrix}
$$

以 $M=3,\, K=3$ 为例，各组的 $\Delta W_k$ 结构（$C_{k,m}$ 简写为 $C$，$\mathbf{0}$ 为零块）：

**第 0 组 $k=0$**（主斜带）：

$$
\Delta W_0 = \begin{bmatrix}
C_{0,0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & C_{0,1} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & C_{0,2}
\end{bmatrix}
$$

**第 1 组 $k=1$**（偏移 1 列）：

$$
\Delta W_1 = \begin{bmatrix}
\mathbf{0} & C_{1,0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & C_{1,1} \\
C_{1,2} & \mathbf{0} & \mathbf{0}
\end{bmatrix}
$$

**第 2 组 $k=2$**（偏移 2 列）：

$$
\Delta W_2 = \begin{bmatrix}
\mathbf{0} & \mathbf{0} & C_{2,0} \\
C_{2,1} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & C_{2,2} & \mathbf{0}
\end{bmatrix}
$$

$K$ 组叠加得到总权重增量：

$$
\Delta W = \frac{\alpha}{K} \sum_{k=0}^{K-1} \Delta W_k
$$

其中 $\Delta W_k$ 是第 $k$ 组构造的稀疏块矩阵（仅在第 $k$ 条斜带上非零）。三组叠加后：

$$
\Delta W_{\text{block}} = \frac{\alpha}{K} \begin{bmatrix} C_{0,0} & C_{1,0} & C_{2,0} \\ C_{2,1} & C_{0,1} & C_{1,1} \\ C_{1,2} & C_{2,2} & C_{0,2} \end{bmatrix}
$$

每个块位置恰好被一组覆盖，块内秩为 1。展开为标量形式——每个 $3 \times 3$ 连续子块均为秩-1 外积矩阵，且**块内所有元素的下标一致**（如子块 $(0,0)$ 全为下标 1，子块 $(1,1)$ 全为下标 2）：

$$
\Delta W_{\text{block}} = \frac{\alpha}{K} \begin{bmatrix} a_1 m_1 & a_1 n_1 & a_1 o_1 & e_1 q_1 & e_1 r_1 & e_1 s_1 & i_1 u_1 & i_1 v_1 & i_1 w_1 \\ b_1 m_1 & b_1 n_1 & b_1 o_1 & f_1 q_1 & f_1 r_1 & f_1 s_1 & j_1 u_1 & j_1 v_1 & j_1 w_1 \\ c_1 m_1 & c_1 n_1 & c_1 o_1 & g_1 q_1 & g_1 r_1 & g_1 s_1 & k_1 u_1 & k_1 v_1 & k_1 w_1 \\ i_2 u_2 & i_2 v_2 & i_2 w_2 & a_2 m_2 & a_2 n_2 & a_2 o_2 & e_2 q_2 & e_2 r_2 & e_2 s_2 \\ j_2 u_2 & j_2 v_2 & j_2 w_2 & b_2 m_2 & b_2 n_2 & b_2 o_2 & f_2 q_2 & f_2 r_2 & f_2 s_2 \\ k_2 u_2 & k_2 v_2 & k_2 w_2 & c_2 m_2 & c_2 n_2 & c_2 o_2 & g_2 q_2 & g_2 r_2 & g_2 s_2 \\ e_3 q_3 & e_3 r_3 & e_3 s_3 & i_3 u_3 & i_3 v_3 & i_3 w_3 & a_3 m_3 & a_3 n_3 & a_3 o_3 \\ f_3 q_3 & f_3 r_3 & f_3 s_3 & j_3 u_3 & j_3 v_3 & j_3 w_3 & b_3 m_3 & b_3 n_3 & b_3 o_3 \\ g_3 q_3 & g_3 r_3 & g_3 s_3 & k_3 u_3 & k_3 v_3 & k_3 w_3 & c_3 m_3 & c_3 n_3 & c_3 o_3 \end{bmatrix}
$$

**初始化策略**：

- $a_{k,m}$：Kaiming uniform 初始化（类比 LoRA 的 $A$ 矩阵）
- $b_{k,m}$：零初始化（类比 LoRA 的 $B$ 矩阵），保证 $\Delta W = 0$ 于训练起始

### 2.2 等价多稀疏LoRA专家构造

TIRA 的权重增量也可以由 $K$ 组稀疏矩阵 $\mathbf{B}_k \in \mathbb{R}^{d_{\text{out}} \times M}$ 与 $\mathbf{A}_k^T \in \mathbb{R}^{M \times d_{\text{in}}}$ 的乘积之和来构建。在该形式中，$\mathbf{B}_k$ 和 $\mathbf{A}_k$ 均为稀疏矩阵——同一组参数的各分量按 $\bmod\, M$ **交错排列**在行/列中，而非连续放置。

**第 0 组 $k=0$**（无错位）：
$$
\mathbf{B}_0 \mathbf{A}_0^T = \begin{bmatrix} a_1 & 0 & 0 \\ 0 & a_2 & 0 \\ 0 & 0 & a_3 \\ b_1 & 0 & 0 \\ 0 & b_2 & 0 \\ 0 & 0 & b_3 \\ c_1 & 0 & 0 \\ 0 & c_2 & 0 \\ 0 & 0 & c_3 \end{bmatrix} \begin{bmatrix} m_1 & 0 & 0 & n_1 & 0 & 0 & o_1 & 0 & 0 \\ 0 & m_2 & 0 & 0 & n_2 & 0 & 0 & o_2 & 0 \\ 0 & 0 & m_3 & 0 & 0 & n_3 & 0 & 0 & o_3 \end{bmatrix}= \begin{bmatrix} a_1 m_1 & 0 & 0 & a_1 n_1 & 0 & 0 & a_1 o_1 & 0 & 0 \\ 0 & a_2 m_2 & 0 & 0 & a_2 n_2 & 0 & 0 & a_2 o_2 & 0 \\ 0 & 0 & a_3 m_3 & 0 & 0 & a_3 n_3 & 0 & 0 & a_3 o_3 \\ b_1 m_1 & 0 & 0 & b_1 n_1 & 0 & 0 & b_1 o_1 & 0 & 0 \\ 0 & b_2 m_2 & 0 & 0 & b_2 n_2 & 0 & 0 & b_2 o_2 & 0 \\ 0 & 0 & b_3 m_3 & 0 & 0 & b_3 n_3 & 0 & 0 & b_3 o_3 \\ c_1 m_1 & 0 & 0 & c_1 n_1 & 0 & 0 & c_1 o_1 & 0 & 0 \\ 0 & c_2 m_2 & 0 & 0 & c_2 n_2 & 0 & 0 & c_2 o_2 & 0 \\ 0 & 0 & c_3 m_3 & 0 & 0 & c_3 n_3 & 0 & 0 & c_3 o_3 \end{bmatrix}
$$

**第 1 组 $k=1$**（列循环偏移 1）：

$$
\mathbf{B}_1 \mathbf{A}_1^T = \begin{bmatrix} e_1 & 0 & 0 \\ 0 & e_2 & 0 \\ 0 & 0 & e_3 \\ f_1 & 0 & 0 \\ 0 & f_2 & 0 \\ 0 & 0 & f_3 \\ g_1 & 0 & 0 \\ 0 & g_2 & 0 \\ 0 & 0 & g_3 \end{bmatrix} \begin{bmatrix} 0 & q_2 & 0 & 0 & r_2 & 0 & 0 & s_2 & 0 \\ 0 & 0 & q_3 & 0 & 0 & r_3 & 0 & 0 & s_3 \\ q_1 & 0 & 0 & r_1 & 0 & 0 & s_1 & 0 & 0 \end{bmatrix}= \begin{bmatrix} 0 & e_1 q_2 & 0 & 0 & e_1 r_2 & 0 & 0 & e_1 s_2 & 0 \\ 0 & 0 & e_2 q_3 & 0 & 0 & e_2 r_3 & 0 & 0 & e_2 s_3 \\ e_3 q_1 & 0 & 0 & e_3 r_1 & 0 & 0 & e_3 s_1 & 0 & 0 \\ 0 & f_1 q_2 & 0 & 0 & f_1 r_2 & 0 & 0 & f_1 s_2 & 0 \\ 0 & 0 & f_2 q_3 & 0 & 0 & f_2 r_3 & 0 & 0 & f_2 s_3 \\ f_3 q_1 & 0 & 0 & f_3 r_1 & 0 & 0 & f_3 s_1 & 0 & 0 \\ 0 & g_1 q_2 & 0 & 0 & g_1 r_2 & 0 & 0 & g_1 s_2 & 0 \\ 0 & 0 & g_2 q_3 & 0 & 0 & g_2 r_3 & 0 & 0 & g_2 s_3 \\ g_3 q_1 & 0 & 0 & g_3 r_1 & 0 & 0 & g_3 s_1 & 0 & 0 \end{bmatrix}
$$

**第 2 组 $k=2$**（列循环偏移 2）：

$$
\mathbf{B}_2 \mathbf{A}_2^T = \begin{bmatrix} i_1 & 0 & 0 \\ 0 & i_2 & 0 \\ 0 & 0 & i_3 \\ j_1 & 0 & 0 \\ 0 & j_2 & 0 \\ 0 & 0 & j_3 \\ k_1 & 0 & 0 \\ 0 & k_2 & 0 \\ 0 & 0 & k_3 \end{bmatrix} \begin{bmatrix} 0 & 0 & u_3 & 0 & 0 & v_3 & 0 & 0 & w_3 \\ u_1 & 0 & 0 & v_1 & 0 & 0 & w_1 & 0 & 0 \\ 0 & u_2 & 0 & 0 & v_2 & 0 & 0 & w_2 & 0 \end{bmatrix}= \begin{bmatrix} 0 & 0 & i_1 u_3 & 0 & 0 & i_1 v_3 & 0 & 0 & i_1 w_3 \\ i_2 u_1 & 0 & 0 & i_2 v_1 & 0 & 0 & i_2 w_1 & 0 & 0 \\ 0 & i_3 u_2 & 0 & 0 & i_3 v_2 & 0 & 0 & i_3 w_2 & 0 \\ 0 & 0 & j_1 u_3 & 0 & 0 & j_1 v_3 & 0 & 0 & j_1 w_3 \\ j_2 u_1 & 0 & 0 & j_2 v_1 & 0 & 0 & j_2 w_1 & 0 & 0 \\ 0 & j_3 u_2 & 0 & 0 & j_3 v_2 & 0 & 0 & j_3 w_2 & 0 \\ 0 & 0 & k_1 u_3 & 0 & 0 & k_1 v_3 & 0 & 0 & k_1 w_3 \\ k_2 u_1 & 0 & 0 & k_2 v_1 & 0 & 0 & k_2 w_1 & 0 & 0 \\ 0 & k_3 u_2 & 0 & 0 & k_3 v_2 & 0 & 0 & k_3 w_2 & 0 \end{bmatrix}
$$

总权重增量——秩-1 结构以**交错方式**分布，同一组参数的元素分散在矩阵的不同行/列（按 $\bmod\, M$ 交错排列），相邻元素来自不同组：

$$
\Delta W_{\text{expert}} = \frac{\alpha}{K} \left( \mathbf{B}_0 \mathbf{A}_0^T + \mathbf{B}_1 \mathbf{A}_1^T + \mathbf{B}_2 \mathbf{A}_2^T \right)= \frac{\alpha}{K} \begin{bmatrix} a_1 m_1 & e_1 q_2 & i_1 u_3 & a_1 n_1 & e_1 r_2 & i_1 v_3 & a_1 o_1 & e_1 s_2 & i_1 w_3 \\ i_2 u_1 & a_2 m_2 & e_2 q_3 & i_2 v_1 & a_2 n_2 & e_2 r_3 & i_2 w_1 & a_2 o_2 & e_2 s_3 \\ e_3 q_1 & i_3 u_2 & a_3 m_3 & e_3 r_1 & i_3 v_2 & a_3 n_3 & e_3 s_1 & i_3 w_2 & a_3 o_3 \\ b_1 m_1 & f_1 q_2 & j_1 u_3 & b_1 n_1 & f_1 r_2 & j_1 v_3 & b_1 o_1 & f_1 s_2 & j_1 w_3 \\ j_2 u_1 & b_2 m_2 & f_2 q_3 & j_2 v_1 & b_2 n_2 & f_2 r_3 & j_2 w_1 & b_2 o_2 & f_2 s_3 \\ f_3 q_1 & j_3 u_2 & b_3 m_3 & f_3 r_1 & j_3 v_2 & b_3 n_3 & f_3 s_1 & j_3 w_2 & b_3 o_3 \\ c_1 m_1 & g_1 q_2 & k_1 u_3 & c_1 n_1 & g_1 r_2 & k_1 v_3 & c_1 o_1 & g_1 s_2 & k_1 w_3 \\ k_2 u_1 & c_2 m_2 & g_2 q_3 & k_2 v_1 & c_2 n_2 & g_2 r_3 & k_2 w_1 & c_2 o_2 & g_2 s_3 \\ g_3 q_1 & k_3 u_2 & c_3 m_3 & g_3 r_1 & k_3 v_2 & c_3 n_3 & g_3 s_1 & k_3 w_2 & c_3 o_3 \end{bmatrix}
$$

### 2.3 对比与等价性

| 特征 | 块错位叠加（2.1） | 多稀疏LoRA专家（2.2） |
| :--- | :--- | :--- |
| 参数量 | $K \cdot (d_{\text{out}} + d_{\text{in}}) = 54$ | 相同 |
| 秩-1 结构 | 每个 $n \times n$ **连续子块**为秩-1 | 每个 $n \times n$ **交错子块**为秩-1 |
| 块内下标 | 一致（全为 1、2 或 3） | 混合（如 $a_1 m_1,\, e_1 q_2,\, i_1 u_3$ 同行） |
| 可表达函数类 | 相同（参数空间维数相同） | 相同 |

两种构造使用相同数量的标量参数，区别仅在于参数到矩阵位置的映射方式：块叠加形式将同一向量的分量**连续排列**，专家形式将其按 $\bmod\, M$ **交错排列**。二者在深度学习反向传播的梯度优化下是等价的，可表达的 $\Delta W$ 空间完全相同。本文采用块错位叠加构造，因其连续块结构更适合高效前向传播实现（见第 6 节）。

---

## 3. 错位叠加

### 3.1 单层覆盖（$K = M$）

以 $M = 3$ 为例，每条斜带恰好被一组覆盖，叠加后所有 $M^2$ 个块位置均为独立的秩-1 矩阵：

$$
\Delta W = \frac{\alpha}{K} \begin{bmatrix}
C_{0,0} & C_{1,0} & C_{2,0} \\
C_{2,1} & C_{0,1} & C_{1,1} \\
C_{1,2} & C_{2,2} & C_{0,2}
\end{bmatrix}
$$

每个块位置恰好被一组覆盖，块内秩为 1。

### 3.2 多层覆盖（$K = LM,\; L \geq 2$）

以 $M = 3,\, K = 6\;(L=2)$ 为例，每条斜带被 2 组参数叠加，同一块位置上两个秩-1 矩阵相加，块内秩最高为 2：

$$
\Delta W = \frac{\alpha}{K} \begin{bmatrix}
C_{0,0}+C_{3,0} & C_{1,0}+C_{4,0} & C_{2,0}+C_{5,0} \\
C_{2,1}+C_{5,1} & C_{0,1}+C_{3,1} & C_{1,1}+C_{4,1} \\
C_{1,2}+C_{4,2} & C_{2,2}+C_{5,2} & C_{0,2}+C_{3,2}
\end{bmatrix}
$$

一般地，$L = K/M$ 组参数叠加在同一块位置，块内秩最高为 $\min(L,\, n_{\text{out}},\, n_{\text{in}})$。

---

## 4. 参数量分析

K组，每组有 $M$ 对短向量：$M$ 个 $a_{k,m} \in \mathbb{R}^{n_{\text{in}}}$ 和 $M$ 个 $b_{k,m} \in \mathbb{R}^{n_{\text{out}}}$。

$$
\text{参数量} = K \cdot M \cdot (n_{\text{out}} + n_{\text{in}}) = K \cdot (d_{\text{out}} + d_{\text{in}})
$$

---

## 5. 秩分析

### 5.1 单分支秩贡献

第 $k$ 组的 $M$ 个秩-1 块位于不同的行块和列块（斜带上无重叠），因此：

$$
\text{rank}(\Delta W_k) = M
$$

### 5.2 总秩上界

记 $L = K/M$（每条斜带的叠加层数），同一块位置上 $L$ 个秩-1 矩阵叠加后块内秩最高为 $r_b = \min(L,\, n_{\text{out}},\, n_{\text{in}})$。$M^2$ 个块共贡献秩上界 $M^2 \cdot r_b$，但不能超过矩阵本身的维度限制：

$$
\text{rank}(\Delta W) \leq \min\!\left( M^2 \cdot \min\!\left(L,\, n_{\text{out}},\, n_{\text{in}}\right),\; d_{\text{out}},\; d_{\text{in}} \right)
$$

| 情况     | 条件       | 秩上界                                                       |
| :------- | :--------- | :----------------------------------------------------------- |
| 单层覆盖 | $K = M$    | $\min(M^2,\, d_{\text{out}},\, d_{\text{in}})$              |
| 多层覆盖 | $K = LM$   | $\min(M^2 \cdot \min(L, n_{\text{out}}, n_{\text{in}}),\, d_{\text{out}},\, d_{\text{in}})$ |

### 5.3 配置示例

以典型 Transformer 层为例：

| 配置 | $d_{\text{out}}$ | $d_{\text{in}}$ | $M$ | $K$ | $L=K/M$ | 参数量 | 秩上界 | 满秩 | LoRA 秩 |
| :--- | :--------------- | :-------------- | :-- | :-- | :------- | :----- | :----- | :--- | :------ |
| A | 1024 | 1024 | 16 | 16 | 1 | 32,768 | $\min(16^2 \cdot 1,\, 1024,\, 1024) = 256$ | 否 | 16 |
| B | 1024 | 1024 | 32 | 32 | 1 | 65,536 | $\min(32^2 \cdot 1,\, 1024,\, 1024) = 1024$ | 是 | 32 |
| C | 1024 | 1024 | 16 | 32 | 2 | 65,536 | $\min(16^2 \cdot 2,\, 1024,\, 1024) = 512$ | 否 | 32 |
| D | 1024 | 4096 | 32 | 32 | 1 | 163,840 | $\min(32^2 \cdot 1,\, 1024,\, 4096) = 1024$ | 是 | 32 |
| E | 4096 | 1024 | 32 | 64 | 2 | 327,680 | $\min(32^2 \cdot 2,\, 4096,\, 1024) = 1024$ | 是 | 64 |
| F | 1024 | 4096 | 64 | 64 | 1 | 327,680 | $\min(64^2 \cdot 1,\, 1024,\, 4096) = 1024$ | 是 | 64 |

配置 E,F 中 $M^2 > d_{\text{out}}$，秩上界被矩阵维度截断为 1024。

### 5.4 达到最大秩的条件

1. 所有短向量 $a_{k,m}, b_{k,m}$ 非零
2. 各秩-1 块 $b_{k,m} a_{k,m}^T$ 线性无关

---

## 6. 高效前向传播

代码避免显式构造 $d_{\text{out}} \times d_{\text{in}}$ 的完整 $\Delta W$，而是利用块结构在前向传播中高效计算 $\Delta W \cdot x$：

1. 将输入 $x$ 按输入维度重塑为 $M$ 个块：$x_{\text{blocks}} \in \mathbb{R}^{\text{batch} \times M \times n_{\text{in}}}$
2. 对每组 $k$，将 $a$ 按偏移对齐到输入块坐标
3. 批量矩阵乘法计算 $x$ 与 $a$ 的内积（标量激活）
4. 将激活值重新索引到输出块坐标
5. 批量矩阵乘法用 $b$ 缩放得到输出块贡献

计算复杂度：$O(K \times \text{batch} \times (d_{\text{in}} + d_{\text{out}}))$，远低于朴素的 $O(\text{batch} \times d_{\text{in}} \times d_{\text{out}})$。

---

## 7. 与标准 LoRA 对比

| 特性     | LoRA                              | TIRA                                   |
| :------- | :-------------------------------- | :----------------------------------------- |
| 可训练参数 | 两个稠密矩阵 $B \in \mathbb{R}^{d_{\text{out}} \times r},\, A \in \mathbb{R}^{r \times d_{\text{in}}}$ | $K \times M$ 对短向量 $(a_{k,m}, b_{k,m})$ |
| 参数量   | $r(d_{\text{out}}+d_{\text{in}})$ | $K(d_{\text{out}}+d_{\text{in}})$          |
| $\Delta W$ 结构 | 稠密低秩（秩 $= r$）             | 块稀疏，每块秩-1，错位斜带排列           |
| 满参数秩 | $r$                               | $\min(M^2,\, d_{\text{out}},\, d_{\text{in}})$（当 $K=M$） |
| 错位机制 | 无                                | 列块索引偏移 $(m+k) \bmod M$               |
| 前向复杂度 | $O(\text{batch} \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$ | $O(K \cdot \text{batch} \cdot (d_{\text{in}} + d_{\text{out}}))$ |

**核心优势**：当 $K = M$ 时，TIRA 以与 LoRA（$r=K$）相同的参数量，达到 $M^2$ 的秩上界（vs LoRA 的 $r$），实现参数效率的显著提升。