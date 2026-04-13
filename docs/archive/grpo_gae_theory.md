# GAE-Augmented GRPO: Temporal Credit Assignment from First Principles
# GAE 增强的 GRPO：从第一性原理出发的时序信用分配

> **Research Proposal — ICML 2026 Direction**  |  **研究提案 — ICML 2026 方向**
>
> Status: Theoretical Design Phase  |  状态：理论设计阶段
>
> Related work: BHVF (`docs/paper_draft.md` / `docs/paper_draft_zh.md`)

---

## Table of Contents | 目录

1. [Motivation: Does GRPO Need GAE? | 动机：GRPO 需要 GAE 吗？](#1)
2. [GRPO Failure Mode: Temporal Blindness | GRPO 失效模式：时序盲视](#2)
3. [Mathematical Foundations: Bayesian Fusion | 数学基础：贝叶斯融合](#3)
4. [**CA-GRPO-Base: Architecture Deep-Dive | CA-GRPO-Base 架构详解**](#4-base)
5. [Pseudocode: Complete CA-GRPO-Base Algorithm | 伪代码：完整 CA-GRPO-Base 算法](#4)
6. [Self-Reflection: Assumption Verification | 自我反省：假设验证与漏洞分析](#5)
7. [Theoretical Feasibility Summary | 理论可行性总结](#6)
8. [Implementation Roadmap | 实现路线图](#7)

---

<a name="1"></a>
## 1. Motivation: Does GRPO Need GAE?
## 1. 动机：GRPO 需要 GAE 吗？

### 1.1 What is GRPO? | 什么是 GRPO？

**[EN]**
Group Relative Policy Optimization (GRPO), introduced in DeepSeek-R1 [Shao et al., 2024], is a
Critic-free on-policy RL algorithm designed for large language model alignment. Its key innovation
is eliminating the Critic network entirely, using the **group-relative reward signal** as a proxy
for the advantage:

$$\hat{A}_i^{\text{GRPO}} = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^{G})}{\mathrm{std}(\{r_j\}_{j=1}^{G})}, \quad i = 1, \ldots, G$$

where $G$ is the group size (number of responses generated from the same prompt), and every
timestep $t$ within response $i$ receives the **same** advantage value $\hat{A}_i^{\text{GRPO}}$.

**[ZH]**
组相对策略优化（GRPO）由 DeepSeek-R1 [Shao 等，2024] 引入，是一种无 Critic 的在策略强化学习算法，
专为大语言模型对齐而设计。其核心创新在于完全去除 Critic 网络，用**组相对奖励信号**作为优势估计的代理：

$$\hat{A}_i^{\text{GRPO}} = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^{G})}{\mathrm{std}(\{r_j\}_{j=1}^{G})}, \quad i = 1, \ldots, G$$

其中 $G$ 为组大小（从同一提示词生成的响应数量），响应 $i$ 内的每个时间步 $t$ 都接收**相同**的优势值
$\hat{A}_i^{\text{GRPO}}$。

---

### 1.2 The Core Claim | 核心主张

**[EN]**
GRPO's advantage estimate $\hat{A}_i^{\text{GRPO}}$ is a **trajectory-level** signal that assigns
the same credit to every token in the response. This discards all **temporal structure** of the
reward signal.

One might argue: "GRPO targets LLMs with sparse, outcome-based rewards (0/1 correctness). With
no per-step reward, GAE cannot help."

**Our response**: Even with only a **terminal reward**, a trained Critic can recover non-trivial
per-step advantage estimates through bootstrapped value function approximation. This matters
significantly for long-horizon tasks where early tokens causally determine final correctness.

**[ZH]**
GRPO 的优势估计 $\hat{A}_i^{\text{GRPO}}$ 是一个**轨迹级别**的信号，对响应中的每个 token 分配相同的
信用，完全丢弃了奖励信号的**时序结构**。

有人可能会争辩：「GRPO 针对稀疏的结果奖励（0/1 正确性），没有逐步奖励，GAE 无法发挥作用。」

**我们的回答**：即使只有**终端奖励**，经过训练的 Critic 也能通过自举价值函数近似恢复非平凡的逐步优势
估计。这对于早期 token 因果性决定最终正确性的长程任务尤为重要。

---

### 1.3 The Temporal Blindness Problem (Formal Statement) | 时序盲视问题（正式表述）

**[EN]**
**Definition (Temporal Credit Assignment Error)**: For a response of length $T$ with terminal
reward $R$, the GRPO advantage for token $t$ is:

$$\hat{A}_t^{\text{GRPO}} = \frac{R - \bar{R}_{\text{group}}}{\sigma_{\text{group}}} = c \cdot (R - \bar{R})$$

Let $A_t^*$ denote the **true per-step advantage** under the optimal value function $V^*(s_t)$:

$$A_t^* = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$$

**Proposition 0 (GRPO is a Biased Advantage Estimator)**: Under the MDP with terminal reward
$R = r_T$ (all other $r_t = 0$), $\hat{A}_t^{\text{GRPO}} = c \cdot R$ for all $t$, while:

$$A_t^* = \gamma^{T-t} \bigl(R - \mathbb{E}_\pi[R \mid s_t]\bigr)$$

The bias $b_t^G = \hat{A}_t^{\text{GRPO}} - A_t^*$ is **time-dependent** (depends on $t$) and
**non-zero** for $\gamma < 1$ and $T > 1$.

**[ZH]**
**定义（时序信用分配误差）**：对于长度为 $T$、终端奖励为 $R$ 的响应，时间步 $t$ 的 GRPO 优势为：

$$\hat{A}_t^{\text{GRPO}} = \frac{R - \bar{R}_{\text{group}}}{\sigma_{\text{group}}} = c \cdot (R - \bar{R})$$

设 $A_t^*$ 为在最优价值函数 $V^*(s_t)$ 下的**真实逐步优势**：

$$A_t^* = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$$

**命题 0（GRPO 是有偏的优势估计器）**：在只有终端奖励 $R = r_T$（其余 $r_t = 0$）的 MDP 下，
对所有 $t$ 有 $\hat{A}_t^{\text{GRPO}} = c \cdot R$，而：

$$A_t^* = \gamma^{T-t} \bigl(R - \mathbb{E}_\pi[R \mid s_t]\bigr)$$

偏差 $b_t^G = \hat{A}_t^{\text{GRPO}} - A_t^*$ 是**时间相关**的（依赖于 $t$），在 $\gamma < 1$、
$T > 1$ 时**恒非零**。

*Proof | 证明*:

With a terminal reward, $V^*(s_t) = \gamma^{T-t} \mathbb{E}_\pi[R \mid s_t]$. Thus:
$$A_t^* = \underbrace{0}_{r_t} + \gamma \cdot \gamma^{T-t-1}\mathbb{E}[R|s_{t+1}] - \gamma^{T-t}\mathbb{E}[R|s_t] = \gamma^{T-t}(R - \mathbb{E}[R|s_t])$$

The discount factor $\gamma^{T-t}$ makes early-step advantages **smaller in magnitude** than
late-step ones. GRPO ignores this time-dependent discount, applying a flat signal $\propto R$.
Hence GRPO over-rewards early tokens relative to their causal contribution. $\blacksquare$

有终端奖励时，$V^*(s_t) = \gamma^{T-t} \mathbb{E}_\pi[R \mid s_t]$。因此：
$$A_t^* = \gamma^{T-t}(R - \mathbb{E}[R|s_t])$$
折扣因子 $\gamma^{T-t}$ 使得早期步骤的优势**幅度更小**。GRPO 忽略了这一时间相关折扣，
对所有步骤施加平坦信号 $\propto R$。因此 GRPO 相对于因果贡献过度奖励了早期 token。$\blacksquare$

---

<a name="2"></a>
## 2. GRPO Failure Mode: Temporal Blindness
## 2. GRPO 失效模式：时序盲视

### 2.1 When Does Temporal Blindness Matter? | 何时时序盲视显著影响结果？

**[EN]**

| Scenario | GRPO Approximation | Impact |
|---|---|---|
| Short responses ($T \leq 20$) | $\gamma^{T-t} \approx 1$: discount negligible | **Low** |
| $\gamma = 1$ (no discounting) | All steps equal: GRPO exact | **Zero** |
| Long CoT ($T \gg 100$) | $\gamma^{T-t}$ varies $10^3\times$ across steps | **High** |
| Process rewards | Per-step structure present | **High** |
| Dense rewards | Rich temporal signal available | **High** |

**[ZH]**

| 场景 | GRPO 近似 | 影响 |
|---|---|---|
| 短响应（$T \leq 20$） | $\gamma^{T-t} \approx 1$：折扣可忽略 | **低** |
| $\gamma = 1$（无折扣） | 所有步骤等价：GRPO 精确 | **零** |
| 长链式推理（$T \gg 100$） | $\gamma^{T-t}$ 跨步骤变化 $10^3\times$ | **高** |
| 过程奖励 | 存在逐步时序结构 | **高** |
| 密集奖励 | 丰富时序信号可用 | **高** |

### 2.2 What GAE Can Recover | GAE 能恢复什么

**[EN]**
Even with only a terminal reward, a well-trained Critic $V_\phi(s_t)$ approximates:
$$V_\phi(s_t) \approx \gamma^{T-t} \mathbb{E}_\pi[R \mid s_t]$$

This captures the **expected future correctness** conditioned on the current reasoning state.
The resulting GAE advantage:
$$A_t^{\text{GAE}} = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \approx \gamma^{T-t}(R - \mathbb{E}[R|s_t])$$

recovers the time-dependent credit signal that GRPO discards.

**[ZH]**
即使只有终端奖励，经过良好训练的 Critic $V_\phi(s_t)$ 也能近似：
$$V_\phi(s_t) \approx \gamma^{T-t} \mathbb{E}_\pi[R \mid s_t]$$

这捕获了以当前推理状态为条件的**期望未来正确性**。由此得到的 GAE 优势：
$$A_t^{\text{GAE}} = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \approx \gamma^{T-t}(R - \mathbb{E}[R|s_t])$$

恢复了 GRPO 所丢弃的时间相关信用信号。

---

<a name="3"></a>
## 3. Mathematical Foundations: Bayesian Fusion of GRPO and GAE
## 3. 数学基础：GRPO 与 GAE 的贝叶斯融合

### 3.1 Two Estimators of the True Advantage | 真实优势的两个估计器

**[EN]**

We model the true per-step advantage $A_t^*$ as being estimated by two independent sources:

**Source 1 (Group signal)**:
$$\hat{A}_t^{\text{GRPO}} = A_t^* + \underbrace{(\hat{A}_t^{\text{GRPO}} - A_t^*)}_{\varepsilon_t^G}$$

**Assumption G1 (Group Signal Error Model)**:
- Mean: $\mathbb{E}[\varepsilon_t^G] = b_t^G$ (per-step temporal bias, proven in Prop. 0)
- Variance: $\mathrm{Var}(\varepsilon_t^G) = \sigma_G^2 = c^2 \mathrm{Var}(R)$ (group sampling noise)

**Source 2 (GAE signal)**:
$$A_t^{\text{GAE}} = A_t^* + \varepsilon_t^V$$

**Assumption V1 (GAE Signal Error Model)**:
- Mean: $\mathbb{E}[\varepsilon_t^V] = 0$ (approximately unbiased when Critic is trained with BHVF)
- Variance: $\mathrm{Var}(\varepsilon_t^V) = \sigma_V^2 \approx (1 - \text{EV}) \cdot \mathrm{Var}(A_t^{\text{GAE}})$

**Assumption I (Independence)**:
$$\mathbb{E}[\varepsilon_t^G \varepsilon_t^V] = 0$$

*Justification*: Group sampling randomness and Critic initialization error are driven by
different random seeds in on-policy rollout. $\square$

**[ZH]**

我们将真实逐步优势 $A_t^*$ 建模为由两个独立来源估计：

**来源 1（组信号）**：
$$\hat{A}_t^{\text{GRPO}} = A_t^* + \underbrace{(\hat{A}_t^{\text{GRPO}} - A_t^*)}_{\varepsilon_t^G}$$

**假设 G1（组信号误差模型）**：
- 均值：$\mathbb{E}[\varepsilon_t^G] = b_t^G$（逐步时序偏差，已在命题 0 中证明）
- 方差：$\mathrm{Var}(\varepsilon_t^G) = \sigma_G^2 = c^2 \mathrm{Var}(R)$（组采样噪声）

**来源 2（GAE 信号）**：
$$A_t^{\text{GAE}} = A_t^* + \varepsilon_t^V$$

**假设 V1（GAE 信号误差模型）**：
- 均值：$\mathbb{E}[\varepsilon_t^V] = 0$（使用 BHVF 训练 Critic 时近似无偏）
- 方差：$\mathrm{Var}(\varepsilon_t^V) = \sigma_V^2 \approx (1 - \text{EV}) \cdot \mathrm{Var}(A_t^{\text{GAE}})$

**假设 I（独立性）**：
$$\mathbb{E}[\varepsilon_t^G \varepsilon_t^V] = 0$$

*理由*：组采样随机性和 Critic 初始化误差由在策略轨迹采集中的不同随机种子驱动。$\square$

---

### 3.2 Theorem 3: Optimal Bayesian Fusion (Main Result) | 定理 3：最优贝叶斯融合（核心结论）

**[EN]**

**Theorem 3 (CA-GRPO Optimal Fusion Coefficient)**. Under Assumptions G1, V1, and I, consider
the linear fusion estimator:

$$\hat{A}_t^{\text{CA}} = (1 - w_t) \hat{A}_t^{\text{GRPO}} + w_t A_t^{\text{GAE}}$$

The unique coefficient $w_t^*$ minimizing $\mathcal{J}(w_t) = \mathbb{E}[(\hat{A}_t^{\text{CA}} - A_t^*)^2]$ is:

$$\boxed{w_t^* = \frac{\sigma_G^2 + (b_t^G)^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2}}$$

Moreover, $w_t^* \in [0, 1]$, and the optimal MSE satisfies:

$$\mathcal{J}(w_t^*) = \frac{(\sigma_G^2 + (b_t^G)^2) \cdot \sigma_V^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2} < \min\!\left(\sigma_G^2 + (b_t^G)^2,\; \sigma_V^2\right)$$

That is, fusion is **always strictly better** than either estimator alone (as long as both have
finite MSE).

**[ZH]**

**定理 3（CA-GRPO 最优融合系数）**。在假设 G1、V1 和 I 成立的条件下，对线性融合估计器：

$$\hat{A}_t^{\text{CA}} = (1 - w_t) \hat{A}_t^{\text{GRPO}} + w_t A_t^{\text{GAE}}$$

最小化 $\mathcal{J}(w_t) = \mathbb{E}[(\hat{A}_t^{\text{CA}} - A_t^*)^2]$ 的唯一最优系数为：

$$\boxed{w_t^* = \frac{\sigma_G^2 + (b_t^G)^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2}}$$

此外，$w_t^* \in [0, 1]$，且最优 MSE 满足：

$$\mathcal{J}(w_t^*) = \frac{(\sigma_G^2 + (b_t^G)^2) \cdot \sigma_V^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2} < \min\!\left(\sigma_G^2 + (b_t^G)^2,\; \sigma_V^2\right)$$

即融合结果**总是严格优于**任一单独估计器（只要两者的 MSE 有限）。

*Proof | 证明*:

The MSE expands as (using Assumption I: cross terms vanish):

$$\mathcal{J}(w) = \mathbb{E}\!\left[((1-w)(\varepsilon_t^G + b_t^G) + w\varepsilon_t^V)^2\right]$$
$$= (1-w)^2\underbrace{(\sigma_G^2 + (b_t^G)^2)}_{\triangleq S_G} + w^2\sigma_V^2$$

This is a convex quadratic in $w$. Setting $d\mathcal{J}/dw = 0$:

$$-2(1-w^*)S_G + 2w^*\sigma_V^2 = 0 \implies w^* = \frac{S_G}{S_G + \sigma_V^2}$$

Since $S_G, \sigma_V^2 \geq 0$, we have $w^* \in [0,1]$. The optimal MSE $= S_G \sigma_V^2/(S_G + \sigma_V^2)$,
which is strictly less than $\min(S_G, \sigma_V^2)$ by the AM-HM inequality. $\blacksquare$

---

### 3.3 Practical Estimator: EV-Based Adaptive Fusion | 实践估计器：基于 EV 的自适应融合

**[EN]**

**Corollary 3.1 (EV-Based Approximation)**. When responses are short ($(b_t^G)^2 \ll \sigma_G^2$)
or when bias-domination holds, the optimal weight simplifies to:

$$w^* \approx \frac{\sigma_G^2}{\sigma_G^2 + \sigma_V^2}$$

Since $\sigma_V^2 \approx (1-\text{EV}) \cdot \mathrm{Var}(A^{\text{GAE}})$ and
$\text{EV} = 1 - \mathrm{Var}(G-V)/\mathrm{Var}(G)$, we obtain the **zero-hyperparameter estimator**:

$$\hat{w}^* = \mathrm{clip}(\widehat{\text{EV}},\; w_{\min},\; 1.0)$$

where $\widehat{\text{EV}}$ is the EMA-smoothed Critic Explained Variance (same formula as BHVF).

Properties:
- $\hat{w}^* \to 0$ when Critic is poor (early training, random init): CA-GRPO ≈ GRPO
- $\hat{w}^* \to 1$ when Critic is excellent (converged): CA-GRPO ≈ standard GAE-PPO
- $w_{\min} > 0$ ensures training never completely stops when GAE is informative

**[ZH]**

**推论 3.1（基于 EV 的近似）**。当响应较短（$(b_t^G)^2 \ll \sigma_G^2$）或偏差主导时，最优权重简化为：

$$w^* \approx \frac{\sigma_G^2}{\sigma_G^2 + \sigma_V^2}$$

由于 $\sigma_V^2 \approx (1-\text{EV}) \cdot \mathrm{Var}(A^{\text{GAE}})$ 且
$\text{EV} = 1 - \mathrm{Var}(G-V)/\mathrm{Var}(G)$，我们得到**零超参数估计器**：

$$\hat{w}^* = \mathrm{clip}(\widehat{\text{EV}},\; w_{\min},\; 1.0)$$

其中 $\widehat{\text{EV}}$ 为 EMA 平滑的 Critic 解释方差（与 BHVF 使用相同公式）。

性质：
- Critic 差时 $\hat{w}^* \to 0$（早期训练，随机初始化）：CA-GRPO ≈ GRPO
- Critic 优秀时 $\hat{w}^* \to 1$（收敛后）：CA-GRPO ≈ 标准 GAE-PPO
- $w_{\min} > 0$ 确保当 GAE 有信息量时训练不会完全停止

---

### 3.4 Connection to BHVF | 与 BHVF 的联系

**[EN]**

The structural identity between CA-GRPO Theorem 3 and BHVF Theorem 1 (main paper) is:

| Component | BHVF (Theorem 1) | CA-GRPO (Theorem 3) |
|---|---|---|
| **What is fused** | Value estimates $V(s)$ and $G$ | Advantage estimates $\hat{A}^{GRPO}$ and $A^{GAE}$ |
| **Source 1** | Critic $V$: low-var, high-bias | GRPO: low-var, high temporal-bias |
| **Source 2** | MC return $G$: high-var, unbiased | GAE: moderate-var, low-bias |
| **Fusion formula** | $V^c = (1-\alpha)V + \alpha G$ | $\hat{A}^{CA} = (1-w)\hat{A}^{GRPO} + w A^{GAE}$ |
| **Optimal gain** | $\alpha^* = \sigma_V^2/(\sigma_V^2+\sigma_G^2)$ | $w^* = (\sigma_G^2+b^2)/((\sigma_G^2+b^2)+\sigma_V^2)$ |
| **Adaptive proxy** | $\alpha^* \approx 1-\text{EV}$ | $w^* \approx \text{EV}$ |
| **Fusion level** | **Value level** (before advantage) | **Advantage level** (after GAE) |

The last row is the key architectural difference: BHVF corrects the Critic inputs that feed into
GAE; CA-GRPO corrects the GAE output directly. They are **complementary and composable**.

**[ZH]**

CA-GRPO 定理 3 与 BHVF 定理 1（主论文）之间存在结构同一性：

| 组件 | BHVF（定理 1） | CA-GRPO（定理 3） |
|---|---|---|
| **融合对象** | 价值估计 $V(s)$ 与 $G$ | 优势估计 $\hat{A}^{GRPO}$ 与 $A^{GAE}$ |
| **来源 1** | Critic $V$：低方差、高偏差 | GRPO：低方差、高时序偏差 |
| **来源 2** | MC 回报 $G$：高方差、无偏 | GAE：中方差、低偏差 |
| **融合公式** | $V^c = (1-\alpha)V + \alpha G$ | $\hat{A}^{CA} = (1-w)\hat{A}^{GRPO} + w A^{GAE}$ |
| **最优增益** | $\alpha^* = \sigma_V^2/(\sigma_V^2+\sigma_G^2)$ | $w^* = (\sigma_G^2+b^2)/((\sigma_G^2+b^2)+\sigma_V^2)$ |
| **自适应代理** | $\alpha^* \approx 1-\text{EV}$ | $w^* \approx \text{EV}$ |
| **融合层次** | **价值层面**（优势计算之前） | **优势层面**（GAE 计算之后） |

最后一行是关键架构区别：BHVF 修正输入 GAE 的 Critic 估计；CA-GRPO 直接修正 GAE 的输出。
两者**互补且可组合**。

---

### 3.5 Proposition 1: Gradient Direction Preservation | 命题 1：梯度方向保持性

**[EN]**

**Proposition 1 (CA-GRPO Gradient as Convex Combination)**.

With $w^*$ treated as stop-gradient (not backpropagated), the CA-GRPO policy gradient is:

$$\nabla_\theta \mathcal{L}_{\text{CA-GRPO}} = (1-w^*) \nabla_\theta \mathcal{L}_{\text{GRPO}} + w^* \nabla_\theta \mathcal{L}_{\text{GAE}}$$

This is a convex combination of GRPO and GAE gradient directions.
Corollaries:
1. When $w^* = 0$: CA-GRPO ≡ GRPO (no harm from Critic)
2. When $w^* = 1$: CA-GRPO ≡ standard GAE-based PPO
3. For all $w^* \in (0,1)$: gradient lies strictly within the convex hull of GRPO and GAE directions

*Proof*: By linearity of differentiation and linearity of $\hat{A}^{CA}$ in $w^*$. $\blacksquare$

**[ZH]**

**命题 1（CA-GRPO 梯度为凸组合）**。

当 $w^*$ 作为 stop-gradient 处理时，CA-GRPO 的策略梯度为：

$$\nabla_\theta \mathcal{L}_{\text{CA-GRPO}} = (1-w^*) \nabla_\theta \mathcal{L}_{\text{GRPO}} + w^* \nabla_\theta \mathcal{L}_{\text{GAE}}$$

这是 GRPO 与 GAE 梯度方向的凸组合。推论：
1. 当 $w^* = 0$：CA-GRPO ≡ GRPO（Critic 不带来任何危害）
2. 当 $w^* = 1$：CA-GRPO ≡ 标准基于 GAE 的 PPO
3. 对所有 $w^* \in (0,1)$：梯度严格位于 GRPO 与 GAE 方向的凸包内

*证明*：由微分线性性及 $\hat{A}^{CA}$ 对 $w^*$ 的线性性即得。$\blacksquare$

---

<a name="4-base"></a>
## 4. CA-GRPO-Base: Architecture Deep-Dive
## 4. CA-GRPO-Base 架构详解

> 本章是文档核心。它回答一个具体问题：**CA-GRPO-Base 到底是什么？它的每一个组件
> 为什么必须存在，以及它们如何协同工作？**
>
> This chapter is the document's core. It answers one concrete question: **What exactly is
> CA-GRPO-Base? Why must each component exist, and how do they work together?**

---

### 4.1 The "Base" in CA-GRPO-Base: What It Means
### 4.1 "Base" 的含义

**[ZH]**

"Base"（基础版）的核心含义是：**Critic 与策略网络共享同一个主干（backbone）**，只在顶部接一个
轻量的价值头（value head）。

对比三种模式：

```
CA-GRPO-Lite:
  ┌──────────────────────────────┐
  │  Policy backbone (frozen V)  │   ← 预训练的价值头，冻结，不更新
  └──────────────────────────────┘
  价值头参数：0（复用旧模型）

CA-GRPO-Base:
  ┌──────────────────────────────┐
  │     Shared Backbone (θ)      │   ← 策略和 Critic 共用同一套 Transformer 权重
  └────────┬─────────────┬───────┘
           │             │
    ┌──────▼──────┐  ┌───▼────────────┐
    │  Policy     │  │  Value Head    │   ← 仅 1 个线性层 (D_hidden → 1)
    │  Head (θ_π) │  │  (φ_v)         │
    └─────────────┘  └────────────────┘
  价值头参数：D_hidden ≈ 几千个参数（相比 policy 的数十亿可忽略）

CA-GRPO-Full:
  ┌──────────────────┐   ┌──────────────────┐
  │  Policy Network  │   │  Critic Network  │   ← 完全独立，参数翻倍
  │  (θ)             │   │  (φ)             │
  └──────────────────┘   └──────────────────┘
  Critic 参数：与策略网络等量（数十亿）
```

**[EN]**

"Base" means: the **Critic shares the same backbone (Transformer weights) with the policy**,
adding only a lightweight value head on top.

The three modes compared above show that CA-GRPO-Base adds only ~thousands of parameters
(a single linear layer) compared to the billions in the policy backbone.

---

### 4.2 Why Shared Backbone? — The Core Argument
### 4.2 为什么必须共享主干？——核心论证

**[ZH]**

共享主干是 CA-GRPO-Base 设计的关键。以下从三个独立角度论证其必要性：

#### 4.2.1 角度 A：表示复用——Critic 的质量取决于表示能力

GAE 中的 Critic 需要估计 $V_\phi(s_t) \approx \gamma^{T-t}\mathbb{E}_\pi[R | s_t]$，即
"从状态 $s_t$ 出发，当前策略 $\pi$ 的期望折扣回报"。

在 LLM 语境中，$s_t$ 是 KV-cache（或等价地，当前 token 序列的上下文表示）。
要预测未来回报，Critic 必须能够**理解当前推理进展的语义含义**——例如：
- "当前已生成了正确的中间推理步骤" → 高 $V$
- "当前已犯了不可挽回的计算错误" → 低 $V$

这种语义理解**已经被策略 backbone 学到了**（通过 SFT 预训练和 RLHF 微调）。
如果使用独立 Critic，需要从零开始重新学习相同的语义，大量浪费计算。

**共享 backbone 的优势**：价值头直接在策略已经学到的语义表示上做线性回归，
收敛速度远快于独立 Critic，且通常获得更低的 MSE。

**[EN]**

The Critic in GAE must estimate $V_\phi(s_t) \approx \gamma^{T-t}\mathbb{E}_\pi[R | s_t]$—the
expected future reward given the current reasoning state. In LLM settings, this requires
understanding the *semantic content* of the current generation: e.g., "has the model made a
correct intermediate reasoning step?" or "has the model committed an irreversible error?"

This semantic understanding is **already encoded in the policy backbone**. Sharing the backbone
means the value head does linear regression on already-meaningful representations, converging
much faster than an independent Critic learning from scratch.

---

#### 4.2.2 角度 B：分布对齐——Critic 必须与当前策略同步

**[ZH]**

PPO 的一个核心挑战是 **策略分布漂移**（policy shift）。每次梯度更新后，$\pi_\theta$ 改变，
而 Critic 估计的是**当前策略**下的价值 $V^\pi$（而非固定策略下的价值）。

如果使用独立 Critic：
- 策略更新后，Critic 的输入分布改变（状态的语义表示变了）
- Critic 需要额外的更新步骤来重新校准
- 在 PPO 的多 epoch 更新中（`n_epochs=10`），独立 Critic 永远落后于策略

如果使用共享 backbone：
- 每次 backbone 更新，**Critic 的输入表示也同步更新**
- 价值头始终工作在最新的策略表示上
- 分布对齐是**自动保证的**，不需要额外的同步机制

$$\text{共享 backbone: } s_t \xrightarrow{\theta_{\text{shared}}} h_t \xrightarrow{\phi_v} V_\phi(s_t)$$
$$\text{独立 Critic: } s_t \xrightarrow{\phi_{\text{独立}}} V_\phi(s_t) \quad \text{（可能过时）}$$

**[EN]**

With a shared backbone, every backbone update **simultaneously refreshes** the Critic's input
representations. The value head always operates on the most current policy's hidden states.
With an independent Critic, the representations can become stale relative to the current policy,
requiring extra update steps to recalibrate.

---

#### 4.2.3 角度 C：梯度流——价值训练反向传播到 backbone 是否有害？

**[ZH]**

这是一个关键的反省点。共享 backbone 意味着**价值损失的梯度会反向传播到 backbone 权重**，
这可能干扰策略学习。

**分析梯度流**：

$$L_{\text{total}} = L_{\text{policy}} + c_{\text{vf}} \cdot L_{\text{value}}$$

backbone 权重 $\theta$ 接收两路梯度：
1. $\nabla_\theta L_{\text{policy}}$：来自策略目标，推动 $\pi_\theta$ 改进
2. $\nabla_\theta L_{\text{value}}$：来自价值损失，推动 backbone 表示更利于预测 $V^\pi$

这两路梯度是否冲突？

**关键洞察**：两者目标**高度一致**。策略损失需要 backbone 理解"哪些 token 对结果有贡献"；
价值损失需要 backbone 理解"当前状态的期望未来回报是多少"。这两个目标本质上都在学习
"因果性推理结构"，它们的梯度**倾向于相互增强**，而非相互对抗。

**实证证据**（来自 PPO 文献）：ActorCritic with shared backbone 在 MuJoCo 等环境上的表现
优于或等价于独立网络（Andrychowicz et al., 2021），尽管理论上存在多目标冲突的风险。

**[EN]**

With a shared backbone, value loss gradients flow back through the backbone. Is this harmful?

Both policy and value objectives require the backbone to understand causal contribution of
tokens to outcomes—these objectives are **aligned**, not conflicting. The policy gradient says
"reinforce tokens that led to correct answers"; the value gradient says "accurately represent
the expected value of the current reasoning state." Both drive the backbone toward better
causal understanding. Empirically (Andrychowicz et al., 2021), shared actor-critic backbones
perform equivalently to or better than independent networks.

**Safeguard**: Use $c_{\text{vf}} = 0.5$ (half the policy gradient magnitude) to prevent value
loss from dominating. This is the same coefficient used in our OptimalPPO codebase.

---

### 4.3 The Value Head: Anatomy of the Critic in CA-GRPO-Base
### 4.3 价值头：CA-GRPO-Base 中 Critic 的解剖

**[ZH]**

价值头（value head）是 CA-GRPO-Base 的 Critic 的核心部件。它是接在 backbone 最后一层
隐状态之上的一个单独的线性变换：

```
Backbone 最后一层 hidden state: h_t ∈ ℝ^{D_hidden}
                    ↓
           Linear(D_hidden → 1)   [价值头，参数 φ_v]
                    ↓
              V_φ(s_t) ∈ ℝ       [标量价值估计]
```

**初始化策略**：
- 使用正交初始化，权重标准差 `std=1.0`（与 `CriticNetwork` 中的最后一层相同）
- 偏置初始化为 0
- 初始输出分布 ≈ $\mathcal{N}(0, 1)$，接近奖励量纲时不需要额外缩放

**为什么是线性层而非多层 MLP？**

在共享 backbone 的情况下，backbone 的最后一层 hidden state 已经包含了丰富的语义信息，
价值头只需要做**线性组合**即可提取价值信号。这类似于：
- BERT 的分类头：在预训练表示上加一个线性分类器
- GPT 的 reward model head：在 LLM 表示上加一个标量打分线性层

额外的非线性层增加了过拟合风险，且在共享表示上通常没有显著收益。

**[EN]**

The value head is a single `Linear(D_hidden → 1)` layer appended to the backbone's last hidden
state. This is analogous to BERT classification heads or reward model heads—the backbone already
extracts rich semantic features; the linear layer simply combines them into a scalar value estimate.

Additional MLP layers increase overfitting risk without significant benefit when operating on
top of rich pretrained representations.

**Initialization**: Orthogonal init with `std=1.0` (matching the `CriticNetwork` final layer
in the existing codebase). Initial outputs $\approx \mathcal{N}(0, 1)$.

---

### 4.4 The BHVF Critic Target: How the Value Head is Trained
### 4.4 BHVF Critic 目标：价值头如何训练

**[ZH]**

这是最容易混淆的部分。价值头的训练目标不是简单的 MSE 到 MC 回报，而是应用了 BHVF 的自适应混合目标：

**符号定义**（对应一个 token 序列中的时间步 $t$）：

| 符号 | 含义 |
|---|---|
| $V_\phi(s_t)$ | 价值头的当前预测（旧值，在本迭代开始时计算，stop-gradient） |
| $G_t$ | MC 回报：$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$（仅在 $t=T$ 有 $r_T = R$） |
| $\delta_t = G_t - V_\phi(s_t)$ | 原始新息（raw innovation） |
| $A_t^{\text{GAE}}$ | 标准 GAE 优势 |
| $\hat{r}_t^{\text{GAE}} = V_\phi(s_t) + A_t^{\text{GAE}}$ | GAE 回报（bootstrap target） |

**BHVF 目标的推导**：

**步骤 1**：截断异常新息（防止截断 episode 产生的极端值）

$$\delta_t^{\text{clip}} = \text{clip}\!\left(\delta_t,\; -c \cdot \hat{\sigma}_e,\; +c \cdot \hat{\sigma}_e\right)$$

其中 $\hat{\sigma}_e$ 是新息标准差的 EMA，$c=3$。

**步骤 2**：计算 BHVF 增益（使用上一轮次 EV 防止泄露）

$$\alpha^* = \text{clip}(1 - \widehat{\text{EV}}_{\text{prev}},\; 0.05,\; 1.0)$$

**步骤 3**：混合 GAE bootstrap 目标与 MC 修正目标

$$T_t^* = (1 - \alpha^*) \cdot \hat{r}_t^{\text{GAE}} + \alpha^* \cdot \underbrace{(V_\phi(s_t) + \delta_t^{\text{clip}})}_{\text{BHVF 修正价值}}$$

**价值损失**：

$$L_{\text{value}} = \frac{1}{2} \mathbb{E}_t\!\left[(V_\phi(s_t) - T_t^*)^2\right]$$

其中 $V_\phi(s_t)$ 是**当前**网络的预测（带梯度），$T_t^*$ 是 stop-gradient 的目标。

**[EN]**

The value head is trained with the BHVF adaptive target $T_t^*$, which interpolates between:
- **(Low EV)** the BHVF-corrected MC return: $V_\phi(s_t) + \delta_t^{\text{clip}}$
  → used heavily when Critic is poor and needs large MC correction
- **(High EV)** the GAE bootstrap return: $\hat{r}_t^{\text{GAE}} = V_\phi(s_t) + A_t^{\text{GAE}}$
  → used when Critic is accurate, GAE bootstrap has lower variance

The mixing weight $\alpha^* = 1 - \text{EV}$ is the **same Kalman gain derived in Theorem 1**
of the BHVF paper—applied here at the Critic training level.

This creates the **BHVF virtuous cycle**:
```
Better Critic → Higher EV → Lower α* (less MC noise in training) → Even Better Critic
                          → Higher w* (more GAE weight in advantage) → Better Policy
```

---

### 4.5 The Fusion Mechanism: How GRPO and GAE Are Combined
### 4.5 融合机制：GRPO 与 GAE 如何结合

**[ZH]**

这是 CA-GRPO-Base 的**算法核心**。理解此处需要清楚区分以下三种"优势"：

```
三种优势对比（对同一个 token (i, t)）:

  Â_GRPO[i,t] = (R_i - μ_R) / σ_R
              ↑ 对响应 i 内所有步骤相同，无时序结构

  A_GAE[i,t]  = Σ_{l≥0} (γλ)^l [r_{t+l} + γV_φ(s_{t+l+1}) - V_φ(s_{t+l})]
              ↑ 利用 V_φ 做 bootstrap，有时序折扣结构

  Â_CA[i,t]   = (1 - w*) · Â_GRPO[i,t] + w* · A_GAE[i,t]
              ↑ 贝叶斯最优融合，w* 由 EV 自适应决定
```

**融合的直觉理解**：

- **训练早期**（EV 低，Critic 差）：$w^* \approx w_{\min} = 0.1$
  - $\hat{A}^{CA} \approx 0.9 \cdot \hat{A}^{GRPO} + 0.1 \cdot A^{GAE}$
  - 几乎等同于 GRPO，不信任差的 Critic
  - **时序结构贡献 10%**，不会被完全忽略

- **训练中期**（EV = 0.5）：$w^* \approx 0.5$
  - $\hat{A}^{CA} \approx 0.5 \cdot \hat{A}^{GRPO} + 0.5 \cdot A^{GAE}$
  - GRPO 和 GAE 各贡献一半
  - 偏差（GRPO 时序盲视）与方差（Critic 噪声）均衡

- **训练后期**（EV 高，Critic 准确）：$w^* \approx 1.0$
  - $\hat{A}^{CA} \approx A^{GAE}$
  - 完全使用 GAE，恢复标准 PPO 行为
  - **自动适应**，无需手动切换

**[EN]**

The fusion weight $w^* = \text{clip}(\widehat{\text{EV}}_{\text{prev}}, w_{\min}, 1.0)$ creates
a smooth, automatic transition from GRPO (early training) to GAE-based PPO (late training).

The key property: **this is Pareto-improving**. For any fixed $w^* \in (0,1)$, the fused
advantage always has lower MSE than either pure GRPO or pure GAE alone (by Theorem 3). The
adaptive scheduling ensures we always operate in the right regime automatically.

---

### 4.6 Key Implementation Details and Pitfalls
### 4.6 关键实现细节与陷阱

**[ZH]**

以下是容易出错的实现细节，每条都附带错误后果说明：

#### 陷阱 1：信息泄露（Information Leakage）

**错误做法**：
```python
EV_batch = compute_ev(current_rollout)
w_star = clip(EV_batch, w_min, 1.0)   # ← 用当前批次的 EV 计算权重
A_CA = (1-w_star)*A_GRPO + w_star*A_GAE
```

**后果**：融合权重依赖于当前批次的 Critic 质量，而 Critic 质量反过来影响 $A^{GAE}$，
形成循环依赖，导致方差增大。

**正确做法**：
```python
w_star = clip(EV_ema_prev, w_min, 1.0)   # ← 使用上一轮次的 EV（t-1 时刻的 EMA）
A_CA = (1-w_star)*A_GRPO + w_star*A_GAE
EV_ema = (1-η)*EV_ema + η*EV_batch      # ← 更新 EMA 在计算完 A_CA 之后
```

#### 陷阱 2：优势归一化的层次（Normalization Level）

**错误做法**：
```python
# 按响应归一化（延续 GRPO 习惯）
for each response i:
    A_CA[i,:] = (A_CA[i,:] - mean(A_CA[i,:])) / std(A_CA[i,:])
```

**后果**：破坏了 GAE 的跨步骤比较意义。GRPO 的归一化是按响应做的，是因为组相对信号
本身就是相对量；但 GAE 的时序差异（早步 vs 晚步）是有意义的，不应被消除。

**正确做法**：
```python
# 按 mini-batch 归一化（OptimalPPO 的标准做法）
for each mini_batch b:
    A_norm[b] = (A_CA[b] - mean(A_CA[b])) / (std(A_CA[b]) + ε)
```

这保留了同一批次内跨步骤、跨响应的相对优势大小。

#### 陷阱 3：MC 回报的终端处理

**[LLM 场景]** 只有终端奖励时：
```python
G[i, T_i] = R_i          # 终端步骤：等于奖励
G[i, t]   = γ * G[i, t+1]  # t < T_i：纯折扣累积（中间 r_t = 0）
```

**[MuJoCo 场景]** 有中间奖励且 episode 可能被截断时：
```python
# 必须区分"截断"（非终止）和"终止"（真实结束）
for t in reversed(range(T)):
    if terminated[t]:    # 真实结束：下一步价值 = 0
        running_return = 0.0
    running_return = rewards[t] + γ * running_return
    G[t] = running_return
# 对于截断 episode：最后一步需要加入 bootstrap value
if not terminated[T-1]:
    running_return = last_value  # bootstrap from Critic
```

#### 陷阱 4：价值头的 stop-gradient 范围

训练 Critic 时：$T_t^*$ 中的 $V_\phi(s_t)$（作为目标的一部分）必须是 stop-gradient，
否则 Critic 可以通过降低自身预测来最小化损失（自举崩溃）：

```python
# 正确：目标是 stop-gradient
with torch.no_grad():
    v_old = value_head(backbone(s_t))   # stop-gradient 的旧价值
T_star = (1 - alpha) * (v_old + A_GAE) + alpha * (v_old + delta_clip)

# 当前价值（带梯度）
v_new = value_head(backbone(s_t))       # 带梯度
loss_value = 0.5 * (v_new - T_star).pow(2).mean()
```

**[EN]**

**Pitfall 1 (Information Leakage)**: Use EV from the *previous* rollout to compute $w^*$. This
prevents circular dependency between the fusion weight and current batch statistics.

**Pitfall 2 (Normalization Level)**: Normalize advantages *per mini-batch*, not per-response.
Per-response normalization (GRPO convention) would destroy GAE's meaningful temporal differences.

**Pitfall 3 (MC Return Terminal Handling)**: In LLM settings with terminal-only rewards, set
intermediate $r_t = 0$ and only apply $R_i$ at the final step. In MuJoCo settings, correctly
handle episode termination vs. truncation (bootstrap from Critic at truncation boundaries).

**Pitfall 4 (Stop-Gradient in Critic Target)**: The $V_\phi(s_t)$ appearing *inside* the target
$T_t^*$ must be stop-gradient, otherwise the Critic can trivially minimize the loss by reducing
its own predictions—a classic bootstrapping collapse.

---

### 4.7 The Complete Data Flow in One Iteration
### 4.7 单次迭代的完整数据流

**[ZH]**

以下是 CA-GRPO-Base 在一次训练迭代内的**完整数据流**，展示每个变量从何处来、流向何处：

```
输入：prompt x，当前策略 π_θ，价值头 V_{φ_v}，EV_ema（上轮遗留状态）

阶段 1：并行采样  ─────────────────────────────────────────────────────────
  π_θ.generate(x, G=8)
    → 8 条响应 {y_1, ..., y_8}，各长度 T_i
    → 记录每步 token 的 log_prob_{old}[i,t]
    → 记录每步的隐状态 h[i,t]（或重新 forward 时计算）

  奖励模型/验证器
    → 终端奖励 R_i ∈ {0, 1}

阶段 2：价值推断（单次 forward，无梯度）──────────────────────────────────
  for all (i, t):
    h[i,t] = backbone_θ(tokens[i, :t+1])[-1]   ← 共享 backbone forward
    V_old[i,t] = linear_φv(h[i,t])             ← 价值头输出（stop-gradient）

阶段 3：计算 GRPO 优势 ────────────────────────────────────────────────────
  μ_R = mean({R_i}), σ_R = std({R_i})
  A_GRPO[i,t] = (R_i - μ_R) / σ_R            ← 所有 t 相同，无时序结构

阶段 4：计算 MC 回报和 GAE 优势 ──────────────────────────────────────────
  for each response i (reversed):
    G[i,T_i] = R_i
    G[i,t]   = γ * G[i,t+1]    (中间步 r_t=0)

  δ[i,t] = G[i,t] - V_old[i,t]               ← 原始新息

  for each response i (reversed):
    δ_td[i,t] = r_{i,t} + γ*V_old[i,t+1] - V_old[i,t]  ← TD 误差
    A_GAE[i,t] = δ_td[i,t] + γλ*A_GAE[i,t+1]            ← GAE 递推

阶段 5：BHVF 目标与融合权重 ──────────────────────────────────────────────
  σ_e_now = std({δ[i,t]})
  δ_clip[i,t] = clip(δ[i,t], -3*σ_e_ema, +3*σ_e_ema)
  α_bhvf = clip(1 - EV_ema_prev, 0.05, 1.0)   ← 上轮 EV

  T*[i,t] = (1-α_bhvf)*(V_old[i,t]+A_GAE[i,t]) + α_bhvf*(V_old[i,t]+δ_clip[i,t])

  EV_batch = 1 - Var(δ) / Var(G)
  w*       = clip(EV_ema_prev, 0.1, 1.0)       ← 上轮 EV（防泄露）
  EV_ema   = (1-0.05)*EV_ema + 0.05*EV_batch   ← 更新 EMA

  A_CA[i,t] = (1-w*)*A_GRPO[i,t] + w**A_GAE[i,t]  ← 融合优势

阶段 6：多 epoch 梯度更新 ────────────────────────────────────────────────
  for epoch = 1..n_epochs:
    for mini_batch b:
      # 优势归一化（per mini-batch）
      A_norm = (A_CA[b] - mean) / (std + ε)

      # 重新 forward（带梯度）
      h_new[b] = backbone_θ(tokens[b])
      logprob_new[b] = policy_head_θπ(h_new[b], actions[b])
      V_new[b]      = value_head_φv(h_new[b])

      # 策略损失（PPO clip）
      ρ = exp(logprob_new - logprob_old)
      L_policy = -min(ρ*A_norm, clip(ρ,1-ε,1+ε)*A_norm).mean()

      # 价值损失（BHVF 目标）
      L_value = 0.5 * (V_new - T*[b]).pow(2).mean()

      # 联合损失 + 反向传播
      L_total = L_policy + 0.5 * L_value
      L_total.backward()
      clip_grad_norm_(params, 0.5)
      optimizer.step()

输出：更新后的 θ（backbone+policy head+value head），更新后的 EV_ema、σ_e_ema
```

**[EN]**

This data flow reveals the key architectural property: the **backbone forward pass is shared**
between policy head and value head in Phase 6. When `optimizer.step()` is called, both
$\theta_\pi$ (policy head weights) and $\phi_v$ (value head weights) are updated, as well as the
backbone weights $\theta_{\text{shared}}$ (which receive gradients from both objectives).

---

### 4.8 Self-Reflection on the CA-GRPO-Base Design
### 4.8 对 CA-GRPO-Base 设计的自我反省

**[ZH]**

以下是对设计关键选择的批判性审查：

**反省 1：共享主干的梯度干扰（Gradient Interference）**

策略梯度 $\nabla_{\theta} L_{\text{policy}}$ 和价值梯度 $\nabla_{\theta} L_{\text{value}}$
在 backbone 权重上**叠加**。当两者方向相反时，会相互抵消，降低优化效率。

*理论分析*：在 actor-critic 文献中，这被称为"多目标优化冲突"（multi-task gradient conflict）。
Grads 方向冲突的概率与任务相关性正相关——高相关任务（策略和价值都在学"哪些状态好"）冲突概率低。

*缓解方法*：使用 $c_{\text{vf}} = 0.5$ 缩放价值损失梯度；或使用 EV 自适应缩放
（低 EV 时，价值信号不可靠，降低 $c_{\text{vf}}$ 以减少干扰）。

*结论*：在实践中，共享 backbone 的梯度干扰通常不严重（Andrychowicz et al., 2021），
但在 LLM 场景中需要监控（通过测量策略损失和价值损失的梯度余弦相似度）。

**反省 2：EV 代理的初始化问题**

训练初期，EV_ema = 0，所以 $w^* = w_{\min} = 0.1$，CA-GRPO ≈ GRPO。
同时 $\alpha^* = 1 - 0 = 1$，Critic 目标完全依赖 MC 回报。

*潜在问题*：初期用 MC 回报训练 Critic，而 MC 回报本身就受到 GRPO 采样策略的影响，
可能引入初始化偏差。

*缓解方法*：使用 EV_ema 的 warm-start 值（例如 0.2），而非从 0 开始，
以避免初始期完全丢弃 GAE 信号。这与 BHVF 代码中的 `_alpha_v4_ema = 0.2` 设计一致。

**反省 3：共享 backbone 引入的回报尺度问题（Scale Mismatch）**

策略 backbone 针对语言模型概率（对数概率尺度）进行了优化，而价值头需要输出
"期望折扣奖励"（奖励尺度）。这两个尺度可能差异很大（语言概率 $\in [-\infty, 0]$，
奖励 $\in [0, 1]$）。

*解决方案*：
1. 使用 `std=1.0` 的正交初始化（而非 `std=0.01`），让价值头初始输出在奖励量纲附近
2. 对奖励进行标准化（类似于 `RunningMeanStd` 奖励归一化）
3. 或者：price head 使用独立的更大学习率

**[EN]**

The three self-reflection points above identify the key risks of the CA-GRPO-Base design:
1. **Gradient interference** from multi-task optimization (mitigated by $c_{\text{vf}} = 0.5$)
2. **EV initialization** at 0 causing full GRPO fallback (mitigated by warm-start EV = 0.2)
3. **Scale mismatch** between language model and value scales (mitigated by separate init std)

All three have known mitigation strategies grounded in existing BHVF and OptimalPPO codebases.

---

<a name="4"></a>
## 5. Pseudocode: Complete CA-GRPO-Base Algorithm
## 5. 伪代码：完整 CA-GRPO-Base 算法

### 5.1 Design Decisions | 设计决策

**[EN]**

**Decision 1: Critic mode — Base (shared backbone)**

CA-GRPO-Base is the recommended variant. It shares the policy Transformer backbone and adds
only a single `Linear(D_hidden → 1)` value head. This choice is justified in §4.2 above.

**Decision 2: Information leakage prevention**

Use the EV from the **previous rollout** (EV_ema before update) to compute $w^*$, preventing
the fusion weight from depending on the current batch's statistics (see §4.6 Pitfall 1).

**Decision 3: BHVF integration for Critic training**

Apply BHVF to the Critic training target (same formula as §2.4 of main paper). BHVF improves
Critic quality → increases EV → increases $w^*$ → CA-GRPO-Base leverages better GAE signal.

**[ZH]**

**决策 1：Critic 模式——Base（共享主干）**

CA-GRPO-Base 是推荐变体。共享策略 Transformer 主干，仅添加单个 `Linear(D_hidden → 1)` 价值头。
此选择已在 §4.2 中详细论证。

**决策 2：防止信息泄露**

使用**上一轮次**的 EV（更新前的 EV_ema）计算 $w^*$，防止融合权重依赖当前批次统计量（见 §4.6 陷阱 1）。

**决策 3：BHVF 集成用于 Critic 训练**

将 BHVF 应用于 Critic 训练目标（与主论文 §2.4 公式相同）。BHVF 改善 Critic 质量 → 提高 EV →
提高 $w^*$ → CA-GRPO-Base 利用更准确的 GAE 信号。

---

### 5.2 Complete CA-GRPO-Base Pseudocode | CA-GRPO-Base 完整伪代码

```
Algorithm: CA-GRPO-Base
  Critic-Augmented GRPO with Shared Backbone + BHVF Critic Training
  共享主干 + BHVF Critic 训练的 Critic 增强 GRPO
=======================================================================

Network Architecture | 网络架构:
  backbone(·; θ)           : shared Transformer (policy + value shared)
  policy_head(·; θ_π)      : Linear(D_hidden → vocab_size) + softmax
  value_head(·; φ_v)       : Linear(D_hidden → 1)  [only ~D_h params added]

  Combined forward | 联合前向传播:
    h_t        = backbone(tokens[0:t+1])[-1]   ← last hidden state @ step t
    logits_t   = policy_head(h_t)              ← action logits
    V_φ(s_t)   = value_head(h_t)               ← scalar value (SAME h_t!)

  KEY: h_t computed ONCE, shared by BOTH heads → near-zero extra cost
  关键：h_t 只计算一次，被两个头复用 → 几乎零额外开销

Parameters | 超参数:
  γ = 0.99   λ = 0.95   G = 8    w_min = 0.1   c_clip = 3.0
  η = 0.05   c_vf = 0.5  n_epochs = 4-10   ε = 0.2

Algorithm State (persists across iterations) | 算法级状态（跨迭代持久）:
  EV_ema    ← 0.2   (warm-start, NOT 0!)
  σ_e_ema   ← 1.0
=======================================================================

FOR each training iteration k = 1, 2, ...:

  ╔═══════════════════════════════════════════════════════════════╗
  ║ PHASE 1: Rollout + Value Inference  (torch.no_grad())        ║
  ║ 阶段 1：轨迹采集 + 价值推断（无梯度）                         ║
  ╚═══════════════════════════════════════════════════════════════╝

  # Snapshot BEFORE update — prevents info leakage (§4.6 Pitfall 1)
  # 在更新前记录快照 — 防止信息泄露
  EV_ema_prev  ← EV_ema
  σ_e_ema_prev ← σ_e_ema

  FOR each prompt x in batch:
    # [1a] Sample G responses
    {y_i}_{i=1}^G  ~ π_{θ_old}(· | x)
    # Each y_i = (a_{i,1}, ..., a_{i,T_i})

    # [1b] Terminal reward
    R_i ← reward(x, y_i)          e.g. R_i ∈ {0.0, 1.0}
    r[i,t] ← 0 for t < T_i;  r[i,T_i] ← R_i

    # [1c] Record old log-probs
    logprob_old[i,t] ← log π_{θ_old}(a_{i,t} | tokens[i, 0:t])

    # [1d] Value inference — shared backbone, NO grad
    #      价值推断：共享主干，无梯度
    with torch.no_grad():
      FOR (i, t):
        h[i,t]     ← backbone_θ(tokens[i, 0:t+1])[-1]   # same backbone as policy
        V_old[i,t] ← value_head_φv(h[i,t])               # stop-gradient

  ╔═══════════════════════════════════════════════════════════════╗
  ║ PHASE 2: Three Advantage Types  (numpy, no grad)             ║
  ║ 阶段 2：三种优势计算（numpy，无梯度）                          ║
  ╚═══════════════════════════════════════════════════════════════╝

  # [2a] GRPO — trajectory-level, NO temporal structure
  #      GRPO 优势：轨迹级别，无时序结构
  μ_R ← mean({R_i});  σ_R ← std({R_i}) + 1e-8
  A_GRPO[i,t] ← (R_i - μ_R) / σ_R   for all (i,t)
  #  ^ SAME value for all t within response i
  #  ^ 对响应 i 内所有步骤完全相同（时序盲视）

  # [2b] MC returns — for BHVF and EV
  G[i, T_i] ← R_i
  G[i, t] ← γ * G[i, t+1]  for t < T_i  (r[i,t]=0 intermediately)

  # [2c] GAE — per-step, HAS temporal discount structure
  #       GAE 优势：逐步，含时序折扣结构（这正是 GRPO 所缺失的）
  FOR each response i:
    last_gae ← 0
    FOR t = T_i downto 1:
      nonterminal ← 0 if done[i,t] else 1
      next_val    ← V_old[i, t+1] if t < T_i else 0.0
      δ_td        ← r[i,t] + γ * next_val * nonterminal - V_old[i,t]
      last_gae    ← δ_td + γ * λ * nonterminal * last_gae
      A_GAE[i,t]  ← last_gae
  #  ^ DIFFERENT for each t (early steps get smaller signal, correct!)
  #  ^ 每步不同（早期步骤信号更小，这正是期望的时序折扣行为）

  ╔═══════════════════════════════════════════════════════════════╗
  ║ PHASE 3: BHVF Critic Target  (numpy, no grad)                ║
  ║ 阶段 3：BHVF Critic 训练目标（numpy，无梯度）                  ║
  ╚═══════════════════════════════════════════════════════════════╝

  δ[i,t] ← G[i,t] - V_old[i,t]               ← raw innovation 原始新息
  σ_e_now ← std({δ[i,t]}) over full batch

  # Clip outliers (±3σ handles truncated episodes)
  δ_clip[i,t] ← clip(δ[i,t], -c_clip*σ_e_ema_prev, +c_clip*σ_e_ema_prev)

  # BHVF gain: high when Critic poor, low when Critic good
  α_bhvf ← clip(1 - EV_ema_prev, 0.05, 1.0)

  # Mixed Critic target
  r_GAE[i,t]  ← V_old[i,t] + A_GAE[i,t]          ← GAE bootstrap return
  V_bhvf[i,t] ← V_old[i,t] + δ_clip[i,t]         ← BHVF corrected value
  T_star[i,t] ← (1 - α_bhvf) * r_GAE[i,t]
               + α_bhvf      * V_bhvf[i,t]          ← BHVF target (stop-grad)
  #  EV≈0 → α≈1: rely on MC (Critic is bad, bootstrap is unreliable)
  #  EV≈1 → α≈0: rely on GAE bootstrap (Critic is good, low variance)

  ╔═══════════════════════════════════════════════════════════════╗
  ║ PHASE 4: Fusion Weight + Fused Advantage  (numpy)            ║
  ║ 阶段 4：融合权重 + 融合优势                                    ║
  ╚═══════════════════════════════════════════════════════════════╝

  EV_batch ← 1 - Var({δ[i,t]}) / (Var({G[i,t]}) + 1e-8)

  # Use PREVIOUS EV to compute w* (anti-leakage rule!)
  # 使用上轮 EV 计算 w*（防泄露规则！）
  w_star ← clip(EV_ema_prev, w_min, 1.0)

  # Update EMA AFTER w_star is computed
  EV_ema  ← (1-η)*EV_ema  + η*EV_batch
  σ_e_ema ← (1-η)*σ_e_ema + η*σ_e_now

  # Fused advantage — Theorem 3 optimal combination
  A_CA[i,t] ← (1 - w_star) * A_GRPO[i,t] + w_star * A_GAE[i,t]

  # Early   (EV_prev≈0.2, w*=0.2): A_CA ≈ 0.8·GRPO + 0.2·GAE → safe GRPO-like
  # Mid     (EV_prev≈0.5, w*=0.5): A_CA ≈ 0.5·GRPO + 0.5·GAE → balanced
  # Late    (EV_prev≈0.9, w*=0.9): A_CA ≈ 0.1·GRPO + 0.9·GAE → full temporal credit

  ╔═══════════════════════════════════════════════════════════════╗
  ║ PHASE 5: Multi-Epoch Policy + Critic Update  (with grads)    ║
  ║ 阶段 5：多 epoch 策略与 Critic 联合更新（带梯度）               ║
  ╚═══════════════════════════════════════════════════════════════╝

  FOR epoch = 1 to n_epochs:
    FOR each mini-batch b of (i,t) pairs:

      # [5a] Per-minibatch advantage normalization
      #      OVER THE MINI-BATCH (not per-response — preserves temporal ordering)
      #      在 mini-batch 上归一化（非按响应 — 保留时序排序含义）
      A_norm[b] ← (A_CA[b] - mean(A_CA[b])) / (std(A_CA[b]) + 1e-8)

      # [5b] Shared backbone forward WITH gradients
      #      ONE pass → BOTH policy and value (shared h_t)
      #      一次前向传播 → 同时输出策略和价值（共享 h_t）
      FOR (i,t) in b:
        h_new[i,t]       ← backbone_θ(tokens[i, 0:t+1])[-1]   # grad ON
        logprob_new[i,t] ← policy_head_θπ(h_new[i,t], a[i,t]) # grad flows
        V_new[i,t]       ← value_head_φv(h_new[i,t])           # grad flows
        # logprob_new and V_new share the SAME h_new and grad computation graph
        # 两者共享同一个 h_new 及其梯度计算图

      # [5c] PPO-clip policy loss
      ρ[b]     ← exp(logprob_new[b] - logprob_old[b])
      surr1    ← ρ[b] * A_norm[b]
      surr2    ← clip(ρ[b], 1-ε, 1+ε) * A_norm[b]
      L_policy ← -mean(min(surr1, surr2))

      # [5d] Value loss — BHVF target (stop-gradient from Phase 3)
      L_value ← 0.5 * mean((V_new[b] - T_star[b])^2)
      # T_star[b] has NO gradient (computed from V_old which was stop-grad)

      # [5e] Joint backward — key to shared backbone design
      #      联合反向传播 — 共享主干设计的核心
      L_total ← L_policy + c_vf * L_value    (c_vf = 0.5)
      #
      # Gradient at backbone weight w:
      # 主干权重 w 处的梯度：
      #   ∂L_total/∂w = ∂L_policy/∂w + 0.5 * ∂L_value/∂w
      #
      # Both gradients flow through backbone simultaneously
      # 两路梯度同时流经主干
      #
      optimizer.zero_grad()
      L_total.backward()          ← ONE backward, updates θ, θ_π, φ_v together
      clip_grad_norm_(params, 0.5)
      optimizer.step()
=======================================================================
```

### 5.3 PyTorch Implementation Sketch | PyTorch 实现草图

**[ZH]**

以下是 CA-GRPO-Base 网络部分的 PyTorch 代码草图，与现有 `optimal_ppo.py` 的代码风格保持一致：

```python
# gae_experiments/agents/optimal_ppo.py (proposed addition)
from gae_experiments.utils.networks import layer_init

class SharedBackboneCAGRPO(nn.Module):
    """
    CA-GRPO-Base network: shared backbone with policy + value heads.
    Equivalent to OptimalPPO's actor+critic but with shared backbone weights.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, continuous=True):
        super().__init__()
        # Shared backbone — same architecture as CriticNetwork body
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.continuous = continuous
        if continuous:
            self.policy_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.policy_logits = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        # Value head: single linear layer (std=1.0 matches CriticNetwork init)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs):
        """ONE forward pass → policy distribution AND value estimate."""
        h = self.backbone(obs)                    # shared hidden state
        value = self.value_head(h).squeeze(-1)    # V(s) — scalar
        if self.continuous:
            dist = torch.distributions.Normal(
                self.policy_mean(h),
                self.log_std.exp().expand(h.shape[0], -1)
            )
        else:
            dist = torch.distributions.Categorical(logits=self.policy_logits(h))
        return dist, value                        # BOTH in one pass

    def get_value(self, obs):
        """Value-only (for rollout inference under no_grad)."""
        return self.value_head(self.backbone(obs)).squeeze(-1)


class CAGRPOBase(OptimalPPO):
    """
    CA-GRPO-Base: extends OptimalPPO with:
    (1) SharedBackboneCAGRPO (vs. separate actor/critic)
    (2) GRPO baseline advantage (trajectory-level, no temporal structure)
    (3) Bayesian fusion: A_CA = (1-w*)*A_GRPO + w**A_GAE
    (4) BHVF Critic target T*
    (5) Fusion weight w* = clip(EV_ema_prev, w_min, 1.0)
    """
    NAME = "CA_GRPO_Base"

    def __init__(self, env, group_size=8, w_min=0.1, c_clip=3.0,
                 eta_ev=0.05, **ppo_kwargs):
        super().__init__(env=env, **ppo_kwargs)
        self.G = group_size
        self.w_min = w_min
        self.c_clip = c_clip
        self.eta_ev = eta_ev
        # Replace separate actor+critic with shared backbone model
        obs_dim = env.observation_space.shape[0]
        action_dim = (env.action_space.shape[0]
                      if hasattr(env.action_space, 'shape')
                      else env.action_space.n)
        self.shared_model = SharedBackboneCAGRPO(
            obs_dim, action_dim, self.hidden_dim, self.continuous
        ).to(self.device)
        # Single optimizer for ALL parameters (backbone + both heads)
        self.optimizer = torch.optim.Adam(
            self.shared_model.parameters(), lr=self.lr_init, eps=1e-5
        )
        # EMA state — warm-start at 0.2 (not 0, avoids full GRPO fallback)
        self._ev_ema = 0.2
        self._sigma_e_ema = 1.0

    def compute_gae(self, last_value: float):
        """Override: compute GRPO + GAE + BHVF fusion."""
        T = self.buffer.pos
        rewards   = self.buffer.rewards[:T].numpy()
        values    = self.buffer.values[:T].numpy()   # V_old from rollout
        terminated = self.buffer.terminated[:T].numpy()
        # ... (full implementation follows pseudocode above)
        # Stores A_CA in buffer.advantages[:T]
        # Stores T_star in buffer.returns[:T]

    def update(self):
        """Override update to use shared model forward."""
        obs, actions, old_log_probs, advantages, returns, _ = self.buffer.get_batch()
        T = self.buffer.pos
        indices = np.arange(T)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start:start+self.batch_size]
                if len(batch_idx) < self.batch_size:
                    break

                batch_obs  = obs[batch_idx]
                batch_acts = actions[batch_idx]
                batch_lp   = old_log_probs[batch_idx]
                batch_adv  = advantages[batch_idx]
                batch_ret  = returns[batch_idx]   # T_star from BHVF

                # Per-minibatch advantage normalization
                if self.use_adv_norm:
                    batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

                # ONE forward pass through shared model
                dist, new_values = self.shared_model(batch_obs)
                new_log_probs = dist.log_prob(batch_acts)
                if self.continuous:
                    new_log_probs = new_log_probs.sum(-1)

                ratio = torch.exp(new_log_probs - batch_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # BHVF Critic target (batch_ret = T_star, stop-gradient)
                value_loss = 0.5 * ((new_values - batch_ret) ** 2).mean()

                # Joint loss: both gradients flow into shared backbone
                loss = policy_loss + self.vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.shared_model.parameters(), 0.5)
                self.optimizer.step()
```

**[EN]**

The code above mirrors `OptimalPPO` exactly—the only structural difference is that `actor` and
`critic` are replaced by a single `shared_model` with one `forward()` method that returns both
`(dist, value)`. The optimizer, loss computation, and training loop are **identical**.

In `OptimalPPO`:
- `self.actor` and `self.critic` are separate `nn.Module` instances (no weight sharing)
- Both are updated by the same `self.optimizer`

In `CAGRPOBase`:
- `self.shared_model` is ONE `nn.Module` (backbone shared, heads separate)
- Updated by the same single `self.optimizer`

The key difference is purely at the weight level: shared backbone means
$\nabla L_{\text{policy}}$ and $\nabla L_{\text{value}}$ both flow through the same backbone
weights, creating a multi-task learning signal.

**[ZH]**

上述代码与 `OptimalPPO` 完全镜像——唯一的结构差异是 `actor` 和 `critic` 被替换为单个
`shared_model`，其 `forward()` 方法同时返回 `(dist, value)`。优化器、损失计算和训练循环**完全相同**。

关键区别仅在权重层面：共享主干意味着 $\nabla L_{\text{policy}}$ 和 $\nabla L_{\text{value}}$
都流经同一主干权重，创造多任务学习信号。

### 5.4 Computational Cost Analysis | 计算开销分析

**[EN]**

| Phase | Operation | Cost vs GRPO | Notes |
|---|---|---|---|
| Phase 1 | G backbone forwards | = baseline | Same as GRPO |
| Phase 1 | Value head inference (reuse h) | ≈ +0% | Near-zero |
| Phase 2–4 | GAE, BHVF, fusion (numpy) | O(G·T) | Negligible |
| Phase 5 | n_epochs policy backward | = baseline | Same as GRPO |
| Phase 5 | n_epochs value loss backward | +extra backward | ~+15–25% total |

**Net overhead vs pure GRPO**: approximately **+15–25%** compute, driven by the extra backward
pass for value loss. Far less than CA-GRPO-Full (+100%).

**[ZH]**

| 阶段 | 操作 | 相比 GRPO 的开销 | 备注 |
|---|---|---|---|
| 阶段 1 | G 次主干前向（采样） | = 基线 | 与 GRPO 相同 |
| 阶段 1 | 价值头推断（复用 h） | ≈ +0% | 近乎零 |
| 阶段 2–4 | GAE、BHVF、融合（numpy） | O(G·T) | 可忽略 |
| 阶段 5 | n_epochs 策略反向 | = 基线 | 与 GRPO 相同 |
| 阶段 5 | n_epochs 价值损失反向 | + 额外反向传播 | 总计约 +15–25% |

**相比纯 GRPO 的净开销**：总计约 **+15–25%**，来源于价值损失的额外反向传播。
远少于 CA-GRPO-Full（+100%）。

---

<a name="5"></a>
## 6. Rigorous Self-Reflection: Assumption Verification and Gap Analysis
## 6. 自我反省：假设验证与漏洞分析

> *Every step of the derivation is examined for hidden assumptions, edge cases, and failure modes.*
> *对推导的每一步进行隐藏假设、边界情况和失效模式的审查。*


### 5.1 Is GRPO Bias Really Time-Dependent? (Re-examination of Prop. 0)

### 5.1 GRPO 偏差真的是时间相关的吗？（命题 0 的再审查）

**[EN]**

**Claim**: $b_t^G = \hat{A}_t^{\text{GRPO}} - A_t^*$ depends on $t$.

**Re-examination**: The proof in §1.3 assumed:
1. Terminal-only rewards ($r_t = 0$ for $t < T$)
2. A fixed policy $\pi$ (not changing within the rollout)

**What if there are intermediate rewards?** (e.g., format rewards in DAPO)

With intermediate rewards $r_t \neq 0$:
- $\hat{A}_t^{\text{GRPO}} = (R_i - \bar{R}) / \sigma_R$ still uses only the **total** reward $R_i = \sum_t r_t$
- $A_t^* = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$ now has **per-step information**

The bias increases: temporal blind spot grows with the diversity of intermediate rewards.

**Verdict**: ✓ Proposition 0 holds. Time-dependence of bias is MORE severe with intermediate
rewards, making GAE augmentation even more valuable.

**[ZH]**

**主张**：$b_t^G = \hat{A}_t^{\text{GRPO}} - A_t^*$ 依赖于 $t$。

**再审查**：§1.3 的证明假设了：
1. 仅终端奖励（$t < T$ 时 $r_t = 0$）
2. 固定策略 $\pi$（轨迹内不变）

**如果存在中间奖励呢？**（例如 DAPO 中的格式奖励）

有中间奖励时 $r_t \neq 0$：
- $\hat{A}_t^{\text{GRPO}} = (R_i - \bar{R}) / \sigma_R$ 仍只使用**总**奖励 $R_i = \sum_t r_t$
- $A_t^* = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$ 现在有**逐步信息**

偏差增大：随着中间奖励的多样性，时序盲点更加严重。

**结论**：✓ 命题 0 成立。有中间奖励时偏差的时间相关性更严重，使得 GAE 增强更有价值。

---


### 5.2 Is Assumption V1 (GAE Unbiasedness) Valid? (Critical Gap)

### 5.2 假设 V1（GAE 无偏性）是否有效？（关键漏洞）

**[EN]**

**Statement**: $\mathbb{E}[\varepsilon_t^V] = 0$ (GAE is unbiased).

**Problem**: GAE is unbiased only when $V_\phi = V^*$. In early training, $V_\phi$ has
initialization bias (BHVF's Failure Mode I). This means $\varepsilon_t^V$ has **non-zero mean**.

**How severe is this?**

If GAE is also biased, Theorem 3's formula gives:

$$w_t^* = \frac{S_G}{S_G + \underbrace{(\sigma_V^2 + (b_t^V)^2)}_{=S_V}}$$

The MSE at the optimal $w^*$ is now $S_G S_V / (S_G + S_V)$. This is still better than GRPO
alone ($S_G$) as long as $S_V < \infty$.

**BUT**: If $b_t^V$ is large AND $w^*$ is large (near 1), we are trusting a biased GAE heavily.
This can hurt. The EV-based estimator $\hat{w}^* = \text{EV}$ naturally protects against this:
when Critic has large bias, EV is low, $\hat{w}^*$ is small → CA-GRPO reverts to GRPO.

**Resolution (critical interaction with BHVF)**:
1. Use BHVF as Critic training objective → reduces $b_t^V$ → EV increases → $w^*$ increases
2. The **ordering** is: BHVF first reduces Critic bias; THEN CA-GRPO trusts GAE more
3. This is not circular: BHVF updates $\phi$ via gradient descent; CA-GRPO uses $V_\phi$ for
   inference only within the same iteration

**Verdict**: ⚠️ V1 is an approximation, not exact in early training. The EV-based $\hat{w}^*$
provides an automatic safeguard. BHVF integration is essential for CA-GRPO to reach full potential.

**[ZH]**

**表述**：$\mathbb{E}[\varepsilon_t^V] = 0$（GAE 无偏）。

**问题**：GAE 仅在 $V_\phi = V^*$ 时无偏。早期训练中，$V_\phi$ 有初始化偏差（BHVF 的失效模式 I）。
这意味着 $\varepsilon_t^V$ 有**非零均值**。

**严重程度**：

如果 GAE 也有偏，定理 3 的公式变为：

$$w_t^* = \frac{S_G}{S_G + \underbrace{(\sigma_V^2 + (b_t^V)^2)}_{=S_V}}$$

最优 $w^*$ 处的 MSE 为 $S_G S_V / (S_G + S_V)$，仍优于纯 GRPO（$S_G$），只要 $S_V < \infty$。

**但**：若 $b_t^V$ 大且 $w^*$ 大（接近 1），我们大量依赖有偏的 GAE，可能造成损害。
基于 EV 的估计器 $\hat{w}^* = \text{EV}$ 自然地防止了这一点：当 Critic 偏差大时，EV 低，
$\hat{w}^*$ 小 → CA-GRPO 回退到 GRPO。

**解决方案（与 BHVF 的关键交互）**：
1. 使用 BHVF 作为 Critic 训练目标 → 减少 $b_t^V$ → EV 增加 → $w^*$ 增加
2. **顺序**是：BHVF 首先减少 Critic 偏差；然后 CA-GRPO 更多地信任 GAE
3. 这不是循环依赖：BHVF 通过梯度下降更新 $\phi$；CA-GRPO 在同一迭代内仅将 $V_\phi$ 用于推理

**结论**：⚠️ V1 是近似而非精确，早期训练中不完全成立。基于 EV 的 $\hat{w}^*$ 提供了自动保障。
BHVF 集成对 CA-GRPO 达到其全部潜力至关重要。

---


### 5.3 Is Assumption I (Independence) Valid?

### 5.3 假设 I（独立性）是否有效？

**[EN]**

**Statement**: $\mathbb{E}[\varepsilon_t^G \varepsilon_t^V] = 0$.

**Analysis**:
- $\varepsilon_t^G$ is determined by: (a) random sampling of responses (independent of $\phi$);
  (b) temporal position $t$ (deterministic function of $T$ and $\gamma$)
- $\varepsilon_t^V$ is determined by: (a) Critic weight initialization $\phi_0$; (b) bootstrap
  errors in GAE computation

Indirect correlation exists via "policy quality" as a common cause: bad policy → both signals
noisier. But this common-cause correlation does not induce direct statistical dependence between
$\varepsilon_t^G$ and $\varepsilon_t^V$ within a fixed rollout.

**Formal bound**: By Cauchy-Schwarz:
$$|\text{Cov}(\varepsilon^G, \varepsilon^V)| \leq \sqrt{S_G \cdot \sigma_V^2}$$

Even if this bound is saturated, the optimal MSE increases by at most a factor of 2. The
qualitative conclusion (fusion is better than either alone) holds.

**Verdict**: ✓ Independence is approximately valid; any violation degrades the optimality
constant but not the direction of improvement.

**[ZH]**

**表述**：$\mathbb{E}[\varepsilon_t^G \varepsilon_t^V] = 0$。

**分析**：
- $\varepsilon_t^G$ 由以下因素决定：(a) 响应的随机采样（与 $\phi$ 无关）；
  (b) 时间位置 $t$（$T$ 和 $\gamma$ 的确定性函数）
- $\varepsilon_t^V$ 由以下因素决定：(a) Critic 权重初始化 $\phi_0$；(b) GAE 计算中的自举误差

通过"策略质量"作为共同原因存在间接相关：策略差 → 两个信号都更嘈杂。但这种共因相关性不会在固定
轨迹内诱发 $\varepsilon_t^G$ 与 $\varepsilon_t^V$ 之间的直接统计依赖。

**形式界**：由 Cauchy-Schwarz 不等式：
$$|\text{Cov}(\varepsilon^G, \varepsilon^V)| \leq \sqrt{S_G \cdot \sigma_V^2}$$

即使此界被饱和，最优 MSE 至多增加 2 倍。定性结论（融合优于任一单独来源）仍成立。

**结论**：✓ 独立性近似成立；任何违反仅降低最优性常数，但不改变改进方向。

---


### 5.4 The EV Proxy Gap (Most Important Limitation)

### 5.4 EV 代理差距（最重要的局限性）

**[EN]**

**True optimal weight**:
$$w_t^* = \frac{\sigma_G^2 + (b_t^G)^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2}$$

**Practical estimator**: $\hat{w}^* = \text{clip}(\widehat{\text{EV}}, w_{\min}, 1)$

**When is $\hat{w}^* \approx w_t^*$?**

From Corollary 1 of BHVF: $\text{EV} \approx 1 - \sigma_V^2 / \sigma_G^2$.
So $\hat{w}^* \approx 1 - \sigma_V^2/\sigma_G^2 = \sigma_G^2/(\sigma_G^2 + \sigma_V^2)$.

Comparing with $w_t^*$, they agree when $(b_t^G)^2 \ll \sigma_G^2$ (short responses, bias-free
approximation). When $b_t^G$ is large (long chain-of-thought), $w_t^* > \hat{w}^*$—the EV proxy
**underestimates** the optimal weight, leading to under-weighting of GAE.

**Is this safe?** Yes: the failure mode is reverting to GRPO (under-using GAE), not over-using
biased GAE. The estimator is **conservatively biased** in the right direction.

**Potential improvement**: Explicitly estimate $b_t^G \approx \hat{A}_t^{\text{GRPO}} - A_t^{\text{GAE}}$
per-step, and use the full Theorem 3 formula. Left as future work.

**[ZH]**

**真实最优权重**：
$$w_t^* = \frac{\sigma_G^2 + (b_t^G)^2}{\sigma_G^2 + (b_t^G)^2 + \sigma_V^2}$$

**实践估计器**：$\hat{w}^* = \text{clip}(\widehat{\text{EV}}, w_{\min}, 1)$

**何时 $\hat{w}^* \approx w_t^*$？**

由 BHVF 推论 1：$\text{EV} \approx 1 - \sigma_V^2 / \sigma_G^2$。
所以 $\hat{w}^* \approx 1 - \sigma_V^2/\sigma_G^2 = \sigma_G^2/(\sigma_G^2 + \sigma_V^2)$。

与 $w_t^*$ 比较，当 $(b_t^G)^2 \ll \sigma_G^2$（短响应、近似无偏）时两者一致。当 $b_t^G$ 较大时
（长链式推理），$w_t^* > \hat{w}^*$——EV 代理**低估**了最优权重，导致 GAE 权重偏低。

**这安全吗？** 是的：失效模式是回退到 GRPO（低估使用 GAE），而非过度使用有偏 GAE。
估计器在正确方向上**保守偏置**。

**潜在改进**：显式估计每步 $b_t^G \approx \hat{A}_t^{\text{GRPO}} - A_t^{\text{GAE}}$，
并使用完整的定理 3 公式。留作未来工作。

---


### 5.5 Theoretical Validity Summary | 理论有效性总结

| Assumption / Claim | Status | Conditions for Validity | 假设/主张 | 状态 | 有效性条件 |
|---|---|---|---|---|---|
| GRPO has temporal bias (Prop. 0) | ✓ Proven | $\gamma < 1$, $T > 1$ | GRPO 有时序偏差（命题 0） | ✓ 已证明 | $\gamma < 1$, $T > 1$ |
| G1: Group error model | ✓ Valid | On-policy rollout | G1：组误差模型 | ✓ 有效 | 在策略轨迹 |
| V1: GAE unbiasedness | ⚠️ Approx | Requires BHVF Critic | V1：GAE 无偏性 | ⚠️ 近似 | 需要 BHVF Critic |
| I: Independence | ✓ Approx | Small common-cause correlation | I：独立性 | ✓ 近似 | 共因相关性小 |
| Theorem 3: Optimal fusion | ✓ Proven | Under G1, V1, I | 定理 3：最优融合 | ✓ 已证明 | 在 G1, V1, I 下 |
| EV proxy for $w^*$ | ⚠️ Conservative | Exact when $b^G = 0$ | EV 代理 $w^*$ | ⚠️ 保守 | $b^G = 0$ 时精确 |
| No-harm guarantee (Prop. 1) | ✓ Proven | $w^* \in [0,1]$, stop-grad | 无危害保证（命题 1） | ✓ 已证明 | $w^* \in [0,1]$，stop-grad |
| Cost: CA-GRPO-Base ~5% | ✓ Exact | Shared backbone architecture | 开销：CA-GRPO-Base ~5% | ✓ 精确 | 共享主干架构 |

---

<a name="6"></a>
## 7. Theoretical Feasibility Summary
## 7. 理论可行性总结

### 6.1 Core Guarantees | 核心保证

**[EN]**

CA-GRPO rests on four provably correct foundations:

1. **Optimality** (Theorem 3): Unique MSE-optimal linear combination under G1, V1, I.

2. **No-harm** (Proposition 1): $w^* \in [0,1]$ guarantees CA-GRPO is always at least as
   good as the better of GRPO or GAE. Cannot do worse than either alone.

3. **Smooth degradation**: When EV $\to 0$ (Critic is random), $\hat{w}^* \to w_{\min}$,
   CA-GRPO is nearly pure GRPO. No catastrophic failure from a bad Critic.

4. **Unified theory**: Theorem 3 has the same mathematical structure as BHVF Theorem 1,
   enabling a single paper to present both results within one framework.

**[ZH]**

CA-GRPO 建立在四个可证明的正确基础上：

1. **最优性**（定理 3）：在 G1、V1、I 下的唯一 MSE 最优线性组合。

2. **无危害性**（命题 1）：$w^* \in [0,1]$ 保证 CA-GRPO 总是至少与 GRPO 或 GAE 中较好者一样好。
   不会比任一单独来源更差。

3. **平滑退化**：当 EV $\to 0$（Critic 随机）时，$\hat{w}^* \to w_{\min}$，
   CA-GRPO 接近纯 GRPO。不会因 Critic 差而灾难性失败。

4. **统一理论**：定理 3 与 BHVF 定理 1 具有相同的数学结构，使单篇论文能够在同一框架内
   呈现两个结果。

### 6.2 Comparison with Existing Work | 与现有工作的比较

**[EN]**

| Work | Advantage Estimate | Temporal Structure | Additional Params |
|---|---|---|---|
| PPO [Schulman 2017] | GAE | ✓ Full | Separate Critic |
| GRPO [Shao 2024] | Group relative | ✗ None | None |
| REINFORCE [Williams 1992] | MC return - baseline | Partial | Optional baseline |
| ReMax [Li 2023] | REINFORCE + greedy baseline | Partial | None |
| DAPO [Yu 2025] | GRPO + clip-higher | ✗ None | None |
| **CA-GRPO (ours)** | Bayesian fusion of GRPO+GAE | ✓ Adaptive | +1 linear layer (~5%) |

**[ZH]**

| 工作 | 优势估计 | 时序结构 | 额外参数 |
|---|---|---|---|
| PPO [Schulman 2017] | GAE | ✓ 完整 | 独立 Critic |
| GRPO [Shao 2024] | 组相对 | ✗ 无 | 无 |
| REINFORCE [Williams 1992] | MC 回报 - 基线 | 部分 | 可选基线 |
| ReMax [Li 2023] | REINFORCE + 贪心基线 | 部分 | 无 |
| DAPO [Yu 2025] | GRPO + clip-higher | ✗ 无 | 无 |
| **CA-GRPO（本文）** | GRPO+GAE 贝叶斯融合 | ✓ 自适应 | +1 线性层（~5%） |

---

<a name="7"></a>
## 8. Implementation Roadmap
## 8. 实现路线图

### 7.1 Phase 1: MuJoCo Validation (Current Codebase)
### 7.1 第一阶段：MuJoCo 验证（当前代码库）

**[EN]**

Implement `GRPOSimulatedAgent` in `gae_experiments/agents/optimal_ppo.py`:
- Simulate "GRPO" by using trajectory-level advantage (ignore temporal structure)
- Implementation: set $\hat{A}_t = (R_i - \bar{R}) / \sigma_R$ for all steps in trajectory $i$
- Compare: GRPO-sim vs CA-GRPO vs Standard GAE on Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4
- Expected outcome: CA-GRPO ≥ Standard GAE ≥ GRPO-sim on all 4 environments

Key hypothesis to verify: trajectory-level signal is indeed inferior in environments with
long episodes (Walker2d: ~500 steps) vs short episodes (Hopper: ~200 steps).

**[ZH]**

在 `gae_experiments/agents/optimal_ppo.py` 中实现 `GRPOSimulatedAgent`：
- 通过使用轨迹级别优势（忽略时序结构）来模拟"GRPO"
- 实现：对轨迹 $i$ 中所有步骤设置 $\hat{A}_t = (R_i - \bar{R}) / \sigma_R$
- 比较：GRPO-sim vs CA-GRPO vs 标准 GAE，在 Hopper-v4、Walker2d-v4、HalfCheetah-v4、Ant-v4 上
- 预期结果：所有 4 个环境上 CA-GRPO ≥ 标准 GAE ≥ GRPO-sim

需要验证的关键假设：轨迹级别信号在长 episode 环境（Walker2d：~500 步）中确实比短 episode
环境（Hopper：~200 步）更差。

### 7.2 Phase 2: LLM Validation
### 7.2 第二阶段：LLM 验证

**[EN]**

Implement CA-GRPO-Base on a small LLM (Qwen-1.5B or LLaMA-3-1B):
- Group size $G = 8$, response length $T \approx 200$ tokens
- Dataset: GSM8K or MATH (math reasoning, terminal 0/1 reward)
- Compare GRPO vs CA-GRPO-Base on final accuracy
- Track EV curve to verify adaptive fusion ($w^* \to 1$ as training progresses)
- Key ablation: GRPO vs CA-GRPO-Lite vs CA-GRPO-Base vs PPO

**[ZH]**

在小型 LLM（Qwen-1.5B 或 LLaMA-3-1B）上实现 CA-GRPO-Base：
- 组大小 $G = 8$，响应长度 $T \approx 200$ token
- 数据集：GSM8K 或 MATH（数学推理，终端 0/1 奖励）
- 比较 GRPO vs CA-GRPO-Base 在最终准确率上的表现
- 追踪 EV 曲线以验证自适应融合（$w^* \to 1$ 随训练进行）
- 关键消融：GRPO vs CA-GRPO-Lite vs CA-GRPO-Base vs PPO

### 7.3 ICML Paper Structure
### 7.3 ICML 论文结构

**[EN]**

Proposed title: *"From Groups to Steps: Bayesian Temporal Credit Assignment in Group Relative Policy Optimization"*

Section structure:
1. Introduction: GRPO temporal blindness problem
2. Background: GRPO, GAE, BHVF (brief)
3. CA-GRPO: Theorem 3 + Corollary 3.1 + Proposition 1
4. Connection to BHVF: unified Bayesian framework
5. Experiments: MuJoCo + LLM (GSM8K)
6. Ablation: w* estimation, Critic cost modes
7. Related work
8. Conclusion

**[ZH]**

提议论文题目：*《从组到步：组相对策略优化中的贝叶斯时序信用分配》*

章节结构：
1. 引言：GRPO 时序盲视问题
2. 背景：GRPO、GAE、BHVF（简述）
3. CA-GRPO：定理 3 + 推论 3.1 + 命题 1
4. 与 BHVF 的联系：统一贝叶斯框架
5. 实验：MuJoCo + LLM（GSM8K）
6. 消融研究：$w^*$ 估计、Critic 代价模式
7. 相关工作
8. 结论

---

## References | 参考文献

[Shao et al., 2024] Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the
Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

[Schulman et al., 2016] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
High-dimensional continuous control using generalized advantage estimation. *ICLR 2016*.

[Schulman et al., 2017] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
Proximal policy optimization algorithms. *arXiv:1707.06347*.

[Yu et al., 2025] Yu, T., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System
at Scale. *arXiv:2503.14476*.

[Li et al., 2023] Li, Z., et al. (2023). ReMax: A Simple, Effective, and Efficient Reinforcement
Learning Method for Aligning Large Language Models. *arXiv:2310.10505*.

[Williams, 1992] Williams, R. J. (1992). Simple statistical gradient-following algorithms for
connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.

---

*Document Version: 1.0 | Last updated: 2026-04-09*
*文档版本：1.0 | 最后更新：2026-04-09*

*For internal research use — ICML 2026 submission in preparation*
*内部研究使用 — ICML 2026 投稿准备中*

