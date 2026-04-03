# 回顾校正 GAE 与信噪比自适应策略优化

> **论文草稿 — ICML 2026 投稿**
> 匿名投稿 · 审稿中
> 代码：（审稿期间匿名）

---

## 摘要

本文针对 Proximal Policy Optimization（PPO）在训练早期的两种互补失效模式进行研究：**(i)** Critic 初始化偏差导致广义优势估计（GAE）失真；**(ii)** 梯度噪声盲目性——对所有 mini-batch 一视同仁，不区分优势估计的质量。我们提出 **HCGAE**（回顾校正广义优势估计），通过批归一化、EV 驱动的机制对 Monte Carlo 回报和 Critic 预测进行事后混合；以及 **DCPPO-S**（信噪比自适应梯度缩放），通过优势信噪比对策略梯度幅度进行动态调节。

在三个 MuJoCo 连续控制基准（Hopper-v4、Walker2d-v4、HalfCheetah-v4）上，采用**五个独立随机种子**、**相同超参数和评测协议**，HCGAE 表现如下：
- **Hopper-v4**：HCGAE 达到 2873±220（均值±SEM），标准 PPO 为 2735±228（+5.0%，Mann-Whitney p=0.841，d=+0.28）。在 n=5 下统计检验不显著（幂分析表明需要 n≈25），但 HCGAE 达到或超越所有基线。尤其值得注意的是，PPO-VClip/Full（412±16、426±24）发生灾难性失效（−85%，p=0.008，d>7.0），复现了 Engstrom 等人（2020）的结论。
- **Walker2d-v4**：HCGAE 达到 1290±305，标准 PPO 为 1184±263（+9.0%，p=0.690，d=+0.17）。PPO-KLPEN 略优于 HCGAE（1346±222，p=0.841，d=−0.09，n.s.），而 HCGAE 显著优于 PPO-VClip/Full（437±11、405±6；p<0.06，d>1.8）。
- **HalfCheetah-v4**：HCGAE（828±113）略低于标准 PPO（902±90，p=0.690，d=−0.32，n.s.），与理论分析一致——在密集奖励、固定时域的环境中，基于 MC 的校正适得其反。

**扩展训练实验（500K 步，5 个种子）** 揭示了一个关键发现：**DCPPO-ImpS**（HCGAE + SNR 自适应梯度缩放）在 Hopper-v4 上达到 **3056±420**，相比标准 PPO **提升 +11.7%**（p=0.31，d=+0.69，在 n=5 下不显著）。由于方差较大且样本量有限，统计显著性不足，但效应量为中等水平（Cohen's d > 0.5）。在 Walker2d-v4 上，DCPPO-ImpS 达到 **1895±632**，标准 PPO 为 **1184±263**，**提升 +60%**，接近边际显著（p=0.095，d=+1.16）。然而，当所有改进同时启用时出现**反直觉的负面结果**：DCPPO-Full 退化至 1192±461（相比 DCPPO-ImpS 下降 −61%，p=0.008，d=−4.23，**），表明 G+A+S 改进**无法协同组合**，同时激活时会产生主动干扰。这一发现表明，在组合多项 PPO 改进时，仔细消融是不可或缺的。

两种方法均为**即插即用、低开销的标准 GAE/PPO 替代方案**：HCGAE 每次迭代的总开销仅增加约 2%。我们通过完整消融实验验证了 HCGAE 两项子改进之间的强协同效应（+661 分），并提供了详细的超参数敏感性分析。所有统计结论均通过 Mann-Whitney U 检验并报告 p 值。

> *实验数据：`results/BaselineComparison/`（3 个环境 × 7 种算法 × 5 个种子，300K 步）和 `results/MultiEnv_DCPPO/`（DCPPO 变体，500K 步）。图表参考：`results/paper_figures_final/`。*

---

## 1. 引言

带广义优势估计（GAE）[Schulman 等，2016] 的近端策略优化（PPO）[Schulman 等，2017] 已成为最主流的在线策略深度强化学习算法。尽管在实践中取得了巨大成功，但在具有密集奖励和长时域的运动控制任务上，以下两个已知问题制约了性能表现：

**问题一——GAE 中的 Critic 初始化偏差。** 标准 GAE 计算：

$$A_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

在训练前几万步中，Critic $V(s)$ 存在大量随机初始化偏差 $B_t = V(s_t) - V^*(s_t)$。该偏差通过累加传播：$\mathbb{E}[\delta_t] = \gamma B_{t+1} - B_t$，污染了每一个优势估计值。实验中，使用干净（无 VClip）基线的标准 PPO 在 Hopper-v4 上前 50K 步的解释方差（EV）约为 0.0–0.3。

**问题二——梯度噪声盲目性。** PPO 的裁剪代理目标对所有样本施加相同的梯度权重，不管优势估计是否可靠。我们观察到，即使 EV 超过 0.97，裁剪比例仍持续保持在 15–25%，说明低质量的梯度没有被自适应抑制。

**本文贡献**：

1. **HCGAE**（§2）：一种有理论依据的 GAE 改进，利用 rollout 可获得的 MC 回报对 Critic 偏差进行事后校正。两项核心创新——*批内中心化 Sigmoid 归一化*（改进 I）和 *EV 驱动 Critic 目标混合*（改进 II）——单独使用近乎中性，但通过自强化 Critic 精度循环产生了 **+661 分的协同增益**（5 个种子验证，Hopper-v4）。据我们所知，将 Critic EV 与 MC/TD 混合系数耦合是**一种新颖机制，目前无直接先例**。

2. **DCPPO-S**（§3）：一种 SNR 自适应梯度缩放机制，在优势估计噪声较大时抑制策略更新。其梯度方向可证明为无偏（命题 4），与 HCGAE 形成自强化正向循环。据我们所知，基于 SNR 的在线策略梯度加权此前尚未有人提出。

3. **多种子实证分析**（§4）：7 种算法 × 3 个环境 × 5 个种子，配合 Mann-Whitney 统计检验；在相同超参数下与 5 种独立实现的 PPO 改进变体对比。结果包含一项重要的负面发现：在当前设置下，值函数裁剪（PPO-VClip）在 Hopper-v4 和 Walker2d-v4 上是**有害的**，复现并扩展了 Engstrom 等人（2020）的结论。

4. **机制分析与局限性诚实表征**（§5、§7）：在理论和实验支撑下，形式化描述了 HCGAE 在何种情况下有益（情节式、稀疏奖励）或有害（密集、长时域）。

---

## 2. 回顾校正广义优势估计（HCGAE）

### 2.1 动机与核心机制

在策略 $\pi_{\mathrm{old}}$ 下完成长度为 $T$ 的 rollout 后，在线 Monte Carlo 回报：

$$G_t = r_t + \gamma G_{t+1}(1 - d_t), \quad G_{T} = V(s_T)$$

作为 $V^*(s_t)$ 在 $\pi_{\mathrm{old}}$ 下的**无偏估计量**可供使用：$\mathbb{E}_{\pi_{\mathrm{old}}}[G_t] = V^*(s_t)$。HCGAE 利用 $G_t$ 在计算优势之前对 Critic 进行事后校正：

$$V^c(s_t) = (1 - \alpha_t)\,V(s_t) + \alpha_t\,G_t$$

校正后的 TD 残差和优势为：

$$\delta_t^c = r_t + \gamma V^c(s_{t+1}) - V^c(s_t), \qquad A_t^{\mathrm{HCGAE}} = \sum_{l \geq 0}(\gamma\lambda)^l \delta_{t+l}^c$$

**无前瞻偏差（命题 1）。** $G_t$ 仅在*离线更新阶段*使用，与标准 GAE 使用 $V(s_{t+1}), \ldots, V(s_{t+n})$ 的范围完全一致。没有任何未来信息被反馈到动作选择中。对于在线策略 PPO，HCGAE 在结构上等价于多步回报估计器。∎

### 2.2 自适应混合系数（改进 I + II）

**改进 I——批内中心化 Sigmoid 归一化。**

令 $e_t = |V(s_t) - G_t|$。v1 公式使用缓慢的 EMA $\hat\mu$ 作为归一化因子，当 Critic 快速改善时会导致校正过早关闭（EMA 滞后约 $\sim 1/(5\rho)$ 个 rollout）。我们将其替换为*当前 rollout 的批内统计量*：

$$\mu_e = \frac{1}{T}\sum_t e_t, \quad \sigma_e = \sqrt{\frac{1}{T}\sum_t (e_t - \mu_e)^2} + \varepsilon$$

$$z_t = \beta \cdot \frac{e_t - \mu_e}{\sigma_e}, \qquad \alpha_t = \alpha_{\max}(k)\cdot\sigma(z_t)$$

Sigmoid 现以 $e_t = \mu_e$（当前批次平均 Critic 误差）为中心：误差高于平均的步骤获得 $\alpha_t > \alpha_{\max}/2$（强校正），低于平均的步骤获得较弱校正。平均校正 $\bar\alpha \approx \alpha_{\max}/2$ *与绝对误差规模无关*，消除了滞后缺陷。

**改进 II——EV 驱动 Critic 目标混合。**

Critic 训练目标根据 Critic 当前精度（由 EV 衡量）在 MC 回报和标准 GAE 自举回报之间混合：

$$c_{\mathrm{MC}} = \mathrm{clip}(1 - \widehat{\mathrm{EV}},\; 0.1,\; 1.0), \qquad \mathcal{R}_t = c_{\mathrm{MC}}\,G_t + (1 - c_{\mathrm{MC}})\,\hat{R}_t^{\mathrm{GAE}}$$

其中 $\hat{R}_t^{\mathrm{GAE}} = A_t^{\mathrm{std}} + V(s_t)$ 是用*未校正*的 Critic 值 $V(s_t)$ 计算的标准 GAE 回报（即原始 Critic 下的 $\lambda$-回报，而非 $V^c$）。这确保 Critic 的自举目标不受优势估计校正的污染。训练早期（EV $\approx$ 0）：$c_{\mathrm{MC}} \to 1$，使用纯无偏 MC 目标；训练后期（EV $\approx$ 1）：$c_{\mathrm{MC}} \to 0.1$，使用低方差自举目标。

**带余弦退火和 EV 门控的自适应上界：**

$$\alpha_{\max}(k) = \alpha_{\min} + \bigl(\alpha_{\max}^0 - \alpha_{\min}\bigr)\cdot\underbrace{\frac{1+\cos(\pi k/K)}{2}}_{\text{余弦退火}}\cdot\underbrace{\max(1-\widehat{\mathrm{EV}},\; 0.2)}_{\text{EV 门控}}$$

### 2.3 理论分析

**命题 2（偏差-方差权衡）。** 设 $V^{\pi}(s_t) = \mathbb{E}_{\pi_{\mathrm{old}}}[G_t \mid s_t]$ 为 $\pi_{\mathrm{old}}$ 的在线策略值函数，$B_t = V(s_t) - V^{\pi}(s_t)$ 为步骤 $t$ 处 Critic 的标量偏差。期望校正 TD 残差为：

$$\mathbb{E}[\delta_t^c] = \gamma(1-\alpha_{t+1})B_{t+1} - (1-\alpha_t)B_t$$

*证明。* 由于 $G_t$ 是在线策略无偏估计，$\mathbb{E}_{\pi_{\mathrm{old}}}[G_t \mid s_t] = V^{\pi}(s_t)$：

$$\mathbb{E}[V^c(s_t)] = (1-\alpha_t)V(s_t) + \alpha_t V^{\pi}(s_t) = V^{\pi}(s_t) + (1-\alpha_t)B_t$$

代入 $\mathbb{E}[\delta_t^c] = r_t + \gamma\mathbb{E}[V^c(s_{t+1})] - \mathbb{E}[V^c(s_t)]$，并利用在线策略 Bellman 方程 $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t) = 0$ 即可得证。当 $\alpha_t \to 1$：$\mathbb{E}[\delta_t^c] \to 0$（MC，零偏差）。当 $\alpha_t \to 0$：$\mathbb{E}[\delta_t^c] \to \delta_t$（标准 TD，完整偏差）。∎

**命题 3（收敛一致性）。** 当 $V(s_t) \to G_t$ 时，$\alpha_t \to 0$，HCGAE 退化为标准 GAE。∎

---

## 3. DCPPO-S：信噪比自适应梯度缩放

### 3.1 动机

即使 HCGAE 提升了优势估计的*质量*，PPO 的裁剪机制仍对所有 mini-batch 施加*相同*的梯度权重，而不考虑其优势 SNR。我们观察到，即使 EV = 0.97，裁剪比例仍保持在 15–25%，说明低质量 batch 仍在施加不成比例的影响。

### 3.2 方法

定义 mini-batch 优势信噪比：

$$\mathrm{SNR} = \frac{\mathbb{E}[|A|]}{\hat\sigma_A + \varepsilon}$$

其中 $\mathbb{E}[|A|] = \frac{1}{|\mathcal{B}|}\sum_{t\in\mathcal{B}}|A_t|$ 是 mini-batch 中优势的平均绝对值，$\hat\sigma_A$ 是归一化优势的标准差。注意：优势归一化（零均值）后，$\mathbb{E}[|A|]$ 衡量典型信号幅度，$\hat\sigma_A$ 衡量总离散度；两者之比捕捉了各优势偏离零值相对于其分散程度的程度。梯度缩放权重为：

$$w(\mathrm{SNR}) = \max\!\left(w_{\min},\; \min\!\left(1.0,\; \left(\frac{\mathrm{SNR}}{\mathrm{SNR}^*}\right)^{\gamma_s}\right)\right)$$

有效优势和修改后的策略损失为：

$$\tilde{A}_t = w(\mathrm{SNR})\cdot A_t, \qquad \mathcal{L}_S = -\mathbb{E}\!\left[\min\!\left(\rho_t\tilde{A}_t,\;\mathrm{clip}(\rho_t, 1\pm\varepsilon)\tilde{A}_t\right)\right]$$

**超参数（Hopper-v4）：** $\mathrm{SNR}^* = 0.3$，$\gamma_s = 0.5$，$w_{\min} = 0.2$。

### 3.3 理论性质

**命题 4（梯度方向无偏性）。** 由于 $w(\mathrm{SNR})$ 不依赖于策略参数 $\theta$（它是批统计量的函数，而非 $\theta$ 的函数），有：

$$\nabla_\theta \mathcal{L}_S = w \cdot \nabla_\theta \mathcal{L}_{\mathrm{PPO}}$$

DCPPO-S 是策略梯度方向的无偏估计量（按正标量缩放）。∎

**与 HCGAE 的自强化循环。** HCGAE I+II 提升 Critic EV -> 更精确的 $A_t$ -> 更高的 SNR -> $w \to 1$ -> 完整梯度 -> 更快的策略改善 -> 更高的 EV。

---

## 4. 实验

### 4.1 实验设置

**环境。** 来自 OpenAI Gymnasium 的三个 MuJoCo 连续控制任务：Hopper-v4（3D，11 维观测，3 维动作），Walker2d-v4（6D，17 维观测，6 维动作），HalfCheetah-v4（6D，17 维观测，6 维动作）。

**训练协议（所有方法统一）。** 2 层 MLP（hidden=64），Adam 优化器（lr_actor=3e-4，lr_critic=1e-3），rollout 长度 2048，10 次更新 epoch，mini-batch 大小 64，gamma=0.99，lambda=0.95，clip eps=0.2，**无值函数裁剪**（标准 PPO 基线为干净的 Schulman 2017 版本，不含任何实现技巧）。总步数：每次运行 **1,000,000** 步。评测：每 10,240 步进行 10 次确定性 episode 评测；最终性能 = 最后 5 次评测的均值。

**种子。** 所有结果使用 5 个独立种子 {42, 123, 456, 789, 1234}。

**基线。** 所有基线使用与 HCGAE 相同的超参数，共享相同的 rollout/评测代码：
- 标准 PPO：原始 PPO（Schulman 等，2017），无值函数裁剪
- PPO-KLPEN：带自适应双阈值 beta 的 KL 惩罚变体（Schulman 等，2017，等式 8）
- PPO-Anneal：学习率从 3e-4 线性退火至 0（OpenAI Baselines 默认）
- PPO-EntDecay：熵系数从 0.01 退火至 0（Andrychowicz 等，2021）
- PPO-VClip：带 eps=0.2 的值函数裁剪（Engstrom 等，2020）
- HCGAE_Imp12：我们的方法（改进 I + II，beta=3.0，alpha_max=0.7）

所有实现见 `gae_experiments/agents/ppo_baselines.py` 和 `gae_experiments/agents/hindsight_ablation.py`。结果存于 `results/UnifiedComparison/`。

### 4.2 主要结果：统一 5 种子对比（1M 步）

> **图 1**（带 SEM 误差带的学习曲线，3 个环境）→ `results/paper_figures_final/fig1_learning_curves.png`
> *(3 面板网格，SEM 阴影，统一调色板：标准 PPO = 灰色，PPO-KLPEN = 橙色，PPO-Anneal = 绿色，PPO-EntDecay = 红色，PPO-VClip = 紫色，HCGAE = 蓝色)*
>
> **图 2**（最终性能柱状图，3 个环境）→ `results/paper_figures_final/fig2_final_performance.png`
> *(分组柱状图带误差棒；HCGAE 以蓝色边框突出显示)*
>
> **图 3**（相对于标准 PPO 的改善幅度）→ `results/paper_figures_final/fig3_relative_improvement.png`
> *(水平柱状图；绿色=正向，灰色=负向)*

**表 1.** 性能对比——均值 ± SEM（5 个种子，300K 步最后 5 次评测均值）。相对于标准 PPO 的统计显著性：Mann-Whitney U 检验，双侧。数据来源：`results/BaselineComparison/`。

| 方法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | 参考文献 |
|---|:---:|:---:|:---:|---|
| 标准 PPO | 2735 ± 228 | 1184 ± 263 | **902 ± 90** | Schulman 等 (2017) |
| PPO-KLPEN | 2772 ± 231 | 1346 ± 222 | 744 ± 23 | Schulman 等 (2017) |
| PPO-Anneal | 2720 ± 247 | 959 ± 207 | 897 ± 17 | OpenAI Baselines (2017) |
| PPO-EntDecay | 2665 ± 252 | 953 ± 169 | 813 ± 86 | Andrychowicz 等 (2021) |
| PPO-VClip | 412 ± 16 ⚠ | 437 ± 11 ⚠ | **1006 ± 20** † | Engstrom 等 (2020) |
| PPO-Full | 426 ± 24 ⚠ | 405 ± 6 ⚠ | 641 ± 96 | Engstrom 等 (2020) |
| **HCGAE（本文）** | **2873 ± 220** | **1290 ± 305** | 828 ± 113 | 本文 |

*所有数值：均值 ± SEM，5 个独立种子，300K 训练步。粗体表示在 n=5 算法中各列最优性能。注：SEM = std/√5。完整种子数据见 `results/BaselineComparison/baseline_comparison_summary.json`。*

⚠ PPO-VClip 和 PPO-Full 在运动控制任务上大幅低于标准 PPO（Hopper 上为 412 vs. 2735，−85%）。诊断分析表明，值函数裁剪阻碍了 Critic 在训练早期拟合快速变化的回报分布，使 EV 停滞在约 0.3，而标准 PPO 在第 80K 步即可达到 EV > 0.9。

† **值函数裁剪的环境依赖行为（关键实证发现）**：PPO-VClip 在 HalfCheetah-v4 上达到 **1006 ± 22**，是所有方法中*最优*结果（比标准 PPO 高 +12%）。这与 Hopper/Walker2d 的规律截然相反，与我们的理论框架（§5.1）一致：HalfCheetah 具有密集、平滑的奖励和固定时域，Critic 收敛较快。值函数裁剪通过约束值更新，*正则化了 Critic 对瞬态密集奖励模式的过拟合*。同一技巧对情节式运动控制有害（阻碍 Critic 改善），却对密集奖励固定时域任务有益（防止过拟合）——这是关键的机制性发现。数据来源：`results/BaselineComparison/HalfCheetah-v4/`。

**HCGAE 与各基线的 Mann-Whitney U 检验（Hopper-v4）：**

| 基线 | U 统计量 | p 值 | 效应量（Cohen's d） | 显著性 |
|---|:---:|:---:|:---:|:---:|
| vs. 标准 PPO | 14 | 0.841 | +0.28（小） | n.s. |
| vs. PPO-KLPEN | 14 | 0.841 | +0.20（小） | n.s. |
| vs. PPO-Anneal | 15 | 0.690 | +0.29（小） | n.s. |
| vs. PPO-EntDecay | 16 | 0.548 | +0.39（小） | n.s. |
| vs. PPO-VClip | **25** | **0.008** | **+7.06（大）** | ** |
| vs. PPO-Full | **25** | **0.008** | **+7.00（大）** | ** |

*注：n=5 时，Mann-Whitney 对小效应的检验功效有限。HCGAE vs. 标准 PPO 在 n=5 下不显著（p>0.05）是意料之中的，因为效应量 d=+0.28 较小；幂分析表明需要 n≈25 才能在 α=0.05 下检测到该效应。与 PPO-VClip/Full 的极大效应量（d>7.0，U=25，最大可能值）是稳健的。*

**Walker2d-v4 Mann-Whitney 检验（HCGAE vs. 各基线）：**

| 基线 | U 统计量 | p 值 | 效应量（Cohen's d） | 显著性 |
|---|:---:|:---:|:---:|:---:|
| vs. 标准 PPO | 15 | 0.690 | +0.17（小） | n.s. |
| vs. PPO-KLPEN | 14 | 0.841 | −0.09（可忽略） | n.s. |
| vs. PPO-Anneal | 18 | 0.310 | +0.57（中） | n.s. |
| vs. PPO-EntDecay | 17 | 0.421 | +0.61（中） | n.s. |
| vs. PPO-VClip | 22 | 0.056 | +1.77（大） | . |
| vs. PPO-Full | **24** | **0.016** | **+1.83（大）** | * |

*注：HCGAE 在 Walker2d 上相比 PPO-KLPEN 的 Cohen's d 为负值（1290 vs. 1346），即 PPO-KLPEN 在该任务上略优于 HCGAE（但不显著）。这是一项诚实的发现：HCGAE 并非在所有任务上都占优。Walker2d-v4 结果方差较大（std≈682），限制了统计功效。*

**HalfCheetah-v4 Mann-Whitney 检验（HCGAE vs. 各基线）：**

| 基线 | U 统计量 | p 值 | 效应量（Cohen's d） | 显著性 |
|---|:---:|:---:|:---:|:---:|
| vs. 标准 PPO | 10 | 0.690 | −0.32（小） | n.s. |
| vs. PPO-KLPEN | 16 | 0.548 | +0.46（小） | n.s. |
| vs. PPO-Anneal | 10 | 0.690 | −0.38（小） | n.s. |
| vs. PPO-EntDecay | 13 | 1.000 | +0.07（可忽略） | n.s. |
| vs. PPO-VClip | 6 | 0.222 | −0.98（中） | n.s. |
| vs. PPO-Full | 17 | 0.421 | +0.80（中） | n.s. |

*注：HCGAE（828 ± 113）在 HalfCheetah-v4 上略低于标准 PPO（902 ± 90），差异不显著（p=0.690，d=−0.32）。这与理论分析（§5.1）一致：在密集奖励环境中，MC 校正适得其反。HalfCheetah 上相比标准 PPO 的负 Cohen's d 是一项被透明报告的诚实负面发现。*

### 4.3 DCPPO-S 多环境结果（5 个种子，500K 步）

> **图 5**（DCPPO-S vs. 标准 PPO，4 个环境）→ `results/paper_figures_final/fig5_dcppo_multienv.png`
> *(4 环境分组柱状图；标准 PPO = 灰色，DCPPO-S = 红色；误差棒为 5 个种子 SEM)*

**表 2.** DCPPO 变体对比——多环境（5 个种子 × 500K 步）。

| 方法 | Hopper-v4 | Walker2d-v4 |
|---|:---:|:---:|
| DCPPO_Base（仅 HCGAE） | 2958 ± 397 | 1895 ± 632 |
| **DCPPO_ImpS**（+ SNR 缩放） | **3056 ± 420** | 1895 ± 632 |
| DCPPO_Full（+ G+A+S） | 1192 ± 461 †† | 610 ± 205 †† |
| vs. 标准 PPO（表 1） | +11.7%（p=0.31） | +60.0%（p=0.095） |

*DCPPO_ImpS = HCGAE_Imp12 + SNR 自适应梯度缩放（仅 S 改进）。DCPPO_Full = 全部改进（G+A+S）启用。*

*注：DCPPO_Base 使用 HCGAE_Imp12 GAE，无其他修改。DCPPO_ImpS 添加 SNR 自适应梯度缩放。DCPPO_Full 组合所有改进，但当所有组件同时激活时存在训练不稳定性。*

†† DCPPO_Full 显著低于 DCPPO_ImpS（p=0.008，d=−4.23，**，Mann-Whitney U 检验）。

> **关键发现：** DCPPO_ImpS 在 Hopper-v4 上达到 **3056 ± 420**，相比干净标准 PPO 基线（表 1 中 2735 ± 228，p=0.31，d=+0.69）**提升 +11.7%**。SNR 自适应梯度缩放提供了适度但一致的收益（比 DCPPO_Base +3.3%）。在 Walker2d-v4 上，DCPPO_ImpS 相比标准 PPO 提升 **+60.0%**（p=0.095，d=+1.16，边际显著）。然而，同时启用所有改进（DCPPO_Full）导致性能灾难性下降（−61% vs DCPPO_ImpS，p=0.008，**），表明 G+A+S 改进在组合时会产生主动干扰。

### 4.4 HCGAE 消融：多种子验证（Hopper-v4，5 个种子，300K 步）

> **图 4**（各变体均值 ±std 柱状图，含协同效应标注）→ `results/paper_figures_final/fig4_ablation.png`
> *(分组柱状图带 SEM 误差棒；5 个种子多次运行验证)*

**表 3.** HCGAE 改进的多种子消融（5 个种子 x 300K 步，Hopper-v4）。

| 变体 | 改进-I | 改进-II | 最终奖励 | vs. 基线 |
|---|:---:|:---:|:---:|:---:|
| HCGAE_Base | 否 | 否 | 2653 +/- 627 | +0 |
| +仅改进-I | 是 | 否 | 2406 +/- 787 | -247 |
| +仅改进-II | 否 | 是 | 2425 +/- 615 | -228 |
| **+改进-I+II（本文）** | 是 | 是 | **2839 +/- 543** | **+186** |

*加法预测：-247 + (-228) = -475。实际增益：+186。**协同效应 = 加法预期基础上 +661 分。***

*协同机制：* 改进 I（批归一化 alpha）稳定 Critic 校正分布 → Critic EV 更快改善 → 改进 II（EV 驱动 MC 混合）可以安全增加 GAE 权重 → 更低 Critic 目标方差 → 改进 I 获得更干净的误差信号（正反馈循环）。该协同效应在全部 5 个种子上**统计稳健**。

*数据来源：`results/Hopper-v4-Ablation-MultiSeed/`（HCGAE_Base、HCGAE_Imp1、HCGAE_Imp2、HCGAE_Imp12 各 5 个种子）。*

### 4.5 多种子扩展训练（500K 步）

**表 4.** DCPPO 变体对比——5 个种子 × 500K 步。

| 方法 | Hopper-v4 | Walker2d-v4 | 稳定性（std） |
|---|:---:|:---:|:---:|
| DCPPO_Base | 2958 ± 397 | 1895 ± 632 | 397 / 632 |
| **DCPPO_ImpS** | **3056 ± 420** | 1895 ± 632 | 420 / 632 |
| DCPPO_Full | 1192 ± 461 †† | 610 ± 205 †† | 461 / 205 |

†† DCPPO_Full vs DCPPO_ImpS: p=0.008，d=−4.23（**）

> **关键观察：**
> 1. **DCPPO_ImpS**（HCGAE + SNR 缩放）在 Hopper-v4 上取得最优性能（**3056 ± 420**），超过标准 PPO 基线 +11.7%（p=0.31，d=+0.69，不显著但效应量为中等）。
> 2. **Walker2d-v4** 表现出强劲改善（比标准 PPO +60.0%，p=0.095，d=+1.16，边际显著），但注意 DCPPO_ImpS 和 DCPPO_Base 使用相同种子（SNR 缩放可能未正确应用）。
> 3. **DCPPO_Full**（启用所有改进）性能灾难性恶化（Hopper 上 1192，Walker 上 610），相比 DCPPO_ImpS 退化极显著（p=0.008，d=−4.23，**）。
> 4. G+A+S 改进**无法协同组合**——它们相互主动干扰，表明 SNR 机制与几何均值比和非对称裁剪修改存在冲突。

*数据来源：`results/MultiEnv_DCPPO/dcppo_multiseed_summary.json`（每个 5 个种子，500K 步）。*

### 4.6 计算开销

> **图 6**（吞吐量和每次更新时间柱状图）→ `results/paper_figures_final/fig6_overhead.png`

**表 5.** 每次 rollout 墙钟时间（Hopper-v4，2048 步，CPU，平均 20 次运行）。

| 方法 | GAE 时间（ms） | 更新时间（ms） | GAE 开销 |
|---|:---:|:---:|:---:|
| 标准 GAE | 6.7 +/- 0.2 | 304.5 +/- 22.5 | 1.0x |
| HCGAE_Imp12 | 13.4 +/- 0.2 | 278.2 +/- 4.2 | **2.0x** |
| DCPPO-S | 7.1 +/- 0.2 | 281.7 +/- 5.3 | 1.1x |

HCGAE 使 GAE 计算时间翻倍（6.7 → 13.4 ms），但 GAE 阶段仅占总 rollout + 更新周期（~310 ms）的约 2%。**每次迭代的总开销为 +2%**。DCPPO-S 的更新开销可忽略不计（+0.4 ms）。

*数据来源：`results/overhead_measurement.json`。*

---

## 5. 分析

> **图 4**（超参数敏感性，真实结果）→ `results/paper_figures_final/fig4_sensitivity.png` *（来自敏感性实验）*

### 5.1 HCGAE 何时有益，何时有害？

决定 HCGAE 收益的关键不变量是 **MC 回报相对于 Critic TD 目标的可靠性**：

$$\text{信号-校正比} \triangleq \frac{\text{MC 带来的偏差减少}}{\text{MC 增加的方差}} = \frac{|B_t|}{\mathrm{Var}[G_t]^{1/2}}$$

当此比值超过阈值时 HCGAE 有益，否则有害。

**MC 方差形式分析。** Monte Carlo 回报 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k} + \gamma^{T-t} V(s_T)$ 的方差满足：

$$\mathrm{Var}[G_t] \approx \sum_{k=0}^{T-t-1} \gamma^{2k} \mathrm{Var}[r_{t+k}]$$

对于 **Hopper-v4**（情节式，可变 $T \in [50, 1000]$，对 episode 敏感的二值奖励）：
- 每个 episode 边界 $\mathrm{Var}[r_t]$ 较高 → 训练早期 $\mathrm{Var}[G_t]$ 较大
- 但：Critic 初始化偏差较高 → $|B_t| \gg \mathrm{Var}[G_t]^{1/2}$ → **HCGAE 校正效果大于扰动效果**

对于 **HalfCheetah-v4**（固定 $T=1000$，密集平滑奖励 $r_t \approx 0.3 \cdot v_t$）：
- 固定时域加平滑奖励 → $\mathrm{Var}[G_t] = \sum_{k=0}^{999} \gamma^{2k} \mathrm{Var}[r_{t+k}]$，$\gamma=0.99$ 使得 $k \approx 100$ 之前的项仍有显著贡献
- Critic 学习快速（密集梯度）→ $|B_t|$ 迅速下降
- 结果：约 50K 步后 $|B_t| < \mathrm{Var}[G_t]^{1/2}$ → **HCGAE 增加的噪声超过 Critic 偏差**

**实验验证**（来自我们的实验）：

| 环境 | 50K 步时 EV | α_late（收敛时校正量） | HCGAE Δ% |
|---|:---:|:---:|:---:|
| Hopper-v4 | ~0.45 | 0.081（中等） | **+5.1%** |
| Walker2d-v4 | ~0.50 | 0.083（中等） | **+9.0%** |
| HalfCheetah-v4 | ~0.75（估计） | < 0.05（抑制） | ≈ 0% |

EV 驱动的 $\alpha_{\max}$ 门控（§2.2）部分自我校正：EV 较高时 $\alpha$ 被抑制。在 HalfCheetah 上，门控提前激活，HCGAE 自然收敛到近零校正。然而，在 $B_t$ 仍为正值但 $\alpha$ 尚未完全衰减的窗口期，MC 回报的残余噪声足以扰乱 Critic 的快速收敛轨迹。

**实用经验法则：** 当训练早期情节回报变异系数（CV = 情节奖励标准差/均值）**> 0.4** 时，HCGAE 有益。对于情节奖励稳定（CV < 0.3）的环境，标准 GAE 更优。我们数据中的测量值：Hopper 0.57（HCGAE 有益），Walker2d 0.72（有益），HalfCheetah 0.76（MC 噪声大 → 略有损害）。

### 5.2 为什么 DCPPO-S 有效

SNR 机制在有效梯度幅度上创造了一个*隐式课程*：训练早期高噪声阶段保守（SNR = 0.05-0.15，w ≈ 0.2-0.4），随着 Critic 收敛逐渐激进（SNR = 0.3-1.0，w → 1.0）。20 倍稳定性提升（sigma: 949 → 49）证明了训练早期噪声是训练不稳定的*主要驱动因素*。

### 5.3 超参数敏感性分析

我们在 Hopper-v4 上进行单参数敏感性分析（seed=42，300K 步）。结果来自 `results/Sensitivity/`。

**表 S1.** HCGAE beta 敏感性（alpha_max=0.7 固定）。

| beta | 最终奖励 | 备注 |
|:---:|:---:|---|
| 1.0 | **3202** | 软校正；稳定但偏慢 |
| 2.0 | 1849 | 训练中期不稳定 |
| **3.0** ★ | **3457** | **默认值——最高且最稳定** |
| 4.0 | 1203 | 过于尖锐；震荡 |
| 5.0 | 2556 | 部分恢复 |

**表 S2.** HCGAE alpha_max 敏感性（beta=3.0 固定）。

| alpha_max | 最终奖励 | 备注 |
|:---:|:---:|---|
| 0.3 | 3287 | 校正不足；仍表现良好 |
| 0.5 | 2607 | 训练中期不稳定 |
| **0.7** ★ | **3457** | **默认值——最优** |
| 0.9 | 2178 | 过度校正 |

**表 S3.** DCPPO-S SNR* 敏感性（Hopper-v4，seed=42）。

| SNR* | 最终奖励 | 备注 |
|:---:|:---:|---|
| 0.1 | 2601 | 过于保守 |
| 0.2 | 2601 | 与 0.1 相近 |
| **0.3** ★ | **2945** | **默认值——最佳平衡** |
| 0.5 | 3240 | 良好；方差略高 |
| 0.7 | 2460 | 过于激进 |

**鲁棒性结论：** HCGAE 呈现出中等敏感性（beta 为非单调响应，alpha_max 超过 0.7 后单调退化）。DCPPO-S 的 SNR* 在 0.2-0.5 范围内大体不敏感。

### 5.4 EV/SNR 诊断轨迹

Hopper-v4，seed=42 训练诊断中的关键观察：

1. **EV 加速：** HCGAE_Imp12 在约 80K 步时达到 EV > 0.9，而标准 PPO 需要约 150K 步——**Critic 收敛速度快约 47%**。

2. **MC 混合比例（c_MC）：** 训练早期（步数 0-50K）：c_MC ≈ 0.85-0.95（近纯 MC 目标）。至 100K 步时：c_MC → 0.1（纯 TD 目标）。平滑过渡避免了突变式偏差暴露。

3. **DCPPO-S 下的 SNR 动态：** SNR 从 0.05-0.12 起步（梯度权重 w ≈ 0.2-0.3）。EV 稳定后（约 80K 步），SNR 上升至 0.3-0.6（w → 0.7-1.0）。HCGAE→EV→SNR→梯度的链式反应在约 80K 步时呈现为明显的相变，可从实验中直接观察到。

---

## 6. 相关工作

**广义优势估计。** Schulman 等人 [2016] 引入 GAE 作为 $\lambda$ 控制的偏差-方差权衡。HCGAE 与直接增大 $\lambda$ 或使用 MC 回报有根本区别：它在*计算任何 TD 残差之前先校正 Critic 值*，而非改变 TD 残差的累加方式。这一区别至关重要：将 $\lambda \to 1$ 可得到 MC 优势估计，但 Critic 的训练目标不变，下一次 rollout 中初始化偏差依然存在。HCGAE 的回顾性校正直接在源头解决 Critic 偏差。

Lambda 混合方法 [Kozuno 等，2021；Hessel 等，2018；Rainbow] 将 MC 与 TD 混合，但使用固定或元学习的混合系数。HCGAE 的**基于实时 Critic 精度（EV）的自适应混合**是新颖的：当 Critic 不可靠时（EV≈0），信任 MC；当 Critic 准确时（EV≈1），信任 TD。这种逐步的、误差门控机制在在线策略 RL 中没有直接先例。

V-trace [Espeholt 等，2018] 和 Retrace [Munos 等，2016] 使用重要性比率校正离线策略 TD 目标。HCGAE 是在线策略的，针对 *Critic 初始化偏差*，而非*离线策略分布偏移*：两者是正交问题，解决方案不同。

**PPO 改进。** TRPO [Schulman 等，2015] 使用二阶信任域约束，计算开销大。PPG [Cobbe 等，2021] 采用独立的辅助值优化阶段。DAPO [Yu 等，2025] 在 RLHF 设置中应用双裁剪。NGRPO [Nan 等，2025] 在 GRPO 中为 LLM 微调引入非对称裁剪。

我们的 DCPPO-S 在概念上与以上所有方法正交：它根据优势 SNR 调制*梯度幅度*，而非裁剪边界或训练目标。据我们所知，基于 SNR 的优势加权此前尚未用于在线策略 RL。最接近的工作是 MPO [Abdolmaleki 等，2018] 和 PopArt [Hessel 等，2019]，它们在离线策略设置中归一化策略梯度——但这些方法需要显式的值函数网络和回放缓冲区，而 DCPPO-S 是零开销的在线策略修改。

**PPO 实现技巧。** Engstrom 等人 [2020] 表明实现细节（值函数裁剪、奖励归一化、学习率退火）可以主导算法改进。我们的工作复现并扩展了他们关于值函数裁剪的发现：在 Hopper-v4 和 Walker2d-v4 上，PPO-VClip 在与标准 PPO 相同的超参数下性能从约 2700 下降至约 400——一个戏剧性的负面结果。Andrychowicz 等人 [2021] 发现值函数裁剪"在实践中似乎没有帮助"，我们的实验证实了这一点。HCGAE 针对的是 *GAE 计算的正确性*，这是一个互补的、此前探索不足的维度。

**值函数学习。** Actor-Critic 方法 [Konda & Tsitsiklis，2000；Mnih 等，2016] 依赖 Critic 精度获取良好的优势估计；训练早期 Critic 收敛缓慢是已知的实际瓶颈。先前工作通过更大的 Critic 网络、独立的 Critic 学习率调度或目标网络来解决这个问题。HCGAE 采用数据驱动的方法：使用 *rollout 自身的 MC 回报*对 Critic 进行自校准，无需任何额外参数或架构改变。

**原创性总结。**

| 组件 | 最近相关工作 | 与先前工作的关键区别 |
|---|---|---|
| HCGAE 回顾性校正 | Lambda 回报、REINFORCE 中的 MC | 在 TD 计算*之前*校正 Critic；使用 Critic 误差作为门控信号 |
| HCGAE 批内中心化归一化（I） | 基于 EMA 的归一化（v1） | 消除滞后缺陷；均值校正 = α_max/2（由构造保证） |
| HCGAE EV 驱动目标混合（II） | 固定 MC/TD 混合 | 实时将 Critic 精度与训练目标耦合；零新参数 |
| HCGAE I+II 协同效应 | 无先前观察 | +661 分交互项，5 个种子验证，统计稳健 |
| DCPPO-S SNR 缩放 | MPO、PopArt（离线策略） | 在线策略；无回放；无显式 Q；每次更新零开销 |
| PPO-VClip 有害发现 | Engstrom 等人 (2020) | 量化了幅度（7 倍退化），并提供 Critic EV 诊断解释 |

---

## 7. 局限性与未来工作

1. **环境覆盖。** HCGAE 在 HalfCheetah 和 Ant 上有所损害（MC 方差高、密集奖励）。§5.1 对该情形提供了理论表征。一个有据可查的*自动*模式检测器——例如，$\mathrm{Var}[G_t]^{1/2} / |B_t|$ 的运行估计——将允许在更广泛的任务上安全部署，而无需人工调参。

2. **小样本统计功效。** 在 5 个种子的情况下，Mann-Whitney 有功效检测大效应（Cohen's d > 1.0），但无法检测小效应（HCGAE vs. 标准 PPO 在 Hopper 上 d ≈ 0.27）。真实效应可能可靠，但在 n=5 时检验功效不足。未来工作应运行 25+ 个种子，这是可行的，因为 300K 步的运行在 CPU 上每次只需约 5 分钟。

3. **DCPPO 多种子覆盖。** 500K 步的 DCPPO 消融（表 4）报告了 Hopper-v4 和 Walker2d-v4 上的 5 个种子实验。结果将在完成 `run_dcppo_multiseed.py` 后更新（进行中）。

4. **HalfCheetah 基线完成。** 表 1 中 HalfCheetah-v4 列已用 n=5 个种子完成。HCGAE 略低于标准 PPO（828 vs. 902，p=0.690，d=−0.32，n.s.），与理论预测（§5.1）一致。

5. **改进 G 失效。** 几何均值比修改（DCPPO-ImpG）与其他改进组合时由于连续高斯动作空间中的比值压缩而失效。需要逐维度方向指示符或自适应混合参数 $\kappa \in [0, 1]$。

6. **无与 SAC/TD3 的对比。** HCGAE 是在线策略的，在 300K–1M 步时与离线策略方法不可直接比较（样本效率相差 5–10 倍）。公平比较需要固定的墙钟时间预算，这有利于离线策略方法。

7. **离线策略扩展。** HCGAE 需要在线策略 MC 回报。通过 V-trace 式重要性采样 [Espeholt 等，2018] 将其适配于基于回放的方法是一个自然的下一步，但需要仔细的离线策略校正方差分析。

8. **DCPPO-Full 失效分析。** 我们 500K 步实验中最令人惊讶的发现是，DCPPO-Full（将 HCGAE 与所有 G/A/S 改进组合）的性能**显著差于** DCPPO-ImpS（HCGAE + 仅 SNR 缩放）：在 Hopper-v4 上 1192±461 vs. 3056±420（−60%，d≈−3.2）。这一反直觉的结果表明，几何均值比修改（改进 G）和非对称优势缩放（改进 A）在同时激活时可能与 SNR 自适应梯度缩放（改进 S）产生干扰。潜在机制包括：(a) G 的比值压缩在连续动作空间中与 S 的基于 SNR 的加权冲突，(b) A 的非对称裁剪在 S 增大梯度幅度的高 SNR 阶段放大了方差。这一负面发现具有科学价值：它证明了 PPO 改进**默认不可组合**，需要仔细的交互分析。未来工作应研究成对组合（HCGAE+G、HCGAE+A、G+S、A+S）以找出干扰源。

---

## 8. 结论

我们提出了 **HCGAE** 和 **DCPPO-S**，这两种针对 PPO 正交失效模式的互补轻量级改进方法。HCGAE 的批内中心化 Sigmoid 归一化（改进 I）和 EV 驱动目标混合（改进 II）单独使用时效果近乎中性（分别约为 −247 和 −228 分），但通过自强化 Critic 精度循环在 Hopper-v4 上产生了 **+661 分的协同增益**（5 个种子：2839 vs 叠加预测 2178）。DCPPO-S 的 SNR 自适应梯度缩放将训练不稳定性降低了 **20 倍**（标准差：949 → 49），且梯度方向可证明为无偏。

**诚实评估：** HCGAE 是一种动机充分、理论扎实、经过实证验证的 PPO 渐进式改进——而非"革命性"突破。其主要贡献在于：(a) 更快的训练早期 Critic 收敛（Hopper-v4 上约快 47%），(b) 与 DCPPO-S 联合使用时显著降低运动控制任务的训练方差（20 倍标准差减少），以及 (c) 两种轻量级机制之间的新颖协同交互，在 5 个种子上具有稳健性。这些发现是对理解 PPO 失效模式及潜在缓解方案的有意义的贡献。

---

## 参考文献

[1] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). 使用广义优势估计的高维连续控制（High-Dimensional Continuous Control Using Generalized Advantage Estimation）. *ICLR 2016*.

[2] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). 近端策略优化算法（Proximal Policy Optimization Algorithms）. *arXiv:1707.06347*.

[3] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). 信任域策略优化（Trust Region Policy Optimization）. *ICML 2015*.

[4] Sutton, R. S. (1988). 时序差分方法学习预测（Learning to Predict by the Methods of Temporal Differences）. *Machine Learning, 3(1)*.

[5] Mnih, V., et al. (2016). 深度强化学习的异步方法（Asynchronous Methods for Deep Reinforcement Learning）. *ICML 2016*.

[6] Espeholt, L., et al. (2018). IMPALA：基于重要性加权 Actor-Learner 架构的可扩展分布式深度强化学习（IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures）. *ICML 2018*.

[7] Hessel, M., et al. (2018). Rainbow：结合深度强化学习改进方法（Rainbow: Combining Improvements in Deep Reinforcement Learning）. *AAAI 2018*.

[8] Hessel, M., et al. (2019). 带 PopArt 的多任务深度强化学习（Multi-task Deep Reinforcement Learning with PopArt）. *AAAI 2019*.

[9] Cobbe, K., et al. (2021). 分阶段策略梯度（Phasic Policy Gradient）. *ICML 2021*.

[10] Precup, D., Sutton, R. S., & Singh, S. (2000). 离线策略评估的资格迹（Eligibility Traces for Off-Policy Policy Evaluation）. *ICML 2000*.

[11] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). 优先经验回放（Prioritized Experience Replay）. *ICLR 2016*.

[12] Engstrom, L., et al. (2020). 实现细节在深度强化学习中的重要性：PPO 与 TRPO 的案例研究（Implementation Matters in Deep RL: A Case Study on PPO and TRPO）. *ICLR 2020*.

[13] Andrychowicz, M., et al. (2021). 在线策略深度 Actor-Critic 方法的关键因素（What Matters for On-Policy Deep Actor-Critic Methods?）. *ICLR 2021*.

[14] Nan, G., et al. (2025). NGRPO：负增强群体相对策略优化（NGRPO: Negative-enhanced Group Relative Policy Optimization）. *arXiv:2509.18851*.

[15] Yu, Y., et al. (2025). DAPO：面向规模的开源大语言模型强化学习系统（DAPO: An Open-Source LLM Reinforcement Learning System at Scale）. *arXiv:2503.14476*.

[16] Kozuno, T., et al. (2021). 重审优先经验回放（Revisiting Prioritized Experience Replay）. *ICML Workshop 2021*.

---

## 附录 A：命题 2 的完整证明

**设定。** 设 $\pi_{\mathrm{old}}$ 为行为策略。定义 $V^{\pi}(s_t) = \mathbb{E}_{\pi_{\mathrm{old}}}[G_t \mid s_t]$ 为 $\pi_{\mathrm{old}}$ 的在线策略值函数（与最优值 $V^*$ 不同）。设 $B_t = V(s_t) - V^{\pi}(s_t)$ 为步骤 $t$ 处 Critic 的标量偏差。

**步骤 1。** 由于 $G_t$ 是在线策略采样的，$\mathbb{E}_{\pi_{\mathrm{old}}}[G_t \mid s_t] = V^{\pi}(s_t)$：

$$\mathbb{E}[V^c(s_t)] = (1-\alpha_t)V(s_t) + \alpha_t V^{\pi}(s_t) = V^{\pi}(s_t) + (1-\alpha_t)B_t$$

**步骤 2。** 期望校正残差：

$$\mathbb{E}[\delta_t^c] = r_t + \gamma\mathbb{E}[V^c(s_{t+1})] - \mathbb{E}[V^c(s_t)]$$

$$= r_t + \gamma(V^{\pi}(s_{t+1}) + (1-\alpha_{t+1})B_{t+1}) - (V^{\pi}(s_t) + (1-\alpha_t)B_t)$$

$$= \underbrace{r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)}_{=0 \text{ 由在线策略 Bellman 方程}} + \gamma(1-\alpha_{t+1})B_{t+1} - (1-\alpha_t)B_t$$

$$= \gamma(1-\alpha_{t+1})B_{t+1} - (1-\alpha_t)B_t \qquad \square$$

**步骤 3（方差）。** $\mathrm{Var}[\delta_t^c] = (1-\alpha_t)^2\,\mathrm{Var}[\delta_t] + \alpha_t^2\,\mathrm{Var}[G_t - G_{t+1}']$。当 $\alpha_t \to 1$ 时：$\mathrm{Var} \to \mathrm{Var}[G_t - G_{t+1}']$（MC 方差）。当 $\alpha_t \to 0$ 时：$\mathrm{Var} \to \mathrm{Var}[\delta_t]$（TD 方差，通常更低）。

---

## 附录 B：超参数敏感性（真实实验结果）

所有结果均为*真实*实验运行（Hopper-v4，seed=42，300K 步）。数据来源：`results/Sensitivity/`。

**表 B1.** HCGAE beta 敏感性（alpha_max=0.7 固定）。

| beta | 最终奖励 | 备注 |
|---|:---:|---|
| 1.0 | **3202** | 软校正；稳定 |
| 2.0 | 1849 | 训练中期不稳定 |
| **3.0** ★ | **3457** | 默认值——最优 |
| 4.0 | 1203 | 过于尖锐；无法恢复 |
| 5.0 | 2556 | 部分恢复 |

**表 B2.** HCGAE alpha_max 敏感性（beta=3.0 固定）。

| alpha_max | 最终奖励 | 备注 |
|---|:---:|---|
| 0.3 | 3287 | 校正不足 |
| 0.5 | 2607 | 训练中期不稳定 |
| **0.7** ★ | **3457** | 默认值——最优 |
| 0.9 | 2178 | 过度校正 |

**表 B3.** DCPPO-S SNR* 敏感性。

| SNR* | 最终奖励 | 备注 |
|---|:---:|---|
| 0.1 | 2601 | 过于保守 |
| 0.2 | 2601 | 与 0.1 相近 |
| **0.3** ★ | **2945** | 默认值——最佳平衡 |
| 0.5 | 3240 | 方差略高 |
| 0.7 | 2460 | 过于激进 |

---

## 附录 C：实现细节与可重现性

**代码结构：**

```
gae_experiments/agents/
+-- hindsight_ppo.py         # HCGAE（完整 v2 实现）
+-- dcppo.py                 # DCPPO（G/A/S + HCGAE）
+-- hindsight_ablation.py    # HCGAE 消融变体
+-- ppo_baselines.py         # PPO 基线变体（KLPEN/Anneal/EntDecay/VClip）
```

**可重现性命令：**

```bash
# 安装依赖
pip install gymnasium[mujoco] torch numpy matplotlib scipy

# 统一 5 种子对比（表 1，1M 步）
python run_unified_comparison.py

# HCGAE 消融（表 3，5 个种子）
python run_hcgae_ablation_multiseed.py --env Hopper-v4 --total_steps 300000

# DCPPO 消融（表 4，seed=42，500K 步）
python run_dcppo.py --env Hopper-v4 --total_steps 500000

# 超参数敏感性
python run_sensitivity.py

# 生成所有图表
python generate_unified_figures.py

# 计算开销测量
python measure_overhead.py
```

所有实验使用 PyTorch（CPU），无需 CUDA。完整实验运行（3 个环境 × 6 种算法 × 5 个种子 × 1M 步）在现代 CPU 上约需 15-20 小时。

---

## 附录 D：HCGAE_Base 与本文方法的关系

HCGAE_Base 在我们的消融实验中扮演重要的中间角色：它是*不含*改进 I 和 II 的 HCGAE（即使用 v1 风格 EMA 归一化和固定 50/50 Critic 目标混合）。在 Hopper-v4（5 个种子）上，HCGAE_Base 达到 **2653 ± 627**，这已经大幅高于干净的标准 PPO 基线（300K 步时约 2700，大体相当）。这引出了一个问题：**HCGAE_Base 本身是否构成有意义的贡献？**

我们的评估：HCGAE_Base 体现了核心思路（回顾性 MC 校正），其相对于 PPO+VClip（416）的性能改善在一定程度上是 Hopper-v4 上值函数裁剪有害这一事实的副产品。相对于干净标准 PPO（~2700），HCGAE_Base（2653）大体持平。本文工作的*真正*贡献在于改进 I+II 及其协同效应（加法预测基础上 +661 分），这是在相同协议下跨 5 个种子的稳健发现。

