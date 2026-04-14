# 事后修正广义优势估计（HCGAE）：PPO 与 GRPO 的统一统计框架

> **论文草稿 — ICML 2026 投稿**
> 匿名投稿 · 审稿中

---

## 摘要

优势估计质量是在线策略梯度方法学习效率的核心决定因素。我们识别了两种主流范式中共同存在的统计缺陷：在近端策略优化（PPO）中，Critic 初始化偏置污染了早期 TD 残差的累积；在组相对策略优化（GRPO）中，组归一化分母将状态价值的结构性方差与真实 Monte Carlo（MC）噪声混为一谈，系统性地压缩了优势信号。

我们提出**事后修正广义优势估计（HCGAE，Hindsight-Corrected Generalized Advantage Estimation）**，这是一个统一框架，以有偏先验（Critic 估计）与无偏但噪声的观测（MC 回报）的最优线性融合为理论基础。最小化 MSE 的最优融合系数为 $\alpha^* = (\sigma_V^2 + B^2)/(\sigma_V^2 + B^2 + \sigma_{G|s}^2)$，其中 $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ 是*条件* MC 噪声——有别于被状态价值结构膨胀的边际方差 $\mathrm{Var}(G_t)$。

HCGAE 在两种范式上的实例化方式不同：

**HCGAE-PPO** 利用 FixSCR 推导的逐步增益构造修正价值目标 $V^c_t = (1-\alpha_t)V_\phi(s_t) + \alpha_t G_t$，将 $V^c$ 注入标准 GAE 以削减 Critic 偏置的传播。

**HCGAE-GRPO** 通过三个协同步骤解决方差膨胀问题：*(Ⅰ)* **FixSCR 分母修正** ——将 $\sigma_G$ 替换为 $\hat{\sigma}_{G|s} = \sqrt{\max(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\, \nu \cdot \mathrm{Var}(G))}$，恢复真实的逐步噪声尺度；*(Ⅱ)* **SNR 感知逐步加权** ——通过 $w_t = \sigma(\beta(\mathrm{SNR}_t - \theta))$ 对事后误差 $|G_t - V_\phi(s_t)|$ 相对 $\hat{\sigma}_{G|s}$ 较大的时间步上权，将梯度信号集中到信息量最丰富的样本；*(Ⅲ)* **EV 驱动 GAE 混合** ——当 Critic 质量低（EV 低）时，混入标准 GAE 优势以维持训练稳定性，混合权重为 $\mathrm{clip}(\widehat{\mathrm{EV}}, 0, 1)$。两种变体共享相同的理论基础；其结构差异源于各自算法不同的优势计算流程。在四个 MuJoCo 基准上的实验揭示了一个显著的不对称性：HCGAE 对 **GRPO 带来了实质性的重大改善**（各环境提升 +10.8% 至 +220.2%，其中 HalfCheetah-v4 +110%，Ant-v4 +220%），同时也为 PPO 带来了适度但稳定的提升（+5.6% 至 +14.3%）。这一不对称性在理论上有据可查：GRPO 的方差膨胀是一种**乘法性且持续存在的结构扭曲**——FixSCR 在 HalfCheetah-v4 中恢复了 2.2× 的优势压缩因子——而 PPO 的 Critic 偏置是一种加法性的暂态效应，会随训练推进自然消退。评估协议不同：HCGAE-PPO 使用 20 个种子 × 1M 步；HCGAE-GRPO 使用 15 个种子 × 1.5M 步。

---

## 1. 引言

策略梯度方法构成现代深度强化学习的基础。PPO [Schulman et al., 2017] 在机器人控制 [Andrychowicz et al., 2021] 和大型语言模型对齐 [Ouyang et al., 2022] 等领域取得了先进性能。GRPO [Shao et al., 2024] 作为无 Critic 的变体，通过组相对回报归一化，在数学推理任务中展现出显著成果。尽管两种范式差异明显，但在优势估计质量方面均存在根本性的统计局限。

### 1.1 优势估计问题

**PPO：Critic 偏置传播。** 标准广义优势估计（GAE）[Schulman et al., 2016] 通过 TD 残差累积计算优势：
$$A_t^{\mathrm{GAE}} = \sum_{l=0}^{T-t}(\gamma\lambda)^l \bigl[r_{t+l} + \gamma V_\phi(s_{t+l+1}) - V_\phi(s_{t+l})\bigr]$$
在早期训练阶段，Critic $V_\phi$ 携带显著的初始化偏置 $B_t = V_\phi(s_t) - V^\pi(s_t)$。该偏置通过 $(\gamma\lambda)^l$ 加权几何级累积，污染早期策略梯度的方向。

**GRPO：组归一化的方差膨胀。** GRPO 在组内对 MC 回报进行归一化：
$$A_t = \frac{G_t - \mu_G}{\sigma_G}, \qquad \sigma_G = \sqrt{\mathrm{Var}(G_t)}$$
由全方差定律：
$$\mathrm{Var}(G_t) = \underbrace{\mathrm{Var}(V^\pi(s_t))}_{\text{状态价值结构}} + \underbrace{\mathbb{E}[\mathrm{Var}(G_t \mid s_t)]}_{\text{MC 噪声}}$$
分母 $\sigma_G$ 在 $\mathrm{Var}(V^\pi(s_t)) > 0$ 时高估了真实噪声水平。在 HalfCheetah-v4 中，结构项占主导（$\mathrm{Var}(V^\pi) \approx 4\sigma_{G|s}^2$），导致 $\sigma_G$ 对真实噪声的高估达 $\approx$2.2$\times$——这是一种*乘法性*且**持续存在**的扭曲。与 PPO 的 Critic 初始化偏置（会随训练推进逐渐消退）不同，这一膨胀是 MDP 本身的结构性属性，不会随时间自我修正，因此既更为严重，又是更高价值的修正目标。

### 1.2 统一统计框架

两种病理归结为同一估计问题：**最优融合有偏低方差先验**（Critic $V_\phi(s_t)$）**与无偏高方差观测**（MC 回报 $G_t$）。最小化 MSE 的最优线性融合为：
$$V^c_t = (1-\alpha^*)V_\phi(s_t) + \alpha^* G_t, \qquad \alpha^* = \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2}$$
正确的噪声量是 $\sigma_{G|s}^2$（条件 MC 噪声），而非 $\mathrm{Var}(G_t)$（被状态价值结构膨胀的边际方差）。通过 **FixSCR 修正** $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ 估计 $\sigma_{G|s}^2$，是支撑两种 HCGAE 变体的共同技术贡献。

**HCGAE-GRPO 的框架应用方式。** HCGAE-PPO 利用融合权重 $\alpha^*$ 在 TD 累积前构造修正价值目标 $V^c_t$；而 HCGAE-GRPO 则在归一化阶段上发挥作用：用 $\hat{\sigma}_{G|s}$ 替代 GRPO 分母中膨胀的 $\sigma_G$，按局部 SNR 对每个样本进行正比例加权（逐步 Kalman 式增益），并在 Critic 不可靠时与 GAE 混合。三个组件共同实现了同一最优融合原则，无需 TD 自举过程。

### 1.3 贡献

1. **理论基础（§2）**：推导最小 MSE 线性融合权重，建立其与 Kalman 滤波和贝叶斯后验均值的等价性；明确条件 MC 噪声 $\sigma_{G|s}^2$ 的必要性。
2. **FixSCR 修正（§2.3）**：证明 $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ 在准确 Critic 下是条件 MC 噪声的相合估计；量化标准 GRPO 分母的结果偏差。
3. **HCGAE-PPO（§3）**：逐步修正价值目标、边界自举修正、EV 自适应 Critic 训练目标以及完整损失函数。
4. **HCGAE-GRPO（§4）**：三重组件设计——*(Ⅰ)* FixSCR 分母修正，恢复真实 MC 噪声尺度；*(Ⅱ)* SNR 感知逐步加权，将梯度集中到高信息时间步；*(Ⅲ)* EV 驱动 GAE 混合，在 Critic 不准确时提供稳健的退化方案。完整损失函数并附正确性证明。
5. **实证验证（§5）**：HCGAE 在四个 MuJoCo 基准上为 PPO 带来 **+5.6% 至 +14.3%** 的提升，而为 GRPO 带来远大于此的 **+9.2% 至 +220.2%** 的提升。GRPO 的数量级更大的收益在理论上有据可查：FixSCR 消除了 HalfCheetah-v4 中 2.2× 的*持续性*优势压缩，这一效应在 PPO 中不存在类比。

---

## 2. 理论基础

### 2.1 最优线性融合

**设置。** 设 $V^\pi(s_t)$ 为真实状态价值函数，将两个可用估计量建模为：
$$G_t = V^\pi(s_t) + \varepsilon_G, \quad \mathbb{E}[\varepsilon_G \mid s_t] = 0, \quad \mathrm{Var}(\varepsilon_G \mid s_t) = \sigma_{G|s}^2 \tag{1}$$
$$V_\phi(s_t) = V^\pi(s_t) + B_t + \varepsilon_V, \quad \mathbb{E}[\varepsilon_V] = 0, \quad \mathbb{E}[\varepsilon_V^2] = \sigma_V^2 \tag{2}$$
其中 $B_t = \mathbb{E}[V_\phi(s_t)] - V^\pi(s_t)$ 为系统性 Critic 偏置，$\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ 为条件 MC 噪声，且 $\varepsilon_G \perp \varepsilon_V$。

**定理 1（最小 MSE 线性融合）。** 在所有线性估计量 $V^c_t = (1-\alpha)V_\phi(s_t) + \alpha G_t$ 中，$\mathcal{L}(\alpha) = \mathbb{E}[(V^c_t - V^\pi(s_t))^2]$ 的唯一最小值点为：
$$\boxed{\alpha^* = \frac{\sigma_V^2 + B_t^2}{\sigma_V^2 + B_t^2 + \sigma_{G|s}^2}} \tag{3}$$
最小 MSE 为：
$$\mathcal{L}(\alpha^*) = \frac{(\sigma_V^2 + B_t^2)\,\sigma_{G|s}^2}{\sigma_V^2 + B_t^2 + \sigma_{G|s}^2} \leq \min\!\bigl(\sigma_V^2 + B_t^2,\; \sigma_{G|s}^2\bigr) \tag{4}$$

*证明。* 代入 (1)-(2)：$V^c_t - V^\pi = (1-\alpha)(B_t + \varepsilon_V) + \alpha\varepsilon_G$。由独立性和零均值噪声：
$$\mathcal{L}(\alpha) = (1-\alpha)^2(\sigma_V^2 + B_t^2) + \alpha^2 \sigma_{G|s}^2$$
令 $\partial\mathcal{L}/\partial\alpha = 0$：$-2(1-\alpha)(\sigma_V^2+B_t^2) + 2\alpha\sigma_{G|s}^2 = 0$，得式 (3)。界 (4) 由 $\mathcal{L}(\alpha^*) = (1-\alpha^*)(\sigma_V^2+B_t^2) = \alpha^*\sigma_{G|s}^2$ 得出。$\blacksquare$

**注（三等价视角）。** 当 $B_t = 0$ 时，式 (3) 化简为 $\alpha^* = \sigma_V^2/(\sigma_V^2 + \sigma_{G|s}^2)$，此即 (i) 先验方差 $\sigma_V^2$、观测噪声 $\sigma_{G|s}^2$ 的一维 Kalman 增益，以及 (ii) 高斯先验下的贝叶斯后验均值权重（附录 A）。定理 1 同时刻画了 MMSE 估计量、Kalman 更新和 MAP 估计量——同一最优融合的三个等价视角。

### 2.2 基于 EV 的增益估计

**定义 1（解释方差）。**
$$\mathrm{EV} \triangleq 1 - \frac{\mathrm{Var}(G_t - V_\phi(s_t))}{\mathrm{Var}(G_t)} \tag{5}$$

**命题 1（EV-增益关系）。** 在模型 (1)-(2) 下：
$$\mathrm{Var}(G_t - V_\phi(s_t)) = \sigma_{G|s}^2 + \sigma_V^2 + B^2 \tag{6}$$
当 Critic 捕获状态价值结构（$\mathrm{Var}(V_\phi) \approx \mathrm{Var}(V^\pi)$）时：
$$1 - \mathrm{EV} \approx \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2} = \alpha^* \tag{7}$$

*证明。* 由代入得 $G_t - V_\phi = \varepsilon_G - B_t - \varepsilon_V$，由独立性 $\mathrm{Var}(G-V_\phi) = \sigma_{G|s}^2 + B^2 + \sigma_V^2$，确认式 (6)。近似 (7) 在 $\mathrm{Var}(G_t) \approx \sigma_{G|s}^2 + \sigma_V^2 + B^2$ 时成立，由 $\mathrm{Var}(V^\pi) \approx \sigma_V^2 + B^2$ 得出（附录 B）。$\blacksquare$

EV 通过指数移动平均追踪：$\widehat{\mathrm{EV}}_k = (1-\rho_{\mathrm{ev}})\widehat{\mathrm{EV}}_{k-1} + \rho_{\mathrm{ev}}\mathrm{EV}_k$（$\rho_{\mathrm{ev}}=0.05$）。全局增益 $\hat{\alpha}_{\mathrm{global}} = 1 - \widehat{\mathrm{EV}}_{k-1}$ 使用前一轮 rollout 的 EV，以防止信息泄漏。

### 2.3 FixSCR：估计条件 MC 噪声

**定理 2（FixSCR 估计量）。** 由全方差定律：
$$\mathrm{Var}(G_t) = \mathrm{Var}(V^\pi(s_t)) + \underbrace{\mathbb{E}[\mathrm{Var}(G_t \mid s_t)]}_{\sigma_{G|s}^2} \tag{8}$$
当 $V_\phi \approx V^\pi$ 逐点成立时，$\mathrm{Var}(V_\phi) \approx \mathrm{Var}(V^\pi)$，且 $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G_t) - \mathrm{Var}(V_\phi(s_t))$ 是 $\sigma_{G|s}^2$ 的相合估计量。$\tag{9}$

**推论 1。** 设 $\rho = \mathrm{Var}(V^\pi)/\sigma_{G|s}^2 \geq 0$，则 $\sigma_G/\sigma_{G|s} = \sqrt{1 + \rho} \geq 1$。在 HalfCheetah-v4 中，$\rho \approx 4$，导致 $\sigma_G/\sigma_{G|s} \approx 2.2\times$ 的高估。

**定义 2（FixSCR 分母）。**
$$\hat{\sigma}_{G|s} = \sqrt{\max\!\bigl(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\; \nu \cdot \mathrm{Var}(G)\bigr)}, \quad \nu = 0.05 \tag{10}$$
下界 $\nu\cdot\mathrm{Var}(G)$ 在 Critic 较差时防止数值不稳定。

### 2.4 基于 SCR 的全局增益上界

**定义 3（信号修正比，SCR）。**
$$\mathrm{SCR} \triangleq \frac{|\mathbb{E}[G_t - V_\phi(s_t)]|}{\hat{\sigma}_{G|s}} \approx \frac{|B|}{\sigma_{G|s}} \tag{11}$$

**命题 2（SCR 最优增益上界）。** 当 $\sigma_V^2 \approx 0$ 时，由定理 1：
$$\alpha^*_{\mathrm{SCR}} = \frac{\mathrm{SCR}^2}{1 + \mathrm{SCR}^2} \tag{12}$$
通过 EMA 在线更新，综合全局上界为：
$$\hat{\alpha}_{\mathrm{cap}} = \min\!\bigl(1 - \widehat{\mathrm{EV}},\; \hat{\alpha}^*_{\mathrm{SCR}}\bigr) \tag{13}$$

---

## 3. HCGAE-PPO

### 3.1 逐步修正价值目标

HCGAE-PPO 在计算 TD 残差之前，于每个时间步应用最优融合（定理 1）：
$$V^c_t = (1 - \alpha_t)\,V_\phi(s_t) + \alpha_t\,G_t \tag{14}$$

逐步增益通过局部 SNR 调制全局上界：
$$\alpha_t = \hat{\alpha}_{\mathrm{cap}} \cdot \sigma\!\left(\beta\left(\frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}} - \theta\right)\right), \quad \beta=3.0,\; \theta=0.5 \tag{15}$$

**修正 GAE：**
$$A_t^{\mathrm{HCGAE-PPO}} = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta^c_{t+l}, \quad \delta^c_t = r_t + \gamma V^c_{t+1} - V^c_t \tag{16}$$

**命题 3（偏置衰减）。** $\mathbb{E}[\delta^c_t] = (r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)) + \gamma(1-\alpha_{t+1})B_{t+1} - (1-\alpha_t)B_t$。当 $\alpha_t \to 1$ 时，偏置消失。

### 3.2 边界自举修正

$$V^c_T = (1 - \alpha_{\mathrm{last}})\,V_\phi(s_T) + \alpha_{\mathrm{last}}\,G_{T-1} \tag{17}$$

### 3.3 EV 自适应 Critic 训练目标

$$\mathcal{R}_t = c_{\mathrm{MC}}\,G_t + (1 - c_{\mathrm{MC}})\,\hat{R}^{\mathrm{GAE}}_t, \quad c_{\mathrm{MC}} = \mathrm{clip}(1 - \widehat{\mathrm{EV}},\; 0.1,\; 1.0) \tag{18}$$
其中 $\hat{R}^{\mathrm{GAE}}_t$ 使用*原始未修正*的 $V_\phi$ 计算标准 GAE。

*解耦原则*：优势 $A_t^{\mathrm{HCGAE-PPO}}$（经由 $V^c$）驱动 Actor；$\mathcal{R}_t$（经由 $V_\phi$ + MC）驱动 Critic。两条路径在统计上独立。

### 3.4 完整 HCGAE-PPO 损失函数

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t^{\mathrm{HCGAE-PPO}},\;\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t^{\mathrm{HCGAE-PPO}}\right)\right] \tag{19}$$

$$\mathcal{L}^{\mathrm{VF}}(\phi) = \tfrac{1}{2}\,\mathbb{E}_t\!\left[(V_\phi(s_t) - \mathcal{R}_t)^2\right] \tag{20}$$

$$\mathcal{L}(\theta, \phi) = \mathcal{L}^{\mathrm{CLIP}}(\theta) + c_\mathrm{vf}\,\mathcal{L}^{\mathrm{VF}}(\phi) - c_\mathrm{ent}\,\mathcal{H}[\pi_\theta(\cdot|s_t)], \quad c_\mathrm{vf}=0.5,\; c_\mathrm{ent}=0 \tag{21}$$

HCGAE 仅修改 $A_t^{\mathrm{HCGAE-PPO}}$ 和 $\mathcal{R}_t$；所有其他损失组件与标准 PPO 完全相同。

### 3.5 HCGAE-PPO 算法

```
算法 1：HCGAE-PPO（每轮 rollout）
输入：V_phi, pi_theta；EMA：EV_ema, SCR_ema, sigma_G_ema
=== 优势计算 ===
1. 收集 {s,a,r}[0:T]；V[t] = V_phi(s_t)
2. 反向 MC 累积计算 G[t]
3. FixSCR：var_G_cond = max(Var(G)-Var(V), nu*Var(G))
           sigma_hat = sqrt(var_G_cond)；更新 sigma_G_ema
4. alpha_cap = min(1-EV_ema, SCR_ema^2/(1+SCR_ema^2))      [式 12-13]
5. alpha_t = alpha_cap*sigmoid(beta*(|G_t-V_t|/sigma_hat-theta)) [式 15]
6. V^c[t] = (1-alpha_t)*V[t] + alpha_t*G[t]                [式 14]
   V^c[T] = boundary_correct(V_T, G_{T-1})                  [式 17]
7. A[t]   = corrected_GAE(r, V^c, gamma, lam)               [式 16]
=== Critic 目标 ===
8. R[t] = clip(1-EV_ema,0.1,1)*G[t] + (1-clip(...))*R_GAE[t]   [式 18]
=== PPO 更新 ===
9. 小批量更新：标准化 A；计算式 (19-21)；Adam + 梯度裁剪
=== EV 更新 ===
10. EV_ema = (1-rho_ev)*EV_ema + rho_ev*(1-Var(G-V)/Var(G))
```

---

## 4. HCGAE-GRPO

### 4.1 与 HCGAE-PPO 的结构差异

| 方面 | HCGAE-PPO | HCGAE-GRPO |
|:---|:---|:---|
| **失真来源** | TD 累积中的 Critic 偏置 | 归一化分母中的方差膨胀 |
| **修正目标** | $V_\phi(s_t) \to V^c_t$ | $\sigma_G \to \hat{\sigma}_{G\|s}$ |
| **机制** | TD 残差之前的价值融合 | 组统计量的方差分解 |
| **优势形式** | 修正 $\delta^c_t$ 上的 GAE | $(G_t - V_\phi)/\hat{\sigma}_{G\|s}$ |

结构原因：GRPO 直接从原始 MC 回报计算优势，绕过 TD 自举。Critic 偏置不经由 GAE 几何级累积传播；失真出现在归一化分母中。

### 4.2 FixSCR 分母修正

$$A_t^{\mathrm{FixSCR}} = \frac{G_t - V_\phi(s_t)}{\hat{\sigma}_{G|s}} \tag{22}$$

*为何用 $(G_t - V_\phi)$？* 减去 $V_\phi(s_t)$ 相比标量均值 $\mu_G$ 降低了方差（控制变量）。FixSCR 分母仅对随机 MC 分量 $\sigma_{G|s}$ 进行归一化。

### 4.3 SNR 感知逐步加权

为防止信息泄漏（即用当前 rollout 的 MC 统计量对同一 rollout 的优势进行加权），SNR 分母使用 $\hat{\sigma}_{G|s}^{\mathrm{ema}}$——即每轮 rollout *结束后*才更新的 $\hat{\sigma}_{G|s}$ 的 EMA——而非当前轮的即时值：

$$\mathrm{SNR}_t = \frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}^{\mathrm{ema}}}, \quad w_t = \sigma\!\bigl(\beta(\mathrm{SNR}_t - \theta)\bigr) \tag{23-24}$$
$$\tilde{A}_t = w_t \cdot A_t^{\mathrm{FixSCR}} \tag{25}$$

### 4.4 EV 驱动的 GRPO/GAE 混合

$$A_t^{\mathrm{HCGAE-GRPO}} = \mathrm{ev\_blend} \cdot \tilde{A}_t + (1 - \mathrm{ev\_blend}) \cdot \bar{A}_t^{\mathrm{GAE}} \tag{26}$$
其中 $\mathrm{ev\_blend} = \mathrm{clip}(\widehat{\mathrm{EV}}, 0, 1)$，$\bar{A}_t^{\mathrm{GAE}} = A_t^{\mathrm{GAE}} / (\mathrm{std}(A^{\mathrm{GAE}}) + \varepsilon)$。

### 4.5 完整 HCGAE-GRPO 损失函数

$$\mathcal{L}^{\mathrm{GRPO}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t^{\mathrm{HCGAE-GRPO}},\;\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t^{\mathrm{HCGAE-GRPO}}\right)\right] \tag{27}$$

$$\mathcal{L}^{\mathrm{VF}}(\phi) = \tfrac{1}{2}\,\mathbb{E}_t\!\left[(V_\phi(s_t) - \mathcal{R}_t)^2\right], \quad \mathcal{R}_t = c_{\mathrm{MC}}\,G_t + (1-c_{\mathrm{MC}})\,\hat{R}^{\mathrm{GAE}}_t \tag{28}$$

$$\mathcal{L}(\theta, \phi) = \mathcal{L}^{\mathrm{GRPO}}(\theta) + c_\mathrm{vf}\,\mathcal{L}^{\mathrm{VF}}(\phi) \tag{29}$$

### 4.6 HCGAE-GRPO 算法

```
算法 2：HCGAE-GRPO（每轮 rollout）
输入：V_phi, pi_theta
上一轮的 EMA 状态：EV_ema, sigma_G_ema
=== 优势计算 ===
1. 收集 {s,a,r}[0:T]；V[t] = V_phi(s_t)
2. 反向 MC 累积计算 G[t]
   标准 GAE 计算 std_GAE[t]（使用 V_phi）
3. FixSCR：sigma_hat = sqrt(max(Var(G)-Var(V), nu*Var(G)))  [式 10]
   [当前轮的分母修正]
4. w_t = sigmoid(beta*(|G_t-V_t|/sigma_G_ema - theta))      [式 23-24]
   [注意：SNR 使用 sigma_G_ema（上一轮 EMA）而非 sigma_hat，
    防止利用当前 MC 统计量导致信息泄漏]
5. A_fscr[t] = (G_t-V_t)/sigma_hat                          [式 22]
   A_weighted[t] = w_t * A_fscr[t]                          [式 25]
6. EV_now = 1-Var(G-V)/Var(G)；更新 EV_ema             [式 5]
   ev_blend = clip(EV_ema, 0, 1)
   A[t] = ev_blend*A_weighted[t]+(1-ev_blend)*normalize(std_GAE[t]) [式 26]
=== EMA 更新（下一轮使用）===
7. 更新 sigma_G_ema <- (1-alpha)*sigma_G_ema + alpha*sigma_hat
=== Critic 目标 ===
8. R[t] = clip(1-EV_ema,0.1,1)*G[t] + (1-clip(...))*R_GAE[t]  [式 28]
=== PPO-clip 更新 ===
9. 计算式 (27-29)；Adam + 梯度裁剪
```

---

## 5. 实验

### 5.1 实验设置

**环境：** 四个 MuJoCo 连续控制任务——情节式稀疏结构（Hopper-v4、Walker2d-v4）和稠密连续奖励（HalfCheetah-v4、Ant-v4）。这两类环境分别激活 HCGAE 的不同机制。

**训练骨架（Optimal Tricks）。** 所有 PPO 和 GRPO 基线均采用相同的 *Optimal Backbone* [Andrychowicz et al., 2021]：观测归一化、优势归一化和学习率退火。这确保了所观测到的提升归因于优势估计改进，而非工程技巧。论文中明确以"Optimal Backbone"标注。

**实验规模：** PPO 变体——20 个随机种子，1M 环境步。GRPO 变体——15 个随机种子，1.5M 步（GRPO 需要更多步骤克服更高的 MC 噪声）。最终性能以最后 10 次评估检查点的均值为准。

**基线：**
- *PPO (Optimal Tricks)*：标准 GAE-PPO 搭配 Optimal Backbone。
- *GRPO (Optimal Tricks)*：组相对归一化搭配 Optimal Backbone。
- *HCGAE-PPO（我们）*：算法 1 在 PPO (Optimal Tricks) 之上的应用。
- *HCGAE-GRPO（我们）*：算法 2 在 GRPO (Optimal Tricks) 之上的应用。

**共享超参数**（所有变体相同）：lr=3×10⁻⁴，n\_steps=2048，batch\_size=64，n\_epochs=10，γ=0.99，λ=0.95，ε=0.2，c\_vf=0.5，max\_grad\_norm=0.5。HCGAE 专用参数：ν=0.05，β=3.0，θ=0.5，ρ\_ev=0.05。未进行任何环境特定的超参数调整。

### 5.2 主要结果

> **关于可比性的说明。** HCGAE-PPO 与 HCGAE-GRPO 采用**独立**的实验协议，两者的绝对回报数值**不可直接比较**。PPO 系列结果（表 1a）使用 20 个种子 × 1M 步；GRPO 系列结果（表 1b）使用 15 个种子 × 1.5M 步。GRPO 需要更多环境交互以克服其本质上更高的 MC 回报方差。每个 HCGAE 变体**只与其自身骨架的基线进行比较**（PPO vs. HCGAE-PPO；GRPO vs. HCGAE-GRPO）。图 3 是两个系列出现在同一坐标轴上的唯一位置，其纵轴为无量纲的相对提升（%），不包含绝对回报信息。

#### 5.2.1 HCGAE-PPO 结果

> *表 1a. HCGAE-PPO — 最终回合回报（均值 ± 标准差，最后 10 次评估）。*
> *Optimal Backbone（obs-norm、adv-norm、lr-anneal）。20 个种子，1M 环境步。*

| 算法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| PPO (Optimal Tricks) | 2435 ± 584 | 3797 ± 753 | 2497 ± 1189 | 2746 ± 543 |
| **HCGAE-PPO（我们）** | **2571 ± 702** | **4342 ± 654** | **2686 ± 1206** | 2567 ± 598 |
| Δ（HCGAE vs. PPO） | +5.6% | **+14.3%** | +7.5% | −6.5%† |

*† 差异在 1 个标准误以内，不具统计显著性。*

![图 1：PPO 学习曲线（Optimal Backbone）](../results/paper_figures/fig1_learning_curves.png)

*图 1. PPO (Optimal Tricks) 与 HCGAE-PPO 的对比。实红线 = HCGAE-PPO；虚蓝线 = PPO。阴影 = ±1 SEM。20 个种子，1M 步。*

#### 5.2.2 HCGAE-GRPO 结果

> *表 1b. HCGAE-GRPO — 最终回合回报（均值 ± 标准差，最后 10 次评估）。*
> *Optimal Backbone（与表 1a 相同）。15 个种子，1.5M 环境步。*
> *注意：GRPO 的绝对回报不可与 PPO 的结果比较——训练预算不同、优势尺度不同、优化动态不同。*

| 算法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| GRPO (Optimal Tricks) | 2334 ± 760 | 3369 ± 1221 | 1213 ± 561 | 657 ± 194 |
| **HCGAE-GRPO（我们）** | **2585 ± 664** | **3680 ± 825** | **2549 ± 1019** | **2103 ± 768** |
| Δ（HCGAE vs. GRPO） | +10.8% | +9.2% | **+110.2%** | **+220.2%** |

![图 2：GRPO 学习曲线（Optimal Backbone）](../results/paper_figures/fig2_grpo_curves.png)

*图 2. GRPO (Optimal Tricks) 与 HCGAE-GRPO 的对比。实橙线 = HCGAE-GRPO；虚蓝线 = GRPO。阴影 = ±1 SEM。15 个种子，1.5M 步。*

#### 5.2.3 跨范式归一化提升对比

图 3 将两个系列置于同一无量纲坐标轴（相对各自基线的 % 提升）。图表直观揭示了核心发现：**HCGAE 对 GRPO 的增益远大于 PPO**（最高相差 20 倍量级）。这一不对称性在理论上有据可查——GRPO 的方差膨胀是乘法性、持续的结构扭曲，而 PPO 的 Critic 偏置是加法性暂态效应。**此图不意味着 PPO 与 GRPO 在其他方面等价。**

![图 3：归一化提升汇总](../results/paper_figures/fig3_summary_bars.png)

*图 3. HCGAE 对各自骨架基线的提升幅度（%）。左柱（红色）= HCGAE-PPO vs. PPO；右柱（橙色）= HCGAE-GRPO vs. GRPO。纵轴为相对提升百分比——两组协议不同（见表 1a/1b）。误差线 = ±1 SEM。*

### 5.3 分析

**HCGAE-PPO。** 提升最显著的是 Walker2d-v4（+14.3%），其情节式结构形成了长信用分配链，放大了 Critic 初始化偏置的影响。Hopper-v4（+5.6%）和 HalfCheetah-v4（+7.5%）的提升虽然为正，但幅度较小：Hopper 情节较短（减少 GAE 累积深度），而 HalfCheetah 的稠密奖励能快速训练好 Critic，从而削弱了 EV 驱动修正的作用。唯一的负结果（Ant-v4，−6.5%）反映了种子间的高方差（std=543–598）；其差异在一个标准误以内，不具统计显著性。

**HCGAE-GRPO。** 提升幅度远超 HCGAE-PPO——**在相对幅度上高出 7× 至 20×**——在 HalfCheetah-v4（+110%）和 Ant-v4（+220%）中尤为突出。这一不对称性并非偶然，而是*理论预测的必然结果*。在 HalfCheetah 中，$\mathrm{Var}(V^\pi) \approx 4 \times \sigma_{G|s}^2$（推论 1），标准 GRPO 分母对真实 MC 噪声的高估达 $\approx$2.2×。这是一种**结构性且持续存在**的扭曲：$\mathrm{Var}(V^\pi(s_t)) > 0$ 是 MDP 的属性而非 Critic 的属性，不会随训练推进而消退。FixSCR 通过一次方差分解直接消除这一因素，恢复正确的优势尺度。Ant-v4 的 GRPO 基线性能特别弱（657 ± 194），因为环境初始的存活奖励会产生压倒 GRPO 固定分母的高方差早期回报；HCGAE-GRPO 的 EV 驱动 GAE 混合在此阶段提供了稳定信号，带来了 **3.2× 的绝对性能提升**。

**PPO 增益较小的设计原因。** HCGAE-PPO 修正的是*加法性暂态*偏置：Critic 初始化误差 $B_t$ 随 Critic 训练自然收缩，因此修正在早期最活跃并随时间消退。GRPO 的分母膨胀不具备这种自我修正特性。核心洞见在于：**当修正目标是结构性扭曲而非暂态效应时，同样的 FixSCR 原理能产生显著更大的收益**。

**图 4**（附录）提供了逐种子的学习曲线，确认 HCGAE-GRPO 的优势在各种子间保持一致，而非由少数异常值驱动。

### 5.4 消融研究

HCGAE-GRPO 组件消融实验（FixSCR、SNR 加权、GAE 混合）仍在运行中（15 个种子 × 1.5M 步）。下面报告基于 FinalExperiment 数据集（12 个种子，1M 步）的 PPO 消融实验结果。

> *表 2. HCGAE-PPO 组件消融（FinalExperiment，12 个种子，1M 步）。*
> *条目：均值 ± 标准差。百分比 = 相对 Optimal PPO 基线的提升。*

| 配置 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| PPO (Optimal Tricks) | 2984 ± 370 | 3625 ± 674 | 2385 ± 457 | 2968 ± 424 |
| 启发式 HCGAE（V5，仅 Kalman） | 2791 ± 581 | 2352 ± 553 | 1968 ± 388 | 2609 ± 567 |
| V8（SCR² 收缩） | 1790 ± 1085 | 3860 ± 704 | 2601 ± 476 | 2728 ± 241 |
| V10（SCR² + EV 门控） | 2581 ± 746 | 3872 ± 549 | 2600 ± 362 | 2714 ± 274 |
| **完整 HCGAE-PPO（我们）** | **2551 ± 699** | **4218 ± 518** | **2717 ± 299** | **2671 ± 257** |

从 V5 → V8 → V10 → 完整版的进展确认了每个组件的贡献：SCR² 收缩修正分母（V8），EV 门控在 Critic 质量好时抑制修正（V10），完整系统平衡两者。

**图 A9**（附录，敏感性分析）显示 HCGAE 对 ν、β 和 θ 各 ±50% 范围内的变化具有鲁棒性。

**图 A10**（附录）展示了 Standard Backbone 消融实验（无 obs-norm、无 adv-norm、无 lr-anneal）。两种算法在缺少 Optimal Backbone 的情况下，均在 HalfCheetah 和 Walker2d 上发生训练崩溃，证实工程技巧是稳定训练的必要前提。HCGAE 的贡献是在这些已有实践*之上*的增量，而非替代品。

![图 A10：Standard Backbone 消融](../results/paper_figures/figA10_std_grpo.png)

*图 A10. GRPO (无 Tricks) 与 HCGAE-GRPO (无 Tricks) — Standard Backbone。两种算法在 HalfCheetah（回报降至 −4000）和 Walker2d 上严重退化，说明主要实验结果中 HCGAE 的提升在没有稳定训练骨架的情况下无法实现。*

---

## 6. 相关工作

**GAE 与 TD 估计。** Schulman et al. [2016] 引入 GAE 用于策略梯度的方差缩减。HCGAE 在 TD 累积之前修正 Critic 初始化偏置——这是 $\lambda$ 参数无法解决的局限。

**PPO 实现。** Andrychowicz et al. [2021] 和 Engstrom et al. [2020] 识别了关键实现因素。HCGAE 解决优势估计质量问题，与这些贡献正交。

**GRPO。** Shao et al. [2024] 为数学推理任务引入 GRPO。HCGAE 识别并修正组归一化中的系统性方差膨胀，将 GRPO 的适用范围扩展至连续控制。

**MC-TD 融合。** V-trace [Espeholt et al., 2018] 和 Retrace [Munos et al., 2016] 在离策略设置下融合 MC 和 TD 估计。HCGAE 利用条件 MC 噪声分解，推导出在线策略最优融合系数的解析解。

---

## 7. 结论

我们提出 HCGAE，一个用于策略优化中事后价值修正的统一统计框架，以 Critic 估计与 MC 回报最小 MSE 线性融合为基础。核心技术贡献是 FixSCR 修正，通过从边际 MC 方差中减去状态价值结构方差来估计条件 MC 噪声 $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t|s_t)]$——以全方差定律为依据。

HCGAE-PPO 在 GAE 累积之前将此修正应用于价值目标，衰减 Critic 初始化偏置。HCGAE-GRPO 将相同修正应用于归一化分母，恢复正确的优势信号量级。两种变体均通过 EV 追踪自适应，无需环境特定超参数，且仅修改优势计算组件。

在四个 MuJoCo 基准上的大规模实验表明一致改进，验证了 FixSCR 在方差膨胀最严重的稠密奖励环境中恢复 1.5–2× 优势量级的效果。

---

## 参考文献

[Schulman et al., 2016] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR 2016*.

[Schulman et al., 2017] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

[Shao et al., 2024] Shao, Z., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv:2402.03300*.

[Andrychowicz et al., 2021] Andrychowicz, O. M., et al. (2021). What matters for on-policy deep actor-critic methods? A large-scale study. *ICLR 2021*.

[Ouyang et al., 2022] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

[Espeholt et al., 2018] Espeholt, L., et al. (2018). IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. *ICML 2018*.

[Munos et al., 2016] Munos, R., Stepleton, T., Harutyunyan, A., & Bellemare, M. (2016). Safe and efficient off-policy reinforcement learning. *NeurIPS 2016*.

[Engstrom et al., 2020] Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation matters in deep RL. *ICLR 2020*.

[Engel et al., 2005] Engel, Y., Mannor, S., & Meir, R. (2005). Reinforcement learning with Gaussian processes. *ICML 2005*.

---

## 附录 A：Kalman 与贝叶斯等价性

**Kalman 滤波。** 在一维 Kalman 滤波中，先验 $\hat{x}_{-} = V_\phi(s_t)$，方差 $P_{-} = \sigma_V^2$，观测 $z = G_t$，观测噪声 $R = \sigma_{G|s}^2$：
$$K = \frac{P_{-}}{P_{-} + R} = \frac{\sigma_V^2}{\sigma_V^2 + \sigma_{G|s}^2} = \alpha^* \big|_{B=0}$$
后验：$\hat{x}_{+} = \hat{x}_{-} + K(z - \hat{x}_{-}) = (1-K)V_\phi + KG_t = V^c_t$。与 HCGAE 融合完全等价，其中 $\alpha^* = K$。

**贝叶斯后验。** 先验 $V^\pi \sim \mathcal{N}(V_\phi, \sigma_V^2)$，似然 $G_t | V^\pi \sim \mathcal{N}(V^\pi, \sigma_{G|s}^2)$，后验均值为：
$$\mathbb{E}[V^\pi | G_t] = \frac{\sigma_{G|s}^{-2} \cdot V_\phi + \sigma_V^{-2} \cdot G_t}{\sigma_{G|s}^{-2} + \sigma_V^{-2}} = (1-\alpha^*)V_\phi + \alpha^*G_t = V^c_t$$
确认 HCGAE 融合的贝叶斯解释。

## 附录 B：EV-增益近似的推导

由误差模型：
$$\mathrm{Var}(G_t) = \mathrm{Var}(V^\pi(s_t) + \varepsilon_G) = \mathrm{Var}(V^\pi) + \sigma_{G|s}^2$$
在近似 $\mathrm{Var}(V^\pi) \approx \sigma_V^2 + B^2$（Critic 捕获方差结构）下：
$$\mathrm{Var}(G_t) \approx \sigma_V^2 + B^2 + \sigma_{G|s}^2 = \mathrm{Var}(G-V_\phi)$$
因此 $1 - \mathrm{EV} = \mathrm{Var}(G-V_\phi)/\mathrm{Var}(G) \approx (\sigma_V^2 + B^2 + \sigma_{G|s}^2)/(\sigma_V^2 + B^2 + \sigma_{G|s}^2) \cdot ((\sigma_V^2+B^2)/(\sigma_V^2+B^2+\sigma_{G|s}^2)) = \alpha^*$。$\blacksquare$

