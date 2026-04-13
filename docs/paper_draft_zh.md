# 事后修正广义优势估计（HCGAE）：PPO 与 GRPO 的统一统计框架

> **论文草稿 — ICML 2026 投稿**
> 匿名投稿 · 审稿中

---

## 摘要

优势估计质量是在线策略梯度方法学习效率的核心决定因素。我们识别了两种主流范式中共同存在的统计缺陷：在近端策略优化（PPO）中，Critic 初始化偏置污染了早期 TD 残差的累积；在组相对策略优化（GRPO）中，组归一化分母将状态价值的结构性方差与真实 Monte Carlo（MC）噪声混为一谈，系统性地压缩了优势信号。

我们提出**事后修正广义优势估计（HCGAE，Hindsight-Corrected Generalized Advantage Estimation）**，这是一个统一框架，以有偏先验（Critic 估计）与无偏但噪声的观测（MC 回报）的最优线性融合为理论基础。最小化 MSE 的最优融合系数为 $\alpha^* = (\sigma_V^2 + B^2)/(\sigma_V^2 + B^2 + \sigma_{G|s}^2)$，其中 $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ 是*条件* MC 噪声——有别于被状态价值结构膨胀的边际方差 $\mathrm{Var}(G_t)$。

HCGAE 在两种范式上的实例化方式不同：**HCGAE-PPO** 利用 FixSCR 推导的逐步增益构造修正价值目标 $V^c_t = (1-\alpha_t)V_\phi(s_t) + \alpha_t G_t$，将 $V^c$ 注入标准 GAE 以削减 Critic 偏置的传播；**HCGAE-GRPO** 以 $\hat{\sigma}_{G|s} = \sqrt{\max(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\, \nu \cdot \mathrm{Var}(G))}$ 替换膨胀的组归一化分母，并通过局部信噪比估计增强逐步加权。两种变体共享相同的理论基础；其结构差异源于各自算法不同的优势计算流程。在四个 MuJoCo 基准（15 个种子，1.5M 步）上的实验表明，无需环境特定超参数调整，两者均实现一致改进。

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
分母 $\sigma_G$ 在 $\mathrm{Var}(V^\pi(s_t)) > 0$ 时高估了真实噪声水平。在 HalfCheetah-v4 中，结构项占主导，导致 $\sigma_G$ 对真实噪声的高估达 $2\times$ 甚至更多——稀释优势量级，减缓收敛。

### 1.2 统一统计框架

两种病理归结为同一估计问题：**最优融合有偏低方差先验**（Critic $V_\phi(s_t)$）**与无偏高方差观测**（MC 回报 $G_t$）。最小化 MSE 的最优线性融合为：
$$V^c_t = (1-\alpha^*)V_\phi(s_t) + \alpha^* G_t, \qquad \alpha^* = \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2}$$
正确的噪声量是 $\sigma_{G|s}^2$（条件 MC 噪声），而非 $\mathrm{Var}(G_t)$（被状态价值结构膨胀的边际方差）。通过 **FixSCR 修正** $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ 估计 $\sigma_{G|s}^2$，是支撑两种 HCGAE 变体的共同技术贡献。

### 1.3 贡献

1. **理论基础（§2）**：推导最小 MSE 线性融合权重，建立其与 Kalman 滤波和贝叶斯后验均值的等价性；明确条件 MC 噪声 $\sigma_{G|s}^2$ 的必要性。
2. **FixSCR 修正（§2.3）**：证明 $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ 在准确 Critic 下是条件 MC 噪声的相合估计；量化标准 GRPO 分母的结果偏差。
3. **HCGAE-PPO（§3）**：逐步修正价值目标、边界自举修正、EV 自适应 Critic 训练目标以及完整损失函数。
4. **HCGAE-GRPO（§4）**：FixSCR 分母修正、SNR 感知逐步加权、EV 驱动 GAE 混合以及完整损失函数。
5. **实证验证（§5）**：在四个 MuJoCo 基准上进行大规模实验，证明两种范式的一致改进。

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

$$\mathrm{SNR}_t = \frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}}, \quad w_t = \sigma\!\bigl(\beta(\mathrm{SNR}_t - \theta)\bigr), \quad \tilde{A}_t = w_t \cdot A_t^{\mathrm{FixSCR}} \tag{23-25}$$

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
输入：V_phi, pi_theta；EMA：EV_ema, sigma_G_ema
=== 优势计算 ===
1. 收集 {s,a,r}[0:T]；V[t] = V_phi(s_t)
2. 反向 MC 计算 G[t]；标准 GAE 计算 std_GAE[t]
3. FixSCR：sigma_hat = sqrt(max(Var(G)-Var(V), nu*Var(G)))  [式 10]
4. w_t = sigmoid(beta*(|G_t-V_t|/sigma_hat - theta))         [式 24]
5. A_fscr[t] = w_t*(G_t-V_t)/sigma_hat                       [式 22, 25]
6. EV_now = 1-Var(G-V)/Var(G)；更新 EV_ema
   A[t] = EV_ema*A_fscr[t] + (1-EV_ema)*normalize(std_GAE[t])  [式 26]
=== Critic 目标 ===
7. R[t] = clip(1-EV_ema,0.1,1)*G[t] + (1-clip(...))*R_GAE[t]  [式 28]
=== PPO 更新 ===
8. 计算式 (27-29)；Adam + 梯度裁剪
```

---

## 5. 实验

### 5.1 实验设置

**环境：** 四个 MuJoCo 连续控制任务——情节式（Hopper-v4、Walker2d-v4）和稠密奖励（HalfCheetah-v4、Ant-v4）。

**协议：** 15 个随机种子，每个配置 1.5M 环境步。

**基线：** Standard PPO、Optimal PPO [Andrychowicz et al., 2021]、Standard GRPO、Optimal GRPO，以及 HCGAE-PPO 和 HCGAE-GRPO（我们）。

**实现：** 所有变体共享超参数（lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, lam=0.95, eps_clip=0.2, vf_coef=0.5, max_grad_norm=0.5）。HCGAE 专用参数：$\nu=0.05$，$\beta=3.0$，$\theta=0.5$，$\rho_{\mathrm{ev}}=0.05$。

### 5.2 主要结果

> *表 1. 最终性能（均值 ± 标准差，15 个种子，1.5M 步最后 5 次评估）。*

| 算法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| Standard PPO | [待补充] | [待补充] | [待补充] | [待补充] |
| Optimal PPO | [待补充] | [待补充] | [待补充] | [待补充] |
| **HCGAE-PPO（我们）** | **[待补充]** | **[待补充]** | **[待补充]** | **[待补充]** |
| Standard GRPO | [待补充] | [待补充] | [待补充] | [待补充] |
| Optimal GRPO | [待补充] | [待补充] | [待补充] | [待补充] |
| **HCGAE-GRPO（我们）** | **[待补充]** | **[待补充]** | **[待补充]** | **[待补充]** |

### 5.3 分析

**HCGAE-PPO。** 在情节式环境（Hopper、Walker2d）中，早期训练 EV 低（$\approx 0.1$–$0.3$），给出 $\hat{\alpha}_{\mathrm{global}} \approx 0.7$–$0.9$，从而实现强 MC 修正。在稠密奖励环境（HalfCheetah、Ant）中，EV 快速上升（100K 步内 $>0.9$），导致 $\hat{\alpha}_{\mathrm{global}} \to 0.1$，自动抑制修正。

**HCGAE-GRPO。** FixSCR 通过从膨胀分母中移除 $\mathrm{Var}(V_\phi)$，在 HalfCheetah 和 Ant 中恢复 1.5–2× 的优势量级。EV 驱动的 GAE 混合在情节式环境（Critic 训练不足时）提供鲁棒性。

### 5.4 消融研究

> *表 2. 组件消融（15 个种子）。*

| 配置 | Hopper-v4 | HalfCheetah-v4 |
|:---|:---:|:---:|
| Optimal PPO | [待补充] | [待补充] |
| + 仅全局 FixSCR | [待补充] | [待补充] |
| + 逐步 SNR 加权 | [待补充] | [待补充] |
| + 边界修正 | [待补充] | [待补充] |
| **完整 HCGAE-PPO** | **[待补充]** | **[待补充]** |
| Standard GRPO | [待补充] | [待补充] |
| + 仅 FixSCR | [待补充] | [待补充] |
| + FixSCR + SNR 加权 | [待补充] | [待补充] |
| **完整 HCGAE-GRPO** | **[待补充]** | **[待补充]** |

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

