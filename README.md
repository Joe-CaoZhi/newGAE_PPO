# newGAE_PPO

> **回顾校正 GAE 与信噪比自适应 PPO 改进**
> Hindsight-Corrected GAE with SNR-Adaptive Policy Optimization

本项目研究并实现了针对 Proximal Policy Optimization (PPO) 的多项改进方案，聚焦于广义优势估计（GAE）和策略更新机制两个关键环节。

---

## 核心方法

### 🔑 HCGAE — 回顾校正广义优势估计

训练早期 Critic 存在大量偏差，污染 GAE 的优势估计。HCGAE 在 rollout 采集完成后，使用可获得的精确蒙特卡洛回报 $G_t$ 对 Critic 值进行"事后校正"：

$$V^c(s_t) = (1 - \alpha_t)\,V(s_t) + \alpha_t\,G_t, \quad \alpha_t = \alpha_{\max}(k)\cdot\sigma\!\left(\beta\cdot\frac{e_t - \mu_e}{\sigma_e}\right)$$

**推荐配置（HCGAE_Imp12）** 包含两项协同改进：
- **改进①**：批内中心化 Sigmoid 归一化——消除 EMA 滞后缺陷，使校正强度相对于当前批次的平均误差水平
- **改进②**：EV 驱动 Critic 目标混合——Critic 精度低时多用 MC 目标，精度高时多用 GAE 自举目标

**效果**：Hopper-v4（500K 步）相比标准 GAE 提升 **+413%**

### 🔑 DCPPO-S — 信噪比自适应梯度缩放

根据批内优势估计的信噪比动态调节策略梯度幅度：

$$\mathrm{SNR} = \frac{|\bar{A}|}{\hat{\sigma}_A + \varepsilon}, \qquad w = \max\!\left(w_{\min},\;\min\!\left(1,\;\left(\frac{\mathrm{SNR}}{\mathrm{SNR}^*}\right)^{\gamma_s}\right)\right)$$

梯度方向可证明无偏（$\nabla_\theta \mathcal{L}_S = w \cdot \nabla_\theta \mathcal{L}_{\mathrm{PPO}}$），仅幅度自适应调节。

**效果**：训练不稳定性降低 **20×**（σ: 949 → 49），配合 HCGAE 形成正向协同循环。

---

## 实验结果速览

| 方法 | Hopper-v4 最终奖励 | 稳定性 σ |
|---|:---:|:---:|
| 标准 GAE（基线） | 656 | — |
| HCGAE_Imp12 | 3363 (+413%) | — |
| **HCGAE_Imp12 + DCPPO-S** | **3495 (+433%)** | **49 (↓20×)** |

---

## 项目结构

```
newGAE_PPO/
│
├── gae_experiments/              # 核心代码库
│   ├── agents/                   # 算法实现
│   │   ├── base_ppo.py           # 标准 PPO 基线
│   │   ├── hindsight_ppo.py      # HCGAE v2（推荐使用）
│   │   ├── dcppo.py              # DCPPO（G/A/S 三项改进 + HCGAE）
│   │   ├── hindsight_ablation.py # HCGAE 消融变体（4 项改进全排列）
│   │   ├── multiscale_ppo.py     # MSGAE（多尺度 GAE）
│   │   ├── causal_attention_ppo.py # CAGAE（因果注意力 GAE）
│   │   └── advance_ppo.py        # 扩展实验变体
│   ├── utils/                    # 工具模块
│   │   ├── networks.py           # Actor/Critic 网络
│   │   ├── rollout_buffer.py     # Rollout 缓冲区
│   │   └── logger.py             # 指标记录器
│   └── experiment.py             # 实验运行框架
│
├── docs/                         # 文档（中英文）
│   ├── 技术报告.md               # 中文技术报告（数学推导 + 实验分析）
│   ├── TECH_REPORT_EN.md         # 英文技术报告（ICML 风格精简版）
│   ├── EXP_RECORD.md             # 完整实验记录（所有推导、消融、原始数据）
│   ├── ABLATION_REPORT.md        # HCGAE 消融实验专项报告
│   └── ADVANCE_PPO_DESIGN.md     # DCPPO 设计文档
│
├── results/                      # 实验数据与图表
│   ├── Hopper-v4-DCPPO/          # DCPPO 消融实验数据（主要结果）
│   ├── Hopper-v4-Ablation/       # HCGAE 消融实验数据
│   ├── MultiEnv/                 # 多环境对比数据
│   │   ├── Hopper-v4/
│   │   ├── HalfCheetah-v4/
│   │   ├── Walker2d-v4/
│   │   └── Ant-v4/
│   ├── Advance-Ablation/         # 扩展消融数据
│   └── Hopper-v4/                # 早期 Hopper 实验数据
│
├── run_ablation.py               # 运行 HCGAE 消融实验
├── run_dcppo.py                  # 运行 DCPPO 消融实验
├── run_multi_env_seeds.py        # 多环境多种子实验
├── run_advance_ablation.py       # 扩展消融实验
├── analyze_dcppo_results.py      # DCPPO 结果分析与可视化
├── analyze_ablation.py           # HCGAE 消融结果分析
├── analyze_advance_results.py    # 扩展实验分析
├── main.py                       # 单环境训练入口
│
├── TECH_REPORT.md                # 英文技术报告（根目录镜像）
└── EXP_RECORD.md                 # 完整实验记录（根目录镜像）
```

---

## 快速开始

### 安装依赖

```bash
pip install gymnasium[mujoco] torch numpy matplotlib
```

### 运行主实验

```bash
# 标准训练（Hopper-v4，使用 HCGAE_Imp12）
python main.py --env Hopper-v4 --agent hindsight --total_steps 500000

# DCPPO 消融实验（9 个变体，Hopper-v4）
python run_dcppo.py --env Hopper-v4 --total_steps 500000

# HCGAE 消融实验（6 个变体，Hopper-v4）
python run_ablation.py --env Hopper-v4 --total_steps 300000

# 多环境多种子对比
python run_multi_env_seeds.py
```

### 分析与可视化

```bash
# 生成 DCPPO 综合分析图（雷达图、热力图、学习曲线）
python analyze_dcppo_results.py

# 生成 HCGAE 消融分析图
python analyze_ablation.py
```

---

## 算法实现说明

### HCGAE 的核心流程

```python
# 1. 计算 MC 回报（rollout 完成后反向累加）
G[T-1] = r[T-1] + γ * V(s_T)
G[t]   = r[t] + γ * G[t+1] * (1 - done[t])

# 2. 批内中心化归一化（改进①）
e = |V - G|           # Critic 绝对误差
z = β * (e - e.mean()) / (e.std() + ε)
α = α_max(k) * sigmoid(z)

# 3. 校正价值
V_c = (1 - α) * V + α * G

# 4. 用校正值重算 TD 残差和 GAE
δ_c = r + γ * V_c_next - V_c
A   = GAE(δ_c, γ, λ)

# 5. EV 驱动目标混合（改进②）
c_mc = clip(1 - EV_ema, 0.1, 1.0)
R    = c_mc * G + (1 - c_mc) * (A + V)
```

### DCPPO-S 的核心流程

```python
# 计算批内 SNR
SNR = |A.mean()| / (A.std() + ε)

# 梯度缩放权重
w = clip((SNR / SNR_target) ** γ_s, w_min, 1.0)

# 有效优势（方向不变，幅度调节）
A_eff = w * A

# 标准 PPO 裁剪目标
L = -E[min(ρ * A_eff, clip(ρ, 1±ε) * A_eff)]
```

---

## 文档导航

| 文档 | 语言 | 内容 |
|---|---|---|
| [`docs/技术报告.md`](docs/技术报告.md) | 中文 | 完整数学推导、实验分析、理论证明 |
| [`docs/TECH_REPORT_EN.md`](docs/TECH_REPORT_EN.md) | 英文 | ICML 风格精简报告 |
| [`docs/EXP_RECORD.md`](docs/EXP_RECORD.md) | 英文 | 完整实验记录（所有变体、原始数据） |
| [`docs/ABLATION_REPORT.md`](docs/ABLATION_REPORT.md) | 英文 | HCGAE 消融分析专项报告 |
| [`docs/ADVANCE_PPO_DESIGN.md`](docs/ADVANCE_PPO_DESIGN.md) | 中文 | DCPPO 设计思路与动机分析 |

---

## 研究背景

本项目探索了以下三类 GAE 改进方向：

| 方法 | 解决的问题 | 状态 |
|---|---|---|
| **HCGAE** | Critic 初始化偏差污染优势估计 | ✅ 已验证，推荐使用 |
| **MSGAE** | 固定时域参数 $\lambda$ 无法自适应 | ✅ 实现完成，效果良好 |
| **CAGAE** | 对所有 TD 残差等权，忽略局部信号质量 | ✅ 实现完成，效果一般 |
| **DCPPO-S** | 策略更新不感知优势信噪比 | ✅ 最优改进，显著提升稳定性 |
| **DCPPO-G** | 高维动作空间 Ratio 方差膨胀 | ⚠️ 单独使用尚可，与其他改进有强拮抗 |
| **DCPPO-A** | 对称裁剪的方向性不一致 | ⚠️ 单独有效，与 G 组合有拮抗 |

---

## 引用

如果您使用了本项目的代码或方法，请引用：

```bibtex
@misc{newgae_ppo_2026,
  title  = {Hindsight-Corrected GAE with SNR-Adaptive Policy Optimization},
  author = {Joe-CaoZhi},
  year   = {2026},
  note   = {GitHub: https://github.com/Joe-CaoZhi/newGAE_PPO}
}
```

---

## 许可证

MIT License
