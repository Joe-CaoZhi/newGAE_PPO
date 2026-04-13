# HCGAE: Hindsight-Corrected Generalized Advantage Estimation

> **事后修正广义优势估计：PPO 与 GRPO 的统一统计框架**
> Hindsight-Corrected Generalized Advantage Estimation: A Unified Statistical Framework for PPO and GRPO

**ICML 2026 投稿草稿** | 匿名提交 · 审稿中

---

## 📋 摘要 / Abstract

优势估计质量是在线策略梯度方法学习效率的核心决定因素。本文识别了两种主流范式中共同存在的统计缺陷：在 PPO 中，Critic 初始化偏置通过 GAE 的 $(\gamma\lambda)^l$ 加权几何级累积，污染早期策略梯度；在 GRPO 中，组归一化分母将状态价值结构性方差与真实 MC 噪声混为一谈，系统性压缩优势信号。

We identify a shared statistical deficiency across two dominant paradigms: in PPO, Critic initialization bias propagates geometrically through GAE's $(\gamma\lambda)^l$ weighting, corrupting early policy gradients; in GRPO, the group normalization denominator conflates state-value structural variance with true MC noise, systematically deflating the advantage signal.

**方案 / Solution**: 我们提出 **HCGAE（事后修正广义优势估计）**，以有偏先验（Critic 估计）与无偏但噪声的观测（MC 回报）的最优线性融合为理论基础。最优融合系数为：

$$\alpha^* = \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2}$$

其中 $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ 是**条件** MC 噪声，与被状态价值结构膨胀的边际方差 $\mathrm{Var}(G_t)$ 有本质区别。通过 **FixSCR 修正** $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$（源自全方差定律）估计该量，是支撑两种变体的统一技术贡献。

---

## 🔑 核心方法 / Core Methods

### 1. 理论基础：最优线性融合

**定理 1（最小 MSE 线性融合）**：在所有线性估计量中，最小化 MSE 的唯一最优融合系数为 $\alpha^*$（上式），等价于：
- **Kalman 增益**：先验方差 $\sigma_V^2$、观测噪声 $\sigma_{G|s}^2$ 的一维 Kalman 滤波
- **贝叶斯后验均值**：高斯先验下的 MAP 估计

**FixSCR（定理 2）**：由全方差定律 $\mathrm{Var}(G_t) = \mathrm{Var}(V^\pi(s_t)) + \sigma_{G|s}^2$，当 Critic 准确时 $\mathrm{Var}(V_\phi) \approx \mathrm{Var}(V^\pi)$，故：
$$\hat{\sigma}_{G|s} = \sqrt{\max\!\bigl(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\; \nu \cdot \mathrm{Var}(G)\bigr)}, \quad \nu = 0.05$$

在 HalfCheetah-v4 中，标准 GRPO 的 $\sigma_G$ 对真实噪声高估达 **2.2×**，FixSCR 直接修正这一偏差。

### 2. HCGAE-PPO

HCGAE-PPO 在计算 TD 残差前，对每个时步的价值估计进行融合修正：

$$V^c_t = (1 - \alpha_t)\,V_\phi(s_t) + \alpha_t\,G_t$$

逐步增益 $\alpha_t$ 通过 EV（解释方差）和 SCR（信号-修正比）联合控制：

$$\alpha_t = \underbrace{\min\bigl(1 - \widehat{\mathrm{EV}},\; \hat{\alpha}_{\mathrm{SCR}}\bigr)}_{\hat{\alpha}_{\mathrm{cap}}} \cdot \sigma\!\left(\beta \cdot \frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}} - \beta\theta\right)$$

修正后 GAE：$A_t^{\mathrm{HCGAE-PPO}} = \sum_{l}(\gamma\lambda)^l (r_{t+l} + \gamma V^c_{t+l+1} - V^c_{t+l})$

**完整损失函数：**
$$\mathcal{L}(\theta, \phi) = \underbrace{-\mathbb{E}_t[\min(r_t A_t^{\mathrm{HCGAE}},\, \mathrm{clip}(r_t,1\pm\epsilon) A_t^{\mathrm{HCGAE}})]}_{\mathcal{L}^{\mathrm{CLIP}}} + c_{\mathrm{vf}} \underbrace{\tfrac{1}{2}\mathbb{E}_t[(V_\phi - \mathcal{R}_t)^2]}_{\mathcal{L}^{\mathrm{VF}}}$$

其中 $\mathcal{R}_t = c_{\mathrm{MC}} G_t + (1 - c_{\mathrm{MC}}) \hat{R}^{\mathrm{GAE}}_t$，$c_{\mathrm{MC}} = \mathrm{clip}(1 - \widehat{\mathrm{EV}}, 0.1, 1.0)$。

### 3. HCGAE-GRPO

HCGAE-GRPO 将 FixSCR 修正应用于 GRPO 的归一化分母，并结合 SNR 感知加权：

| 方面 | HCGAE-PPO | HCGAE-GRPO |
|:---|:---|:---|
| **失真来源** | TD 累积中的 Critic 偏置 | 归一化分母中的方差膨胀 |
| **修正目标** | $V_\phi(s_t) \to V^c_t$ | $\sigma_G \to \hat{\sigma}_{G\|s}$ |
| **优势形式** | 修正 $\delta^c_t$ 上的 GAE | $(G_t - V_\phi)/\hat{\sigma}_{G\|s}$ |

**FixSCR 归一化：**
$$A_t^{\mathrm{FixSCR}} = \frac{G_t - V_\phi(s_t)}{\hat{\sigma}_{G|s}}$$

**SNR 感知加权 + EV 驱动混合：**
$$A_t^{\mathrm{HCGAE-GRPO}} = \mathrm{ev\_blend} \cdot w_t \cdot A_t^{\mathrm{FixSCR}} + (1 - \mathrm{ev\_blend}) \cdot \bar{A}_t^{\mathrm{GAE}}$$

---

## 📊 实验结果 / Results

> 四个 MuJoCo 基准，15 个随机种子，1.5M 环境交互步数

| 算法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---:|:---:|:---:|:---:|:---:|
| Standard PPO | [TBD] | [TBD] | [TBD] | [TBD] |
| Optimal PPO | [TBD] | [TBD] | [TBD] | [TBD] |
| **HCGAE-PPO (ours)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |
| Standard GRPO | [TBD] | [TBD] | [TBD] | [TBD] |
| Optimal GRPO | [TBD] | [TBD] | [TBD] | [TBD] |
| **HCGAE-GRPO (ours)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

> *完整实验数据和图表见 [`docs/paper_draft.md`](docs/paper_draft.md)*

---

## 🗂 项目结构 / Project Structure

```
newGAE_ppo/
│
├── gae_experiments/               # 核心代码库 / Core library
│   ├── agents/                    # 算法实现 / Algorithm implementations
│   │   ├── optimal_ppo.py         # 🔑 HCGAE-PPO & HCGAE-GRPO（主实现）
│   │   ├── base_ppo.py            # Standard PPO 基线
│   │   ├── ppo_baselines.py       # Optimal PPO 及其他 PPO 变体基线
│   │   ├── dcppo.py               # DCPPO 系列
│   │   ├── hindsight_ppo.py       # 早期启发式版本（用于消融对比）
│   │   ├── hindsight_ablation.py  # HCGAE 消融变体
│   │   └── ...                    # 其他早期探索变体
│   ├── utils/                     # 工具模块 / Utilities
│   │   ├── networks.py            # Actor/Critic 网络（正交初始化等）
│   │   ├── rollout_buffer.py      # Rollout 缓冲区
│   │   ├── logger.py              # 指标记录器
│   │   └── visualizer.py          # 可视化工具
│   └── experiment.py              # 实验运行框架
│
├── scripts/                       # 脚本目录 / Scripts
│   ├── experiments/               # 实验运行脚本
│   │   ├── run_large_scale_experiment.py   # 主实验：4环境×15种子×1.5M步
│   │   ├── run_icml_experiment.py          # ICML 正式实验配置
│   │   ├── run_grpo_experiment.py          # GRPO 系列实验
│   │   ├── run_final_optimal.py            # Final Optimal 实验
│   │   └── ...
│   └── analysis/                  # 分析与可视化脚本
│       ├── generate_paper_figures.py       # 生成论文图表
│       ├── compute_unified_stats.py        # 统一统计分析
│       └── ...
│
├── docs/                          # 文档目录 / Documentation
│   ├── paper_draft.md             # 📄 论文草稿（英文版，ICML 2026）
│   ├── paper_draft_zh.md          # 📄 论文草稿（中文版）
│   ├── TECH_REPORT.md             # 完整技术报告（中文）
│   ├── TECH_REPORT_EN.md          # 技术报告（英文版）
│   ├── EXP_RECORD.md              # 完整实验记录
│   ├── ABLATION_REPORT.md         # 消融实验专项报告
│   └── archive/                   # 归档旧版文档
│
├── results/                       # 实验数据 / Experiment data
│   ├── ICMLExperiment/            # 🔑 ICML 正式实验（4环境×多算法×12种子）
│   ├── FinalExperiment/           # Final 实验（含所有版本对比）
│   ├── FinalOptimal/              # 最优配置实验（4环境×12种子×1M步）
│   ├── QuickValid_Optimal/        # 快速验证实验
│   ├── GRPO/                      # GRPO 系列实验
│   ├── MultiSeedPower/            # 多种子统计功效实验
│   ├── ExtendedEnvs/              # 扩展环境实验（Swimmer/Humanoid等）
│   └── paper_figures_final/       # 论文最终图表
│
├── logs/                          # 实验日志
├── main.py                        # 单环境训练入口
├── run_grpo_experiment.py         # GRPO 实验启动脚本
├── run_final_optimal.py           # Final Optimal 实验启动脚本
└── README.md                      # 本文件
```

---

## 🚀 快速开始 / Quick Start

### 安装依赖 / Installation

```bash
pip install gymnasium[mujoco] torch numpy matplotlib scipy
```

### 运行主实验 / Run Main Experiments

```bash
# ICML 正式实验（4 环境，15 种子，1.5M 步）
python scripts/experiments/run_icml_experiment.py

# GRPO 系列实验
python run_grpo_experiment.py

# Final Optimal 实验（HCGAE-PPO vs Baselines）
python run_final_optimal.py

# 单环境快速验证
python main.py --env Hopper-v4
```

### 生成论文图表 / Generate Figures

```bash
# 生成论文图表
python scripts/analysis/generate_paper_figures.py

# ICML 专项统计分析
python scripts/icml_analysis.py
```

---

## 🔬 算法实现核心 / Algorithm Core

### HCGAE-PPO 完整算法

```python
# === 每个 rollout 执行 ===
# 1. 计算 FixSCR 条件 MC 噪声
var_G_cond = max(Var(G) - Var(V_phi), nu * Var(G))   # nu=0.05
sigma_hat = sqrt(var_G_cond)

# 2. 全局增益上限（EV + SCR 联合控制）
alpha_cap = min(1 - EV_ema, SCR_ema**2 / (1 + SCR_ema**2))

# 3. 逐步增益（局部 SNR 调制）
snr_t = |G_t - V_t| / sigma_hat
alpha_t = alpha_cap * sigmoid(beta * (snr_t - theta))   # beta=3.0, theta=0.5

# 4. 修正价值目标
V_c[t] = (1 - alpha_t) * V_phi(s_t) + alpha_t * G[t]

# 5. 修正 GAE
A[t] = corrected_GAE(r, V_c, gamma=0.99, lam=0.95)

# 6. EV 自适应 Critic 训练目标
c_mc = clip(1 - EV_ema, 0.1, 1.0)
R_target[t] = c_mc * G[t] + (1 - c_mc) * R_GAE[t]
```

### HCGAE-GRPO 完整算法

```python
# === 每个 rollout 执行 ===
# 1. FixSCR 分母修正
sigma_hat = sqrt(max(Var(G) - Var(V_phi), nu * Var(G)))

# 2. SNR 感知加权
snr_t = |G_t - V_phi(s_t)| / sigma_hat
w_t = sigmoid(beta * (snr_t - theta))

# 3. FixSCR 归一化优势
A_fscr[t] = w_t * (G_t - V_phi(s_t)) / sigma_hat

# 4. EV 驱动 GRPO/GAE 混合
ev_blend = clip(EV_ema, 0, 1)
A[t] = ev_blend * A_fscr[t] + (1 - ev_blend) * normalize(A_GAE[t])
```

---

## 📚 文档导航 / Documentation

| 文档 | 语言 | 内容 |
|---|---|---|
| [`docs/paper_draft.md`](docs/paper_draft.md) | EN | 📄 完整论文草稿（ICML 2026，含完整定理证明）|
| [`docs/paper_draft_zh.md`](docs/paper_draft_zh.md) | ZH | 📄 论文草稿中文版（与英文版完全同步）|
| [`docs/TECH_REPORT.md`](docs/TECH_REPORT.md) | ZH | 完整技术报告：数学推导、实验分析 |
| [`docs/TECH_REPORT_EN.md`](docs/TECH_REPORT_EN.md) | EN | Technical Report (English) |
| [`docs/EXP_RECORD.md`](docs/EXP_RECORD.md) | EN/ZH | 完整实验记录（所有变体与原始数据）|
| [`docs/ABLATION_REPORT.md`](docs/ABLATION_REPORT.md) | EN | 消融实验专项分析报告 |

---

## 🔧 超参数说明 / Hyperparameters

**所有 HCGAE 变体在所有环境中使用完全相同的超参数**，无任何环境专属调整：

| 超参数 | 值 | 说明 |
|---|:---:|---|
| `nu` | 0.05 | FixSCR 地板系数（防止数值不稳定）|
| `beta` | 3.0 | SNR 逻辑增益（逐步 SNR 加权斜率）|
| `theta` | 0.5 | SNR 逻辑偏置（半信号截止点）|
| `rho_ev` | 0.05 | EV EMA 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE $\lambda$ |
| `lr` | 3e-4 | 学习率（带线性退火）|
| `n_steps` | 2048 | Rollout 长度 |
| `batch_size` | 64 | 小批量大小 |
| `n_epochs` | 10 | PPO 更新轮数 |
| `eps_clip` | 0.2 | PPO 截断系数 |
| `vf_coef` | 0.5 | 价值损失系数 |

---

## 📈 研究演进 / Research Evolution

| 阶段 | 方法 | 状态 | 说明 |
|---|---|:---:|---|
| 基线 | Standard PPO | ✅ | 标准 PPO 基线 |
| 基线 | Optimal PPO | ✅ | Andrychowicz et al. (2021) 最佳实践 |
| 早期探索 | HCGAE 启发式 v1 | ✅ | 多层叠加门控，难以泛化（归档）|
| 早期探索 | HCGAE v2 | ✅ | 批内归一化 + EV 驱动目标混合（归档）|
| **当前主线** | **HCGAE-PPO (Optimal)** | **✅ 推荐** | **最优线性融合，FixSCR 条件 MC 估计** |
| **当前主线** | **HCGAE-GRPO** | **✅ 推荐** | **FixSCR 分母修正 + SNR 感知加权** |
| 消融 | FixSCR Only | ✅ | FixSCR 单独消融 |
| 消融 | No Boundary Correction | ✅ | 边界修正消融 |
| 消融 | No EV Gate | ✅ | EV 门控消融 |

---

## 引用 / Citation

如果您使用了本项目的代码或方法，请引用：

If you use this code or methods, please cite:

```bibtex
@misc{hcgae_ppo_2026,
  title  = {Hindsight-Corrected Generalized Advantage Estimation:
            A Unified Statistical Framework for PPO and GRPO},
  author = {Anonymous},
  year   = {2026},
  note   = {ICML 2026 Submission (Under Review)}
}
```

---

## 许可证 / License

MIT License
