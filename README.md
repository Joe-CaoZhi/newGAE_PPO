# BHVF-PPO

> **贝叶斯后见价值融合与可靠性加权策略优化**
> Bayesian Hindsight Value Fusion and Reliability-Weighted Policy Optimization

**ICML 2026 投稿草稿** | 基于贝叶斯第一性原理，从根源解决 PPO 早期训练的两大失效模式。

---

## 📋 摘要 / Abstract

本项目针对 PPO+GAE 在训练早期存在的两类根本性失效模式，提出统一的贝叶斯解决方案：

This project addresses two fundamental failure modes of PPO+GAE during early training with a unified Bayesian solution:

**失效模式 I（Failure Mode I）**：Critic 初始化偏差系统性地污染 GAE 优势估计，在 50K–100K 步暖机完成前持续破坏策略梯度方向。

**失效模式 II（Failure Mode II）**：Clip 代理目标对低质量早期批次与高质量晚期批次施加等权梯度，缺乏对估计质量的自适应能力。

**方案**：
- **BHVF（贝叶斯后见价值融合）**：将价值校正建模为1D卡尔曼滤波问题，推导出解析最优融合增益 $\alpha^* = SCR^2/(SCR^2+1)$（SCR = 信号-校正比），无需任何环境专属超参数。**鲁棒创新截断**以严格统计推断替代全部先验启发式边界规则。
- **DCPPO-S（可靠性加权PPO）**：基于解释方差（EV）的线性收缩调制策略梯度幅度，可证明等价于加性噪声模型下MSE最优的线性估计量，并严格保证梯度方向不变性。

四个 MuJoCo 连续控制基准（Hopper-v4、Walker2d-v4、HalfCheetah-v4、Ant-v4）、12 个随机种子、100 万步训练，验证了 BHVF+DCPPO-S 在所有环境中实现一致性显著提升，无任何环境专属调参。

---

## 🔑 核心方法 / Core Methods

### BHVF — 贝叶斯后见价值融合

在计算任何 TD 残差之前，用蒙特卡洛回报 $G_t$ 对 Critic 进行"后见校正"：

$$V^c(s_t) = V(s_t) + \alpha^* \cdot \mathrm{clip}(G_t - V(s_t),\; -3\sigma_e,\; +3\sigma_e)$$

**定理 1（最优卡尔曼增益）**：在误差独立无偏假设下，最小化 MSE 的唯一最优融合系数为：

$$\alpha^* = \frac{\sigma_V^2}{\sigma_V^2 + \sigma_G^2} = \frac{SCR^2}{SCR^2 + 1}, \quad SCR \triangleq \frac{\sigma_V}{\sigma_G}$$

| 环境类型 | 特征 | SCR | $\alpha^*$ | 效果 |
|:---:|:---:|:---:|:---:|:---:|
| 片段式（Hopper, Walker2d） | Critic 偏差大，MC 相对稳定 | $\gg 1$ | $\to 1$ | 强校正，快速修复 Critic |
| 密集奖励（HalfCheetah 后期）| Critic 已收敛，MC 方差巨大 | $\ll 1$ | $\to 0$ | 自动抑制，防止过度校正 |
| 极端噪声（Ant 全程）| MC 方差极大 | $\approx 0$ | $\approx 0$ | 保守融合，依赖截断保障 |

SCR 通过批内统计在线估计（EMA 平滑，$\eta=0.1$），**单一公式**自动处理全部场景，无需多层叠加的启发式门控。

### DCPPO-S — 可靠性加权策略优化

**定理 2（加性噪声下的最优线性收缩）**：在优势估计 = 真实优势 + 加性噪声的模型下，使 MSE 最小的唯一最优收缩系数等于解释方差：

$$w^\star = \mathrm{EV}_A = \frac{\mathrm{Var}(A_t^\star)}{\mathrm{Var}(\hat{A}_t)}$$

实现中使用 Critic 的可观测 EV 作为代理：

$$\tilde{A}_t = \mathrm{clip}(\widehat{\mathrm{EV}},\; 0.1,\; 1.0) \cdot A_t^{\mathrm{BHVF}}$$

**命题 1（梯度方向不变性）**：通过 stop-gradient 解耦，DCPPO-S 满足 $\nabla_\theta \mathcal{L}_{\mathrm{DCPPO-S}} = w \cdot \nabla_\theta \mathcal{L}_{\mathrm{PPO}}$，仅调节步长幅度，严格保持优化轨迹方向。

---

## 📊 实验结果 / Results

> 四个 MuJoCo 基准，12 个随机种子，100 万步训练（最后 5 次评估的均值 ± 标准差）

| 算法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---:|:---:|:---:|:---:|:---:|
| Standard PPO | — | — | — | — |
| Optimal PPO  | — | — | — | — |
| Heuristic HCGAE | — | — | ↓（过度校正）| — |
| **BHVF（ours）** | **—** | **—** | **—** | **—** |
| **BHVF + DCPPO-S（ours）** | **—** | **—** | **—** | **—** |

> *详细数据见论文草稿 [`docs/paper_draft.md`](docs/paper_draft.md)*

**核心发现**：BHVF 在 HalfCheetah 上彻底克服了先前启发式方法的性能退化，HalfCheetah 正是以往 MC 校正方法的"臭名昭著"失败案例；同时在 Hopper/Walker2d 上保持强烈的正向增益，实现了跨奖励结构的统一提升。

---

## 🗂 项目结构 / Project Structure

```
BHVF-PPO/
│
├── main.py                        # 单环境训练入口（用于快速对比早期方法）
│
├── gae_experiments/               # 核心代码库 / Core library
│   ├── agents/                    # 算法实现 / Algorithm implementations
│   │   ├── base_ppo.py            # 标准 PPO 基线
│   │   ├── optimal_ppo.py         # Optimal PPO + BHVF（主推荐，含贝叶斯框架）
│   │   ├── hindsight_ppo.py       # HCGAE v2（早期启发式版本，用于消融对比）
│   │   ├── dcppo.py               # DCPPO 系列（含 DCPPO-S 可靠性加权）
│   │   ├── ppo_baselines.py       # 多种 PPO 变体基线（KL-Pen, VClip, Anneal, EntDecay）
│   │   ├── hindsight_ablation.py  # HCGAE 消融变体（4 项改进全排列）
│   │   ├── multiscale_ppo.py      # MSGAE（多尺度 GAE，早期探索）
│   │   ├── causal_attention_ppo.py# CAGAE（因果注意力 GAE，早期探索）
│   │   ├── advance_ppo.py         # 扩展实验变体
│   │   ├── sac_td3.py             # SAC/TD3 离策略基线（对比用）
│   │   └── ...                    # 其他变体
│   ├── utils/                     # 工具模块 / Utilities
│   │   ├── networks.py            # Actor/Critic 网络
│   │   ├── rollout_buffer.py      # Rollout 缓冲区
│   │   ├── logger.py              # 指标记录器
│   │   └── visualizer.py          # 可视化工具
│   └── experiment.py              # 实验运行框架
│
├── scripts/                       # 脚本目录 / Scripts
│   ├── experiments/               # 实验运行脚本（41 个）
│   │   ├── run_large_scale_experiment.py   # 🔑 主实验：8环境×12种子×1M步
│   │   ├── run_icml_experiment.py          # ICML 正式实验配置
│   │   ├── run_v4_full.py                  # V4 完整实验
│   │   ├── run_baseline_comparison.py      # 基线对比实验
│   │   ├── run_unified_comparison.py       # 统一比较实验
│   │   ├── run_aligned_experiment.py       # 对齐实验（消融）
│   │   ├── run_ablation.py                 # HCGAE 消融
│   │   ├── run_dcppo.py                    # DCPPO 消融
│   │   └── ...                             # 其他实验脚本
│   ├── analysis/                  # 分析与可视化脚本（140 个）
│   │   ├── generate_paper_figures.py       # 生成论文图表
│   │   ├── generate_icml_figures.py        # 生成 ICML 图表
│   │   ├── compute_unified_stats.py        # 统一统计分析
│   │   ├── verify_paper_data.py            # 数据核验
│   │   └── ...                             # 其他分析脚本
│   ├── compute_aulc.py            # 面积下学习曲线计算
│   ├── compute_stats.py           # 统计量计算
│   ├── full_analysis.py           # 完整分析流程
│   ├── icml_analysis.py           # ICML 专项分析
│   ├── stat_analysis_dcppo.py     # DCPPO 统计分析
│   ├── generate_icml_figures.py   # ICML 图表生成
│   ├── analyze_aligned_results.py # 对齐实验结果分析
│   └── monitor_icml.sh            # ICML 实验监控脚本
│
├── docs/                          # 文档目录 / Documentation
│   ├── paper_draft.md             # 📄 论文草稿（英文版，ICML 风格）
│   ├── paper_draft_zh.md          # 📄 论文草稿（中文版）
│   ├── 技术报告.md                 # 完整数学推导与实验分析（中文）
│   ├── TECH_REPORT_EN.md          # 技术报告（英文精简版）
│   ├── EXP_RECORD.md              # 完整实验记录（所有变体、原始数据）
│   ├── ABLATION_REPORT.md         # HCGAE 消融专项报告
│   └── ADVANCE_PPO_DESIGN.md      # DCPPO 设计文档
│
├── results/                       # 实验数据 / Experiment data
│   ├── ICMLExperiment/            # 🔑 ICML 正式实验（4环境×多算法×12种子）
│   ├── BaselineComparison/        # 基线对比数据
│   ├── AlignedExperiment/         # 对齐消融数据
│   ├── MultiEnv/                  # 多环境快速验证
│   ├── V4FullExperiment/          # V4 完整实验
│   ├── UnifiedComparison/         # 统一比较实验
│   ├── paper_figures/             # 论文图表
│   └── ...                        # 其他实验数据
│
└── logs/                          # 日志文件 / Logs
    └── bayesian_experiment.log
```

---

## 🚀 快速开始 / Quick Start

### 安装依赖 / Installation

```bash
pip install gymnasium[mujoco] torch numpy matplotlib scipy
```

### 运行主实验 / Run Main Experiments

```bash
# 大规模实验（12 种子，100 万步，推荐）
python scripts/experiments/run_large_scale_experiment.py

# ICML 正式实验（4 环境，12 种子）
python scripts/experiments/run_icml_experiment.py

# V4 完整实验（含 BHVF 各变体对比）
python scripts/experiments/run_v4_full.py

# 基线对比（PPO 变体 vs BHVF）
python scripts/experiments/run_baseline_comparison.py

# 单环境快速对比（CartPole，用于早期方法验证）
python main.py --env Hopper-v4
```

### 生成论文图表 / Generate Figures

```bash
# 生成 ICML 论文图表
python scripts/analysis/generate_paper_figures.py

# ICML 专项分析
python scripts/icml_analysis.py

# 统计显著性分析
python scripts/analysis/compute_unified_stats.py
```

---

## 🔬 算法实现核心 / Algorithm Core

### BHVF 完整算法

```python
# 批内统计
sigma_V = mean(|G - V|)        # Critic MAE（鲁棒 RMSE 代理）
sigma_G = std(G)               # MC 回报标准差
SCR = sigma_V / sigma_G        # 信号-校正比（EMA 平滑）

# 最优卡尔曼增益
alpha_star = SCR**2 / (SCR**2 + 1) + delta_relax   # delta_relax=0.05

# 鲁棒创新截断（99.7% 置信区间）
sigma_e = std(G - V)
innovation_clipped = clip(G - V, -3*sigma_e, +3*sigma_e)

# 后见校正价值
V_corrected = V + alpha_star * innovation_clipped

# 用校正值重算 GAE
A_BHVF = GAE(r, V_corrected, gamma=0.99, lam=0.95)

# EV 驱动 Critic 训练目标混合
c_mc = clip(1 - EV_ema, 0.1, 1.0)
R_target = c_mc * G + (1 - c_mc) * (A_BHVF + V_corrected)
```

### DCPPO-S 核心

```python
# 基于 EV 的线性收缩（MSE 最优）
w = clip(EV_ema, w_min=0.1, w_max=1.0)

# 有效优势（方向严格不变，幅度自适应调节）
A_eff = w * A_BHVF

# 标准 PPO 裁剪目标（梯度方向 = w * ∇L_PPO）
L = -E[min(rho * A_eff, clip(rho, 1±eps) * A_eff)]
```

---

## 📚 文档导航 / Documentation

| 文档 | 语言 | 内容 |
|---|---|---|
| [`docs/paper_draft.md`](docs/paper_draft.md) | EN | 📄 完整论文草稿（ICML 风格，含定理证明）|
| [`docs/paper_draft_zh.md`](docs/paper_draft_zh.md) | ZH | 📄 论文草稿中文版 |
| [`docs/技术报告.md`](docs/技术报告.md) | ZH | 完整数学推导、实验分析、理论证明 |
| [`docs/TECH_REPORT_EN.md`](docs/TECH_REPORT_EN.md) | EN | 技术报告英文精简版 |
| [`docs/EXP_RECORD.md`](docs/EXP_RECORD.md) | EN | 完整实验记录（所有变体与原始数据） |
| [`docs/ABLATION_REPORT.md`](docs/ABLATION_REPORT.md) | EN | HCGAE 消融分析专项报告 |
| [`docs/ADVANCE_PPO_DESIGN.md`](docs/ADVANCE_PPO_DESIGN.md) | ZH | DCPPO 设计思路与动机分析 |

---

## 📈 研究演进 / Research Evolution

本项目经历了从启发式工程到贝叶斯第一性原理的系统性演进：

| 版本 | 方法 | 状态 | 说明 |
|---|---|:---:|---|
| v0 | Standard PPO（基线） | ✅ | 参考实现 |
| v1 | HCGAE（早期启发式）| ✅ | 多层叠加门控，环境专属，难以泛化 |
| v2 | HCGAE_v2（改进启发式）| ✅ | 批内归一化 + EV 驱动目标混合 |
| **v3** | **BHVF（贝叶斯框架）** | **✅ 当前主线** | **解析最优增益，零环境专属超参** |
| **v3+** | **BHVF + DCPPO-S** | **✅ 当前推荐** | **完整框架，ICML 投稿版本** |
| 探索 | MSGAE（多尺度 GAE）| ✅ | 实现完成，效果良好 |
| 探索 | CAGAE（因果注意力 GAE）| ✅ | 实现完成，效果一般 |
| 探索 | DCPPO-G（几何均值 Ratio）| ⚠️ | 单独尚可，与其他改进有拮抗 |
| 探索 | DCPPO-A（非对称裁剪）| ⚠️ | 单独有效，与 G 组合有拮抗 |

---

## 🔧 超参数说明 / Hyperparameters

所有提出方法在**所有环境使用完全相同的超参数**，无任何环境专属调整：

| 超参数 | 值 | 说明 |
|---|:---:|---|
| `clip_c` | 3.0 | 鲁棒创新截断系数（对应 99.7% 置信区间）|
| `scr_ema_lr` | 0.1 | SCR EMA 学习率 |
| `scr_relax` | 0.05 | SCR 数值松弛项（防止 $\alpha^*$ 塌陷到零）|
| `w_min` | 0.1 | DCPPO-S 收缩下界（防止训练完全停止）|
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE $\lambda$ |
| `lr` | 3e-4 | 学习率（带线性退火）|
| `n_steps` | 2048 | Rollout 长度 |
| `batch_size` | 64 | 小批量大小 |
| `n_epochs` | 10 | PPO 更新轮数 |

---

## 引用 / Citation

如果您使用了本项目的代码或方法，请引用：

If you use this code or methods, please cite:

```bibtex
@misc{bhvf_ppo_2026,
  title  = {Bayesian Hindsight Value Fusion and Reliability-Weighted Policy Optimization},
  author = {Anonymous},
  year   = {2026},
  note   = {ICML 2026 Submission (Under Review)}
}
```

---

## 许可证 / License

MIT License
