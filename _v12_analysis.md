# V12 vs V12b 分析报告

## 实验结果对比

### HalfCheetah-v4
| 版本 | 步数 | seeds | max_mean | last10_mean | 结论 |
|------|------|-------|----------|-------------|------|
| **V12** | 1M | 3 | **2872.6** ± 852 | 2308.4 | 🏆 历史最佳 |
| V12b | 200k | 4 | 789.1 ± 118 | 714.9 | 过度保守 |
| V6 | 200k | 4 | 886.7 ± 160 | 781.0 | 基准 |
| Heuristic | 1M | 3 | 2457.7 ± 1058 | 2197.0 | 高方差 |
| Optimal_PPO | 1M | 12 | 2353.2 ± 1112 | 2099.6 | 基准 |

### Hopper-v4
| 版本 | 步数 | seeds | max_mean | last10_mean | 结论 |
|------|------|-------|----------|-------------|------|
| Optimal_PPO | 1M | 12 | **3773.7** ± 105 | 2225.8 | 🏆 最佳 |
| V12 | 1M | 3 | 2480.2 ± **1273** | 1440.5 | ❌ s2崩溃(680) |
| V12b | 200k | 4 | 1248.1 ± 516 | 713.6 | 稳定但差 |
| V8 | 200k | 4 | 1980.2 ± 302 | 1217.2 | 200k最佳 |

## 核心问题诊断

### V12 的问题
```
公式: α* = (b_V² + σV² - Cov) / (b_V² + σV² + σG² - 2·Cov)
缺陷: σV² = var_d * (1 - EV)
      当 EV 低时 → σV² 过大 → α* → 0.7 上界 → 正反馈崩溃
```

**证据**: Hopper s2 在 EV 下降后，α* 被推到 0.7，导致 Critic 目标混乱。

### V12b 的修复与代价
```
修复:
1. 删除 σV² 估计
2. 用 V6 OLS 公式作为 base: α_cov = -Cov/Var(δ)
3. 添加 EV Guard: α_ev_max = 0.30*(1-EV) + 0.10

代价:
- EV=0.5 → α_max = 0.25 (太保守!)
- HC 需要 α ≈ 0.35-0.45 才能充分发挥 MC 纠偏效果
- V12b 在 HC 200k 只有 789，不如 V6 的 886
```

## V12c 改进方案

### 设计原则
1. **保留 V12 的 bias-aware 思想**：当 Critic 有偏时，需要更大的 α
2. **避免 V12 的 σV² 估计错误**：这是正反馈的根源
3. **调整 EV Guard 的参数**：不那么激进地限制 α

### V12c 核心公式

```python
# Step 1: V6 OLS base (proved optimal for unbiased critic)
alpha_cov = clip(-Cov(r_GAE, δ) / Var(δ), 0, 0.5)

# Step 2: Bias correction (bounded, but less conservative than V12b)
b_V_est = max(0, EMA(mean(G-V)))  # only positive direction
b_V_norm = b_V_est / (sigma_e_ema + ε)
# V12c: higher threshold (0.7 instead of 0.5), smaller max (0.15 instead of 0.20)
bias_add = 0.15 * sigmoid(3 * (b_V_norm - 0.7))

# Step 3: Soft EV Guard (V12c key improvement)
# Instead of hard cap, use soft penalty when EV drops below threshold
ev_clamped = clip(EV_ema, 0, 1)
# Only penalize when EV < 0.4 (HC typical is 0.35-0.5)
if ev_clamped < 0.4:
    ev_penalty = (0.4 - ev_clamped) * 0.5  # max penalty at EV=0: 0.2
    alpha_pre_cap = alpha_cov + bias_add - ev_penalty
else:
    alpha_pre_cap = alpha_cov + bias_add

# Step 4: Final cap (V12c: 0.55 instead of 0.7)
alpha_global = clip(alpha_pre_cap, 0, 0.55)
```

### 关键改进点

| 特性 | V12 | V12b | V12c |
|------|-----|------|------|
| σV² 估计 | ✅ (错误) | ❌ 删除 | ❌ 删除 |
| OLS base | 无 | α_cov | α_cov |
| Bias add | 公式内置 | 0.20*sigmoid(2*(x-0.5)) | 0.15*sigmoid(3*(x-0.7)) |
| EV Guard | ❌ | 硬上限: 0.30*(1-EV)+0.10 | 软惩罚: 仅当 EV<0.4 时 |
| α 上限 | 0.70 | 动态 (0.10-0.40) | 0.55 |
| HC 预期 α | 0.50-0.70 (太高) | 0.15-0.25 (太低) | 0.30-0.45 (合理) |
| Hopper 预期 α | 0.50-0.70 (太高) | 0.15-0.18 (合理) | 0.15-0.25 (合理) |

### 预期效果
- **HC**: α ≈ 0.35-0.45，比 V12b 更能纠偏，但不会像 V12 那样崩溃
- **Hopper**: α ≈ 0.15-0.25，与 V12b 类似，稳定
- **关键**: 软惩罚只在大偏差(EV<0.4)时起效，正常情况下不限制

## 下一步实验

1. 实现 V12c
2. 在 HC 和 Hopper 上跑 4 seeds × 200k 验证
3. 对比 V12/V12b/V12c 的 EV 和 α 轨迹

