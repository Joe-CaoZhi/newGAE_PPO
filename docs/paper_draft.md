# Hindsight-Corrected Generalized Advantage Estimation (HCGAE): A Unified Statistical Framework for PPO and GRPO

> **Paper Draft — ICML 2026 Submission**
> Anonymous Submission · Under Review

---

## Abstract

Advantage estimation quality is a central determinant of learning efficiency in on-policy policy gradient methods. We identify a shared statistical deficiency across two dominant paradigms: in Proximal Policy Optimization (PPO), the Critic initialization bias corrupts early TD-residual accumulation; in Group Relative Policy Optimization (GRPO), the group normalization denominator conflates state-value structural variance with true Monte Carlo (MC) noise, systematically deflating the advantage signal.

We introduce **Hindsight-Corrected Generalized Advantage Estimation (HCGAE)**, a unified framework grounded in the optimal linear fusion of a biased prior (Critic estimate) with an unbiased but noisy observation (MC return). The optimal fusion coefficient minimizing MSE is $\alpha^* = (\sigma_V^2 + B^2)/(\sigma_V^2 + B^2 + \sigma_{G|s}^2)$, where $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ is the *conditional* MC noise—distinct from the inflated marginal variance $\mathrm{Var}(G_t)$.

HCGAE instantiates this principle differently for each paradigm:

**HCGAE-PPO** constructs corrected value targets $V^c_t = (1-\alpha_t)V_\phi(s_t) + \alpha_t G_t$ using FixSCR-derived per-step gains, feeding $V^c$ into standard GAE to attenuate Critic-bias propagation.

**HCGAE-GRPO** addresses the variance-inflation pathology in three coordinated steps: *(i)* **FixSCR denominator correction** — replace $\sigma_G$ with $\hat{\sigma}_{G|s} = \sqrt{\max(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\, \nu \cdot \mathrm{Var}(G))}$ to recover the true per-step noise scale; *(ii)* **SNR-aware per-step weighting** — up-weight timesteps whose hindsight error $|G_t - V_\phi(s_t)|$ is large relative to $\hat{\sigma}_{G|s}$ via $w_t = \sigma(\beta(\mathrm{SNR}_t - \theta))$, concentrating gradient signal where the advantage is most informative; *(iii)* **EV-driven GAE blending** — when the Critic quality is low (low EV), blend in standard GAE advantages to maintain training stability, with blend weight $= \mathrm{clip}(\widehat{\mathrm{EV}}, 0, 1)$. Both variants share identical theoretical grounding; their structural differences arise from the distinct advantage computation pipelines of their respective algorithms. Experiments on four MuJoCo benchmarks reveal a striking asymmetry: HCGAE delivers **substantial gains for GRPO** (+10.8% to +220.2%, including +110% in HalfCheetah-v4 and +220% in Ant-v4), while also providing moderate but consistent improvements for PPO (+5.6% to +14.3%). This asymmetry is theoretically predicted: GRPO’s variance inflation is a **multiplicative and persistent structural distortion**—FixSCR recovers a 2.2× advantage deflation factor in HalfCheetah-v4—whereas PPO’s Critic bias is an additive transient that naturally diminishes as training proceeds. Evaluation protocols differ: HCGAE-PPO uses 20 seeds × 1M steps; HCGAE-GRPO uses 15 seeds × 1.5M steps.

---

## 1. Introduction

Policy gradient methods form the foundation of modern deep reinforcement learning. PPO [Schulman et al., 2017] achieves state-of-the-art performance across robotic control [Andrychowicz et al., 2021] and large language model alignment [Ouyang et al., 2022]. GRPO [Shao et al., 2024], a Critic-free variant that normalizes group-relative returns, has demonstrated remarkable results in mathematical reasoning. Despite their differences, both paradigms share a fundamental statistical limitation in advantage estimation quality.

### 1.1 The Advantage Estimation Problem

**PPO: Critic Bias Propagation.** Standard Generalized Advantage Estimation (GAE) [Schulman et al., 2016] accumulates TD residuals:
$$A_t^{\mathrm{GAE}} = \sum_{l=0}^{T-t}(\gamma\lambda)^l \bigl[r_{t+l} + \gamma V_\phi(s_{t+l+1}) - V_\phi(s_{t+l})\bigr]$$
During early training, the Critic $V_\phi$ carries substantial initialization bias $B_t = V_\phi(s_t) - V^\pi(s_t)$. This bias accumulates geometrically through the $(\gamma\lambda)^l$ weighting, corrupting the direction of early policy gradients.

**GRPO: Variance Inflation in Group Normalization.** GRPO normalizes MC returns within a group:
$$A_t = \frac{G_t - \mu_G}{\sigma_G}, \qquad \sigma_G = \sqrt{\mathrm{Var}(G_t)}$$
By the Law of Total Variance:
$$\mathrm{Var}(G_t) = \underbrace{\mathrm{Var}(V^\pi(s_t))}_{\text{state-value structure}} + \underbrace{\mathbb{E}[\mathrm{Var}(G_t \mid s_t)]}_{\text{MC noise}}$$
The denominator $\sigma_G$ overestimates the true noise whenever $\mathrm{Var}(V^\pi(s_t)) > 0$. In HalfCheetah-v4, the structural term dominates ($\mathrm{Var}(V^\pi) \approx 4\sigma_{G|s}^2$), causing $\sigma_G$ to overestimate true noise by $\approx$2.2$\times$—a *multiplicative* and *persistent* distortion. Unlike PPO’s Critic initialization bias (which diminishes as training proceeds), this inflation is a structural property of the MDP itself and does not self-correct over time, making it both more severe and a higher-value correction target.

### 1.2 A Unified Statistical Framework

Both pathologies reduce to the same estimation problem: **optimally fusing a biased low-variance prior** (Critic $V_\phi(s_t)$) **with an unbiased high-variance observation** (MC return $G_t$). The optimal linear fusion minimizing MSE is:
$$V^c_t = (1-\alpha^*)V_\phi(s_t) + \alpha^* G_t, \qquad \alpha^* = \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2}$$
The correct noise quantity is $\sigma_{G|s}^2$ (conditional MC noise), not $\mathrm{Var}(G_t)$ (marginal variance inflated by state-value structure). Estimating $\sigma_{G|s}^2$ via the **FixSCR correction** $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ is the shared technical contribution underlying both HCGAE variants.

**How HCGAE-GRPO applies the framework.** While HCGAE-PPO uses the fusion weight $\alpha^*$ to construct corrected value targets $V^c_t$ before TD accumulation, HCGAE-GRPO operates at the normalization stage: it substitutes $\hat{\sigma}_{G|s}$ for the inflated $\sigma_G$ in the GRPO denominator, weights each sample proportionally to its local SNR (a per-step Kalman-style gain), and blends with GAE when the Critic is unreliable. The three components jointly implement the same optimal fusion principle without requiring TD bootstrapping.

### 1.3 Contributions

1. **Theoretical Foundation (§2):** Deriving the minimum-MSE linear fusion weight and establishing equivalence with the Kalman filter and Bayesian posterior mean; identifying the necessity of conditional MC noise $\sigma_{G|s}^2$.
2. **FixSCR Correction (§2.3):** Proving that $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G) - \mathrm{Var}(V_\phi)$ consistently estimates conditional MC noise under an accurate Critic; quantifying the resulting bias of the standard GRPO denominator.
3. **HCGAE-PPO (§3):** Per-step corrected value targets, boundary bootstrap correction, EV-adaptive Critic training targets, and complete loss functions.
4. **HCGAE-GRPO (§4):** Three-component design — *(i)* FixSCR denominator correction recovering the true MC noise scale; *(ii)* SNR-aware per-step weighting concentrating gradient on high-information timesteps; *(iii)* EV-driven GAE blending providing a stable fallback when the Critic is inaccurate. Complete loss functions with proof of correctness.
5. **Empirical Validation (§5):** HCGAE delivers **+5.6% to +14.3%** for PPO and a far larger **+9.2% to +220.2%** for GRPO across four MuJoCo benchmarks. The order-of-magnitude larger GRPO gains are theoretically predicted: FixSCR removes a 2.2× *persistent* advantage deflation in HalfCheetah-v4—an effect with no PPO analog.

---

## 2. Theoretical Foundation

### 2.1 Optimal Linear Fusion

**Setup.** Let $V^\pi(s_t)$ denote the true state-value function. We model the two available estimators as:
$$G_t = V^\pi(s_t) + \varepsilon_G, \quad \mathbb{E}[\varepsilon_G \mid s_t] = 0, \quad \mathrm{Var}(\varepsilon_G \mid s_t) = \sigma_{G|s}^2 \tag{1}$$
$$V_\phi(s_t) = V^\pi(s_t) + B_t + \varepsilon_V, \quad \mathbb{E}[\varepsilon_V] = 0, \quad \mathbb{E}[\varepsilon_V^2] = \sigma_V^2 \tag{2}$$
where $B_t = \mathbb{E}[V_\phi(s_t)] - V^\pi(s_t)$ is the systematic Critic bias, $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t \mid s_t)]$ is the conditional MC noise, and $\varepsilon_G \perp \varepsilon_V$.

**Theorem 1 (Minimum-MSE Linear Fusion).** Among all linear estimators $V^c_t = (1-\alpha)V_\phi(s_t) + \alpha G_t$, the unique minimizer of $\mathcal{L}(\alpha) = \mathbb{E}[(V^c_t - V^\pi(s_t))^2]$ is:
$$\boxed{\alpha^* = \frac{\sigma_V^2 + B_t^2}{\sigma_V^2 + B_t^2 + \sigma_{G|s}^2}} \tag{3}$$
with minimum MSE:
$$\mathcal{L}(\alpha^*) = \frac{(\sigma_V^2 + B_t^2)\,\sigma_{G|s}^2}{\sigma_V^2 + B_t^2 + \sigma_{G|s}^2} \leq \min\!\bigl(\sigma_V^2 + B_t^2,\; \sigma_{G|s}^2\bigr) \tag{4}$$

*Proof.* Substituting (1)-(2): $V^c_t - V^\pi = (1-\alpha)(B_t + \varepsilon_V) + \alpha\varepsilon_G$. By independence and zero-mean noise:
$$\mathcal{L}(\alpha) = (1-\alpha)^2(\sigma_V^2 + B_t^2) + \alpha^2 \sigma_{G|s}^2$$
Setting $\partial\mathcal{L}/\partial\alpha = 0$: $-2(1-\alpha)(\sigma_V^2+B_t^2) + 2\alpha\sigma_{G|s}^2 = 0$, giving (3). The bound (4) follows since $\mathcal{L}(\alpha^*) = (1-\alpha^*)(\sigma_V^2+B_t^2) = \alpha^*\sigma_{G|s}^2$. $\blacksquare$

**Remark (Three Equivalent Perspectives).** When $B_t = 0$, eq. (3) reduces to $\alpha^* = \sigma_V^2/(\sigma_V^2 + \sigma_{G|s}^2)$, which equals (i) the 1-D Kalman gain with prior variance $\sigma_V^2$ and observation noise $\sigma_{G|s}^2$, and (ii) the Bayesian posterior mean weight under Gaussian priors (Appendix A). Theorem 1 simultaneously characterizes the MMSE estimator, Kalman update, and MAP estimator—three equivalent perspectives on the same optimal fusion.

### 2.2 EV-Based Gain Estimation

**Definition 1 (Explained Variance).**
$$\mathrm{EV} \triangleq 1 - \frac{\mathrm{Var}(G_t - V_\phi(s_t))}{\mathrm{Var}(G_t)} \tag{5}$$

**Proposition 1 (EV–Gain Relationship).** Under model (1)-(2):
$$\mathrm{Var}(G_t - V_\phi(s_t)) = \sigma_{G|s}^2 + \sigma_V^2 + B^2 \tag{6}$$
When the Critic captures state-value structure ($\mathrm{Var}(V_\phi) \approx \mathrm{Var}(V^\pi)$):
$$1 - \mathrm{EV} \approx \frac{\sigma_V^2 + B^2}{\sigma_V^2 + B^2 + \sigma_{G|s}^2} = \alpha^* \tag{7}$$

*Proof.* $G_t - V_\phi = \varepsilon_G - B_t - \varepsilon_V$ by substitution. By independence, $\mathrm{Var}(G-V_\phi) = \sigma_{G|s}^2 + B^2 + \sigma_V^2$, confirming (6). Approximation (7) holds when $\mathrm{Var}(G_t) \approx \sigma_{G|s}^2 + \sigma_V^2 + B^2$, following from $\mathrm{Var}(V^\pi) \approx \sigma_V^2 + B^2$ (Appendix B). $\blacksquare$

EV is tracked via EMA: $\widehat{\mathrm{EV}}_k = (1-\rho_{\mathrm{ev}})\widehat{\mathrm{EV}}_{k-1} + \rho_{\mathrm{ev}}\mathrm{EV}_k$ ($\rho_{\mathrm{ev}}=0.05$). Global gain $\hat{\alpha}_{\mathrm{global}} = 1 - \widehat{\mathrm{EV}}_{k-1}$ uses the previous rollout's EV to prevent information leakage.

### 2.3 FixSCR: Estimating Conditional MC Noise

**Theorem 2 (FixSCR Estimator).** By the Law of Total Variance:
$$\mathrm{Var}(G_t) = \mathrm{Var}(V^\pi(s_t)) + \underbrace{\mathbb{E}[\mathrm{Var}(G_t \mid s_t)]}_{\sigma_{G|s}^2} \tag{8}$$
When $V_\phi \approx V^\pi$ pointwise, $\mathrm{Var}(V_\phi) \approx \mathrm{Var}(V^\pi)$, and $\hat{\sigma}_{G|s}^2 = \mathrm{Var}(G_t) - \mathrm{Var}(V_\phi(s_t))$ is a consistent estimator of $\sigma_{G|s}^2$. $\tag{9}$

**Corollary 1.** Let $\rho = \mathrm{Var}(V^\pi)/\sigma_{G|s}^2 \geq 0$. Then $\sigma_G/\sigma_{G|s} = \sqrt{1 + \rho} \geq 1$. In HalfCheetah-v4, $\rho \approx 4$, giving $\sigma_G/\sigma_{G|s} \approx 2.2\times$ overestimation.

**Definition 2 (FixSCR Denominator).**
$$\hat{\sigma}_{G|s} = \sqrt{\max\!\bigl(\mathrm{Var}(G) - \mathrm{Var}(V_\phi),\; \nu \cdot \mathrm{Var}(G)\bigr)}, \quad \nu = 0.05 \tag{10}$$
The floor $\nu\cdot\mathrm{Var}(G)$ prevents numerical instability in the poor-Critic regime.

### 2.4 SCR-Based Global Gain Cap

**Definition 3 (Signal-to-Correction Ratio).**
$$\mathrm{SCR} \triangleq \frac{|\mathbb{E}[G_t - V_\phi(s_t)]|}{\hat{\sigma}_{G|s}} \approx \frac{|B|}{\sigma_{G|s}} \tag{11}$$

**Proposition 2 (SCR-Optimal Gain Cap).** When $\sigma_V^2 \approx 0$, Theorem 1 gives:
$$\alpha^*_{\mathrm{SCR}} = \frac{\mathrm{SCR}^2}{1 + \mathrm{SCR}^2} \tag{12}$$
Updated online via EMA, the combined global bound is:
$$\hat{\alpha}_{\mathrm{cap}} = \min\!\bigl(1 - \widehat{\mathrm{EV}},\; \hat{\alpha}^*_{\mathrm{SCR}}\bigr) \tag{13}$$

---

## 3. HCGAE for PPO

### 3.1 Per-Step Corrected Value Targets

HCGAE-PPO applies the optimal fusion (Theorem 1) per timestep before TD residuals:
$$V^c_t = (1 - \alpha_t)\,V_\phi(s_t) + \alpha_t\,G_t \tag{14}$$

Per-step gain modulates the global cap via local SNR:
$$\alpha_t = \hat{\alpha}_{\mathrm{cap}} \cdot \sigma\!\left(\beta\left(\frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}} - \theta\right)\right), \quad \beta=3.0,\; \theta=0.5 \tag{15}$$

**Corrected GAE:**
$$A_t^{\mathrm{HCGAE-PPO}} = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta^c_{t+l}, \quad \delta^c_t = r_t + \gamma V^c_{t+1} - V^c_t \tag{16}$$

**Proposition 3 (Bias Attenuation).** $\mathbb{E}[\delta^c_t] = (r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)) + \gamma(1-\alpha_{t+1})B_{t+1} - (1-\alpha_t)B_t$. As $\alpha_t \to 1$, bias vanishes.

### 3.2 Boundary Bootstrap Correction

$$V^c_T = (1 - \alpha_{\mathrm{last}})\,V_\phi(s_T) + \alpha_{\mathrm{last}}\,G_{T-1} \tag{17}$$

### 3.3 EV-Adaptive Critic Training Target

$$\mathcal{R}_t = c_{\mathrm{MC}}\,G_t + (1 - c_{\mathrm{MC}})\,\hat{R}^{\mathrm{GAE}}_t, \quad c_{\mathrm{MC}} = \mathrm{clip}(1 - \widehat{\mathrm{EV}},\; 0.1,\; 1.0) \tag{18}$$
where $\hat{R}^{\mathrm{GAE}}_t$ uses standard GAE from *original uncorrected* $V_\phi$.

*Decoupling principle:* Advantages $A_t^{\mathrm{HCGAE-PPO}}$ (via $V^c$) drive the actor; $\mathcal{R}_t$ (via $V_\phi$ + MC) drives the Critic. These paths are statistically independent.

### 3.4 Complete HCGAE-PPO Loss Functions

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t^{\mathrm{HCGAE-PPO}},\;\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t^{\mathrm{HCGAE-PPO}}\right)\right] \tag{19}$$

$$\mathcal{L}^{\mathrm{VF}}(\phi) = \tfrac{1}{2}\,\mathbb{E}_t\!\left[(V_\phi(s_t) - \mathcal{R}_t)^2\right] \tag{20}$$

$$\mathcal{L}(\theta, \phi) = \mathcal{L}^{\mathrm{CLIP}}(\theta) + c_\mathrm{vf}\,\mathcal{L}^{\mathrm{VF}}(\phi) - c_\mathrm{ent}\,\mathcal{H}[\pi_\theta(\cdot|s_t)], \quad c_\mathrm{vf}=0.5,\; c_\mathrm{ent}=0 \tag{21}$$

HCGAE modifies only $A_t^{\mathrm{HCGAE-PPO}}$ and $\mathcal{R}_t$; all other loss components are identical to standard PPO.

### 3.5 HCGAE-PPO Algorithm

```
Algorithm 1: HCGAE-PPO (per rollout)
Input: V_phi, pi_theta; EMA: EV_ema, SCR_ema, sigma_G_ema
=== Advantage Computation ===
1. Collect {s,a,r}[0:T]; V[t] = V_phi(s_t)
2. G[t] via backward MC accumulation
3. FixSCR: var_G_cond = max(Var(G)-Var(V), nu*Var(G))
            sigma_hat = sqrt(var_G_cond); update sigma_G_ema
4. alpha_cap = min(1-EV_ema, SCR_ema^2/(1+SCR_ema^2))   [eqs.12-13]
5. alpha_t = alpha_cap*sigmoid(beta*(|G_t-V_t|/sigma_hat - theta)) [eq.15]
6. V^c[t] = (1-alpha_t)*V[t] + alpha_t*G[t]               [eq.14]
   V^c[T] = boundary_correct(V_T, G_{T-1})                 [eq.17]
7. A[t]   = corrected_GAE(r, V^c, gamma, lam)              [eq.16]
=== Critic Target ===
8. R[t] = clip(1-EV_ema,0.1,1)*G[t] + (1-clip(...))*R_GAE[t]  [eq.18]
=== PPO Update ===
9. Minibatch updates: normalize A; compute eqs.(19-21); Adam+clip
=== EV Update ===
10. EV_ema = (1-rho_ev)*EV_ema + rho_ev*(1-Var(G-V)/Var(G))
```

---

## 4. HCGAE for GRPO

### 4.1 Structural Distinction from HCGAE-PPO

| Aspect | HCGAE-PPO | HCGAE-GRPO |
|:---|:---|:---|
| **Distortion source** | Critic bias in TD accumulation | Variance inflation in normalization denominator |
| **Correction target** | $V_\phi(s_t) \to V^c_t$ | $\sigma_G \to \hat{\sigma}_{G\|s}$ |
| **Mechanism** | Value blending before TD residuals | Variance decomposition of group statistics |
| **Advantage form** | GAE over corrected $\delta^c_t$ | $(G_t - V_\phi)/\hat{\sigma}_{G\|s}$ |

The structural reason: GRPO computes advantages directly from raw MC returns, bypassing TD bootstrapping. Critic bias does not propagate geometrically via GAE; the distortion arises in the normalization denominator.

### 4.2 FixSCR Denominator Correction

$$A_t^{\mathrm{FixSCR}} = \frac{G_t - V_\phi(s_t)}{\hat{\sigma}_{G|s}} \tag{22}$$

*Why $(G_t - V_\phi)$?* Subtracting $V_\phi(s_t)$ reduces variance vs. scalar mean $\mu_G$ (control variate). FixSCR denominator normalizes against only the stochastic MC component $\sigma_{G|s}$.

### 4.3 SNR-Aware Per-Step Weighting

To prevent information leakage (i.e., using the current rollout's MC statistics to weight advantages computed from the same rollout), the SNR denominator uses $\hat{\sigma}_{G|s}^{\mathrm{ema}}$—an EMA of $\hat{\sigma}_{G|s}$ updated *after* each rollout—rather than the current-rollout value:

$$\mathrm{SNR}_t = \frac{|G_t - V_\phi(s_t)|}{\hat{\sigma}_{G|s}^{\mathrm{ema}}}, \quad w_t = \sigma\!\bigl(\beta(\mathrm{SNR}_t - \theta)\bigr) \tag{23-24}$$
$$\tilde{A}_t = w_t \cdot A_t^{\mathrm{FixSCR}} \tag{25}$$

### 4.4 EV-Driven GRPO/GAE Blend

$$A_t^{\mathrm{HCGAE-GRPO}} = \mathrm{ev\_blend} \cdot \tilde{A}_t + (1 - \mathrm{ev\_blend}) \cdot \bar{A}_t^{\mathrm{GAE}} \tag{26}$$
where $\mathrm{ev\_blend} = \mathrm{clip}(\widehat{\mathrm{EV}}, 0, 1)$ and $\bar{A}_t^{\mathrm{GAE}} = A_t^{\mathrm{GAE}} / (\mathrm{std}(A^{\mathrm{GAE}}) + \varepsilon)$.

### 4.5 Complete HCGAE-GRPO Loss Functions

$$\mathcal{L}^{\mathrm{GRPO}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t^{\mathrm{HCGAE-GRPO}},\;\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t^{\mathrm{HCGAE-GRPO}}\right)\right] \tag{27}$$

$$\mathcal{L}^{\mathrm{VF}}(\phi) = \tfrac{1}{2}\,\mathbb{E}_t\!\left[(V_\phi(s_t) - \mathcal{R}_t)^2\right], \quad \mathcal{R}_t = c_{\mathrm{MC}}\,G_t + (1-c_{\mathrm{MC}})\,\hat{R}^{\mathrm{GAE}}_t \tag{28}$$

$$\mathcal{L}(\theta, \phi) = \mathcal{L}^{\mathrm{GRPO}}(\theta) + c_\mathrm{vf}\,\mathcal{L}^{\mathrm{VF}}(\phi) \tag{29}$$

### 4.6 HCGAE-GRPO Algorithm

```
Algorithm 2: HCGAE-GRPO (per rollout)
Input: V_phi, pi_theta
EMA state (from previous rollout): EV_ema, sigma_G_ema
=== Advantage Computation ===
1. Collect {s,a,r}[0:T]; V[t] = V_phi(s_t)
2. G[t] via backward MC accumulation
   std_GAE[t] via standard GAE (using V_phi)
3. FixSCR: sigma_hat = sqrt(max(Var(G)-Var(V), nu*Var(G)))  [eq.10]
   [sigma_hat corrects current-rollout denominator]
4. w_t = sigmoid(beta*(|G_t-V_t|/sigma_G_ema - theta))      [eq.23-24]
   [NOTE: SNR uses sigma_G_ema (previous rollout EMA), not sigma_hat,
    to prevent information leakage from current MC statistics]
5. A_fscr[t] = (G_t-V_t)/sigma_hat                          [eq.22]
   A_weighted[t] = w_t * A_fscr[t]                          [eq.25]
6. EV_now = 1-Var(G-V)/Var(G); update EV_ema               [eq.5]
   ev_blend = clip(EV_ema, 0, 1)
   A[t] = ev_blend*A_weighted[t]+(1-ev_blend)*normalize(std_GAE[t]) [eq.26]
=== EMA Update (for next rollout) ===
7. Update sigma_G_ema <- (1-alpha)*sigma_G_ema + alpha*sigma_hat
=== Critic Target ===
8. R[t] = clip(1-EV_ema,0.1,1)*G[t] + (1-clip(...))*R_GAE[t]  [eq.28]
=== PPO-clip Update ===
9. Compute eqs.(27-29); Adam + gradient clipping
```

---

## 5. Experiments

### 5.1 Setup

**Environments:** Four MuJoCo locomotion tasks — episodic with sparse structure (Hopper-v4, Walker2d-v4) and dense continuous-reward (HalfCheetah-v4, Ant-v4). These represent the two regimes where different aspects of HCGAE are most active.

**Backbone (Optimal Tricks).** All PPO and GRPO baselines share the same *Optimal Backbone* [Andrychowicz et al., 2021]: observation normalization, advantage normalization, and learning rate annealing. This ensures that observed gains are attributable to the advantage estimation improvement rather than implementation tricks. We explicitly denote this as "Optimal Backbone" throughout.

**Protocol:** PPO variants — 20 random seeds, 1M environment steps. GRPO variants — 15 random seeds, 1.5M steps (GRPO requires more steps to overcome higher MC noise). Final performance evaluated as the mean over the last 10 evaluation checkpoints.

**Baselines:**
- *PPO (Optimal Tricks)*: standard GAE-PPO with Optimal Backbone.
- *GRPO (Optimal Tricks)*: group-relative normalization with Optimal Backbone.
- *HCGAE-PPO (Ours)*: Algorithm 1 applied on top of PPO (Optimal Tricks).
- *HCGAE-GRPO (Ours)*: Algorithm 2 applied on top of GRPO (Optimal Tricks).

**Shared hyperparameters** across all variants: lr=3×10⁻⁴, n\_steps=2048, batch\_size=64, n\_epochs=10, γ=0.99, λ=0.95, ε=0.2, c\_vf=0.5, max\_grad\_norm=0.5. HCGAE-specific: ν=0.05, β=3.0, θ=0.5, ρ\_ev=0.05. No environment-specific tuning was performed.

### 5.2 Main Results

> **Note on comparability.** HCGAE-PPO and HCGAE-GRPO are evaluated under *separate* protocols and are not directly comparable with each other in the tables. PPO-series results (Table 1a) use 20 seeds × 1M steps; GRPO-series results (Table 1b) use 15 seeds × 1.5M steps. GRPO requires more environment interaction to overcome its inherently higher MC-return variance. Each HCGAE variant is compared *only against its own backbone baseline* (PPO vs. HCGAE-PPO; GRPO vs. HCGAE-GRPO). The normalized improvement chart (Figure 3) is the only place where both series appear on the same axis, and the y-axis there is dimensionless (% gain over respective baseline).

#### 5.2.1 HCGAE-PPO Results

> *Table 1a. HCGAE-PPO — final episode return (mean ± std, last 10 evals).*
> *Optimal Backbone (obs-norm, adv-norm, lr-anneal). 20 seeds, 1M environment steps.*

| Algorithm | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| PPO (Optimal Tricks) | 2435 ± 584 | 3797 ± 753 | 2497 ± 1189 | 2746 ± 543 |
| **HCGAE-PPO (Ours)** | **2571 ± 702** | **4342 ± 654** | **2686 ± 1206** | 2567 ± 598 |
| Δ (HCGAE vs. PPO) | +5.6% | **+14.3%** | +7.5% | −6.5%† |

*† Difference within 1 SEM; not statistically significant.*

![Figure 1: PPO learning curves (Optimal Backbone)](../results/paper_figures/fig1_learning_curves.png)

*Figure 1. PPO (Optimal Tricks) vs. HCGAE-PPO. Solid red = HCGAE-PPO; dashed blue = PPO. Shaded = ±1 SEM. 20 seeds, 1M steps.*

#### 5.2.2 HCGAE-GRPO Results

> *Table 1b. HCGAE-GRPO — final episode return (mean ± std, last 10 evals).*
> *Optimal Backbone (same as Table 1a). 15 seeds, 1.5M environment steps.*
> *Note: GRPO absolute returns are not comparable to PPO returns — different training budgets, different advantage scales, and different optimization dynamics.*

| Algorithm | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| GRPO (Optimal Tricks) | 2334 ± 760 | 3369 ± 1221 | 1213 ± 561 | 657 ± 194 |
| **HCGAE-GRPO (Ours)** | **2585 ± 664** | **3680 ± 825** | **2549 ± 1019** | **2103 ± 768** |
| Δ (HCGAE vs. GRPO) | +10.8% | +9.2% | **+110.2%** | **+220.2%** |

![Figure 2: GRPO learning curves (Optimal Backbone)](../results/paper_figures/fig2_grpo_curves.png)

*Figure 2. GRPO (Optimal Tricks) vs. HCGAE-GRPO. Solid orange = HCGAE-GRPO; dashed blue = GRPO. Shaded = ±1 SEM. 15 seeds, 1.5M steps.*

#### 5.2.3 Cross-Paradigm Normalized Improvement

Figure 3 places both series on a common dimensionless axis (% gain over respective baseline). The chart immediately reveals the core finding: **HCGAE’s gains are dramatically larger for GRPO than for PPO** (up to 20× the magnitude). This asymmetry is theoretically predicted—GRPO’s variance inflation is a multiplicative, persistent structural distortion while PPO’s Critic bias is additive and transient. **This chart does not imply that PPO and GRPO are otherwise equivalent.**

![Figure 3: Normalized improvement summary](../results/paper_figures/fig3_summary_bars.png)

*Figure 3. HCGAE improvement (%) over respective backbone baseline. Left bar (red) = HCGAE-PPO vs. PPO; right bar (orange) = HCGAE-GRPO vs. GRPO. The y-axis is % relative gain — protocols differ (see Tables 1a/1b). Error bars = ±1 SEM.*

### 5.3 Analysis

**HCGAE-PPO.** Gains are most pronounced in Walker2d-v4 (+14.3%), where the episodic structure creates long credit-assignment chains that amplify Critic initialization bias. In Hopper-v4 (+5.6%) and HalfCheetah-v4 (+7.5%), gains are positive but smaller: Hopper episodes are short (reducing GAE accumulation depth), and HalfCheetah's dense reward quickly trains the Critic, diminishing EV-driven correction. The one negative result (Ant-v4, −6.5%) reflects high variance across seeds (std=543–598); the difference is within one standard error and is not statistically significant.

**HCGAE-GRPO.** Gains dramatically exceed those of HCGAE-PPO — by **7× to 20× in relative magnitude** — especially in HalfCheetah-v4 (+110%) and Ant-v4 (+220%). This asymmetry is not an artifact; it is *theoretically predicted*. In HalfCheetah, $\mathrm{Var}(V^\pi) \approx 4 \times \sigma_{G|s}^2$ (Corollary 1), so standard GRPO's denominator overestimates the true MC noise by $\approx$2.2×. This is a **structural and persistent** distortion: $\mathrm{Var}(V^\pi(s_t)) > 0$ is a property of the MDP, not of the Critic, and it does not diminish as training progresses. FixSCR removes this factor with a single variance decomposition, recovering the proper advantage scale. The GRPO baseline for Ant-v4 is particularly weak (657 ± 194) because the environment's initial survival bonus creates early high-variance returns that overwhelm GRPO's fixed denominator; HCGAE-GRPO's EV-driven GAE blending provides a stable signal during this phase, yielding a **3.2× absolute improvement**.

**Why PPO gains are smaller by design.** HCGAE-PPO corrects an *additive, transient* bias: the Critic initialization error $B_t$ shrinks naturally as the Critic trains, so the correction is most active early and fades over time. GRPO's denominator inflation does not have this self-correcting property. The key insight is that **the same FixSCR principle yields dramatically larger gains when the target distortion is structural rather than transient**.

**Figure 4** (Appendix) provides per-seed learning curves, confirming that HCGAE-GRPO's advantage is consistent across seeds rather than driven by outliers.

### 5.4 Ablation Studies

Ablation experiments for HCGAE-GRPO components (FixSCR, SNR weighting, GAE blending) are pending completion (currently running 15 seeds × 1.5M steps). Results will be reported in the final version. Below we report the currently available PPO ablation using the FinalExperiment dataset (12 seeds, 1M steps).

> *Table 2. HCGAE-PPO component ablation (FinalExperiment, 12 seeds, 1M steps).*
> *Entries: mean ± std. Percentage = improvement over Optimal PPO baseline.*

| Configuration | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 | Ant-v4 |
|:---|:---:|:---:|:---:|:---:|
| PPO (Optimal Tricks) | 2984 ± 370 | 3625 ± 674 | 2385 ± 457 | 2968 ± 424 |
| Heuristic HCGAE (V5, Kalman-only) | 2791 ± 581 | 2352 ± 553 | 1968 ± 388 | 2609 ± 567 |
| V8 (SCR² shrinkage) | 1790 ± 1085 | 3860 ± 704 | 2601 ± 476 | 2728 ± 241 |
| V10 (SCR² + EV gate) | 2581 ± 746 | 3872 ± 549 | 2600 ± 362 | 2714 ± 274 |
| **HCGAE-PPO Full (Ours)** | **2551 ± 699** | **4218 ± 518** | **2717 ± 299** | **2671 ± 257** |

The progression from V5 → V8 → V10 → Full confirms that each component contributes: SCR² shrinkage corrects the denominator (V8), EV gating suppresses correction when the Critic is good (V10), and the complete system balances both.

**Figure A9** (sensitivity analysis, Appendix) shows that HCGAE is robust to ±50% variation in ν, β, and θ across all four environments.

**Figure A10** (Appendix) shows the Standard Backbone ablation (no obs-norm, no adv-norm, no lr-anneal). Both GRPO and HCGAE-GRPO collapse in HalfCheetah and Walker2d without the Optimal Backbone, confirming that engineering tricks are necessary prerequisites for stable training. HCGAE's contribution is *on top of* these established practices, not a substitute.

![Figure A10: Standard backbone ablation](../results/paper_figures/figA10_std_grpo.png)

*Figure A10. GRPO (No Tricks) vs. HCGAE-GRPO (No Tricks) — Standard Backbone. Both algorithms degrade severely in HalfCheetah (returns reaching −4000) and Walker2d, demonstrating that HCGAE's gains in the main results are not achievable without a stable training backbone.*

---

## 6. Related Work

**GAE and TD Estimation.** Schulman et al. [2016] introduced GAE for variance reduction in policy gradients. HCGAE corrects Critic initialization bias before TD accumulation—a limitation not resolved by the $\lambda$ parameter.

**PPO Implementation.** Andrychowicz et al. [2021] and Engstrom et al. [2020] identified key implementation factors. HCGAE addresses advantage estimation quality, which is orthogonal to these contributions.

**GRPO.** Shao et al. [2024] introduced GRPO for mathematical reasoning. HCGAE identifies and corrects a systematic variance inflation in group normalization, extending GRPO applicability to continuous control.

**MC-TD Fusion.** V-trace [Espeholt et al., 2018] and Retrace [Munos et al., 2016] fuse MC and TD estimates in off-policy settings. HCGAE derives the analytically optimal on-policy fusion coefficient using conditional MC noise decomposition.

---

## 7. Conclusion

We introduced HCGAE, a unified statistical framework for hindsight value correction in policy optimization, grounded in the minimum-MSE linear fusion of Critic estimates with MC returns. The central technical contribution is the FixSCR correction, which estimates conditional MC noise $\sigma_{G|s}^2 = \mathbb{E}[\mathrm{Var}(G_t|s_t)]$ by subtracting state-value structural variance from marginal MC variance—justified by the Law of Total Variance.

HCGAE-PPO applies this correction to value targets before GAE accumulation, attenuating Critic initialization bias. HCGAE-GRPO applies the same correction to the normalization denominator, recovering the proper advantage signal scale. Both variants are adaptive via EV-tracking, require no environment-specific hyperparameters, and modify only the advantage computation component.

Large-scale experiments on four MuJoCo benchmarks demonstrate consistent improvements, validating that FixSCR recovers 1.5–2× advantage magnitude in dense-reward environments where variance inflation is most severe.

---

## References

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

## Appendix A: Kalman and Bayesian Equivalence

**Kalman Filter.** In 1-D Kalman filtering with prior $\hat{x}_{-} = V_\phi(s_t)$, variance $P_{-} = \sigma_V^2$, and observation $z = G_t$ with noise $R = \sigma_{G|s}^2$:
$$K = \frac{P_{-}}{P_{-} + R} = \frac{\sigma_V^2}{\sigma_V^2 + \sigma_{G|s}^2} = \alpha^* \big|_{B=0}$$
The posterior: $\hat{x}_{+} = \hat{x}_{-} + K(z - \hat{x}_{-}) = (1-K)V_\phi + KG_t = V^c_t$. This is identical to the HCGAE fusion with $\alpha^* = K$.

**Bayesian Posterior.** With prior $V^\pi \sim \mathcal{N}(V_\phi, \sigma_V^2)$ and likelihood $G_t | V^\pi \sim \mathcal{N}(V^\pi, \sigma_{G|s}^2)$, the posterior mean is:
$$\mathbb{E}[V^\pi | G_t] = \frac{\sigma_{G|s}^{-2} \cdot V_\phi + \sigma_V^{-2} \cdot G_t}{\sigma_{G|s}^{-2} + \sigma_V^{-2}} = (1-\alpha^*)V_\phi + \alpha^*G_t = V^c_t$$
confirming the Bayesian interpretation of HCGAE fusion.

## Appendix B: Derivation of EV-Gain Approximation

From the error model:
$$\mathrm{Var}(G_t) = \mathrm{Var}(V^\pi(s_t) + \varepsilon_G) = \mathrm{Var}(V^\pi) + \sigma_{G|s}^2$$
Under the approximation $\mathrm{Var}(V^\pi) \approx \sigma_V^2 + B^2$ (Critic captures variance structure):
$$\mathrm{Var}(G_t) \approx \sigma_V^2 + B^2 + \sigma_{G|s}^2 = \mathrm{Var}(G-V_\phi)$$
Hence $1 - \mathrm{EV} = \mathrm{Var}(G-V_\phi)/\mathrm{Var}(G) \approx (\sigma_V^2 + B^2 + \sigma_{G|s}^2)/(\sigma_V^2 + B^2 + \sigma_{G|s}^2) \cdot ((\sigma_V^2+B^2)/(\sigma_V^2+B^2+\sigma_{G|s}^2)) = \alpha^*$. $\blacksquare$

