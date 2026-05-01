# NTC vs TC 2-Agent Navigation Simulation

This repository studies a central question:

> **Can collaboration improve performance for a given task at state $s$?**

We compare **Trivial Collaboration (TC)** models with **Non-Trivial Collaboration (NTC)** models using distributions over trajectory pairs.

This README is intended as a development document. The definitions below are written to match the current implementation in `simple_ntc_tc_sim_v20.py`.

---

## 1. Simulation Setup

Two agents move toward each other along the x-axis.

Agent 1:

```math
h_0=(-s/2,0), \qquad h_T=(s/2,0)
```

Agent 2:

```math
r_0=(s/2,0), \qquad r_T=(-s/2,0)
```

In the current implementation, $s$ is both the initial separation and the path length.

Each agent selects a trajectory from a structured trajectory library. A trajectory is a sequence of 2D points:

```math
h=(h_1,\ldots,h_T), \qquad r=(r_1,\ldots,r_T)
```

where:

```math
h_t=(x_h(t),y_h(t)), \qquad r_t=(x_r(t),y_r(t))
```

The trajectory library contains straight trajectories, left/right lateral deviations, several lateral profile shapes, and lateral displacement magnitudes from `LATERAL_LEVELS`.

---

## 2. Trajectory Preference Model

Each trajectory receives a preference cost.

For a trajectory $z$, define the maximum lateral deviation:

```math
d_{\max}(z)=\max_t |y_z(t)|
```

The implemented preference cost is:

```math
\phi(z)
=
3 d_{\max}(z)^2
+
12 \max(0,d_{\max}(z)-0.30)^2
+
30 \max(0,d_{\max}(z)-0.60)^2
+
1.4 \sum_t (\Delta y_z(t))^2
+
2.8 \sum_t (\Delta^2 y_z(t))^2
```

where:

```math
\Delta y_z(t)=y_z(t+1)-y_z(t)
```

and:

```math
\Delta^2 y_z(t)=y_z(t+2)-2y_z(t+1)+y_z(t)
```

The marginal trajectory distributions are:

```math
p_h(h_i)
=
\frac{\exp(-\lambda_{\mathrm{pref}}\phi(h_i))}
{\sum_k \exp(-\lambda_{\mathrm{pref}}\phi(h_k))}
```

```math
p_r(r_j)
=
\frac{\exp(-\lambda_{\mathrm{pref}}\phi(r_j))}
{\sum_k \exp(-\lambda_{\mathrm{pref}}\phi(r_k))}
```

In the current code:

```math
\lambda_{\mathrm{pref}}=1.6
```

---

## 3. Models

All models are computed over the finite trajectory library.

Let $h_i$ be a human trajectory sample and $r_j$ be a robot trajectory sample.

---

### 3.1 Independent Model: TC, $p_h p_r$

The independent model assumes the agents do not respond to each other.

```math
\gamma_{\mathrm{ind}}(i,j)=p_h(h_i)p_r(r_j)
```

---

### 3.2 Response Model: Fixed Human Sample

First choose the most likely human trajectory:

```math
h^*=\arg\max_i p_h(h_i)
```

The robot then computes:

```math
q_r^*
=
\arg\min_{q_r}
\left[
\sum_j q_r(j)c(h^*,r_j)
+
\lambda_{\mathrm{resp}}\mathrm{KL}(q_r \| p_r)
\right]
```

The implemented closed-form solution is:

```math
q_r^*(j)
=
\frac{p_r(r_j)\exp(-c(h^*,r_j)/\lambda_{\mathrm{resp}})}
{\sum_k p_r(r_k)\exp(-c(h^*,r_k)/\lambda_{\mathrm{resp}})}
```

The corresponding joint distribution is:

```math
\gamma_{\mathrm{resp\_sample}}(i,j)
=
\delta(i=i^*)q_r^*(j)
```

where $i^*$ indexes $h^*$.

In the current code:

```math
\lambda_{\mathrm{resp}}=0.25
```

---

### 3.3 Response Model: Human Marginal

The robot responds to the full human marginal $p_h$.

First compute the expected cost of each robot trajectory:

```math
\bar c(r_j)
=
\sum_i p_h(h_i)c(h_i,r_j)
```

Then:

```math
q_r^*(j)
=
\frac{p_r(r_j)\exp(-\bar c(r_j)/\lambda_{\mathrm{resp}})}
{\sum_k p_r(r_k)\exp(-\bar c(r_k)/\lambda_{\mathrm{resp}})}
```

The corresponding joint distribution is:

```math
\gamma_{\mathrm{resp\_marg}}(i,j)=p_h(h_i)q_r^*(j)
```

---

### 3.4 NTC: KL(joint)

This model regularizes the joint distribution toward the independent product distribution.

```math
\gamma_{\mathrm{joint}}^*
=
\arg\min_{\gamma}
\left[
\sum_{i,j}\gamma(i,j)c(h_i,r_j)
+
\lambda_{\mathrm{joint}}\mathrm{KL}(\gamma \| p_hp_r)
\right]
```

The implemented closed form is:

```math
\gamma_{\mathrm{joint}}^*(i,j)
=
\frac{p_h(h_i)p_r(r_j)\exp(-c(h_i,r_j)/\lambda_{\mathrm{joint}})}
{\sum_{a,b}p_h(h_a)p_r(r_b)\exp(-c(h_a,r_b)/\lambda_{\mathrm{joint}})}
```

In the current code:

```math
\lambda_{\mathrm{joint}}=0.45
```

---

### 3.5 NTC: KL(marginals)

This is the primary collaborative model.

Let:

```math
\gamma_h(i)=\sum_j \gamma(i,j)
```

and:

```math
\gamma_r(j)=\sum_i \gamma(i,j)
```

The model solves:

```math
\gamma_{\mathrm{marg}}^*
=
\arg\min_{\gamma}
\left[
\sum_{i,j}\gamma(i,j)c(h_i,r_j)
+
\lambda_h\mathrm{KL}(\gamma_h \| p_h)
+
\lambda_r\mathrm{KL}(\gamma_r \| p_r)
\right]
```

In the current implementation, this is solved by iterative multiplicative updates.

In the current code:

```math
\lambda_h=0.30, \qquad \lambda_r=0.30
```

---

## 4. Pointwise Optimizer

The pointwise optimizer is a deterministic comparison against the modal OT solution.

Let $h_{\mathrm{lin}}$ and $r_{\mathrm{lin}}$ be the straight-line trajectories.

Define trajectory deviation scores:

```math
d_h(h_i)
=
\frac{1}{T}\sum_t \|h_i(t)-h_{\mathrm{lin}}(t)\|
```

```math
d_r(r_j)
=
\frac{1}{T}\sum_t \|r_j(t)-r_{\mathrm{lin}}(t)\|
```

The cost matrix and deviation vectors are min-max normalized:

```math
\tilde c(i,j)
=
\frac{c(i,j)-\min_{a,b}c(a,b)}
{\max_{a,b}c(a,b)-\min_{a,b}c(a,b)+\epsilon}
```

```math
\tilde d_h(i)
=
\frac{d_h(i)-\min_a d_h(a)}
{\max_a d_h(a)-\min_a d_h(a)+\epsilon}
```

```math
\tilde d_r(j)
=
\frac{d_r(j)-\min_b d_r(b)}
{\max_b d_r(b)-\min_b d_r(b)+\epsilon}
```

The pointwise objective is:

```math
J_{\mathrm{pair}}(i,j)
=
\tilde c(i,j)
+
\alpha_h \tilde d_h(i)
+
\alpha_r \tilde d_r(j)
```

The pointwise optimum is:

```math
(i_{\mathrm{pair}}^*,j_{\mathrm{pair}}^*)
=
\arg\min_{i,j}J_{\mathrm{pair}}(i,j)
```

The OT modal pair is:

```math
(i_{\gamma}^*,j_{\gamma}^*)
=
\arg\max_{i,j}\gamma_{\mathrm{marg}}^*(i,j)
```

In the current code:

```math
\alpha_h=\lambda_h, \qquad \alpha_r=\lambda_r
```

---

## 5. Metrics

All metrics are computed for each trajectory pair $(h_i,r_j)$.

Expected metric values are computed using the model distribution.

For a joint distribution $\gamma$:

```math
\mathbb{E}_{\gamma}[m]
=
\sum_{i,j}\gamma(i,j)m(h_i,r_j)
```

For the fixed-sample response model:

```math
\mathbb{E}_{q_r}[m]
=
\sum_j q_r(j)m(h^*,r_j)
```

---

### 5.1 Nominal Cost Metric

The nominal cost is also reported as a metric.

Define:

```math
d_t=\|h_t-r_t\|
```

The implemented nominal cost is:

```math
m_{\mathrm{nom}}(h,r)
=
\sum_{t=1}^T
\frac{1}{t}
\left[
3\left(\frac{1}{1+\exp(12(d_t-0.65))}\right)
+
7\exp\left(-\left(\frac{d_t}{0.22}\right)^2\right)
\right]
```

Smaller is better.

---

### 5.2 Number of Collisions

A trajectory pair has one collision if its minimum distance is at most the collision threshold.

The threshold is:

```math
d_{\mathrm{collision}}=0.5
```

Define:

```math
m_{\mathrm{collision}}(h,r)
=
\mathbf{1}
\left[
\min_t \|h_t-r_t\| \leq 0.5
\right]
```

Smaller is better.

---

### 5.3 Minimum Distance to Person

```math
m_{\mathrm{MDP}}(h,r)
=
\min_t \|h_t-r_t\|
```

Larger is better.

---

### 5.4 Average Safety Distance

```math
m_{\mathrm{ASD}}(h,r)
=
\frac{1}{T}\sum_{t=1}^T \|h_t-r_t\|
```

Larger is better.

---

### 5.5 Imbalance

Let:

```math
\ell_h(h)=\max_t |y_h(t)|
```

and:

```math
\ell_r(r)=\max_t |y_r(t)|
```

The imbalance metric is:

```math
m_{\mathrm{imbalance}}(h,r)
=
|\ell_h(h)-\ell_r(r)|
```

Smaller is better.

---

### 5.6 Passing Side Consistency

The current implementation computes passing side consistency using the midpoint of the trajectories.

Let:

```math
t_{\mathrm{mid}}=\lfloor T/2 \rfloor
```

Define:

```math
s_h=\mathrm{sign}(y_h(t_{\mathrm{mid}}))
```

and:

```math
s_r=\mathrm{sign}(y_r(t_{\mathrm{mid}}))
```

where the implemented sign function returns $0$ when the value is sufficiently close to zero.

The current implemented PSC metric is:

```math
m_{\mathrm{PSC}}(h,r)
=
-s_hs_r
```

Larger is better.

Interpretation:

- $m_{\mathrm{PSC}}=1$: agents pass on opposite sides
- $m_{\mathrm{PSC}}=0$: at least one midpoint is on the centerline
- $m_{\mathrm{PSC}}=-1$: agents pass on the same side

Development note: a time-averaged PSC may be preferable, but this README documents the current implementation.

---

### 5.7 Path Efficiency

For a trajectory $z$, define path length:

```math
L(z)=\sum_{t=1}^{T-1}\|z_{t+1}-z_t\|
```

and straight-line distance:

```math
D(z)=\|z_T-z_1\|
```

The pairwise path efficiency metric is:

```math
m_{\mathrm{eff}}(h,r)
=
\frac{1}{2}
\left[
\frac{D(h)}{L(h)}
+
\frac{D(r)}{L(r)}
\right]
```

Larger is better.

---

### 5.8 Control Effort

Let the discrete second difference be:

```math
a_z(t)=z_{t+2}-2z_{t+1}+z_t
```

The implemented control effort metric is:

```math
m_{\mathrm{ctrl}}(h,r)
=
\sum_t \|a_h(t)\|^2
+
\sum_t \|a_r(t)\|^2
```

Smaller is better.

---

## 6. Cost Functions

Every optimization cost is defined so that smaller is better.

The implemented costs are derived from the metrics above.

---

### 6.1 Nominal Cost

```math
c_{\mathrm{nom}}(h,r)
=
m_{\mathrm{nom}}(h,r)
```

---

### 6.2 Collision Cost

```math
c_{\mathrm{collision}}(h,r)
=
m_{\mathrm{collision}}(h,r)
```

---

### 6.3 MDP Cost

Since MDP is larger-is-better, the optimization cost is:

```math
c_{\mathrm{MDP}}(h,r)
=
-m_{\mathrm{MDP}}(h,r)
```

---

### 6.4 ASD Cost

Since ASD is larger-is-better, the optimization cost is:

```math
c_{\mathrm{ASD}}(h,r)
=
-m_{\mathrm{ASD}}(h,r)
```

---

### 6.5 Imbalance Cost

```math
c_{\mathrm{imbalance}}(h,r)
=
m_{\mathrm{imbalance}}(h,r)
```

---

### 6.6 PSC Cost

Since PSC is larger-is-better, the optimization cost is:

```math
c_{\mathrm{PSC}}(h,r)
=
\frac{1-m_{\mathrm{PSC}}(h,r)}{2}
```

---

### 6.7 Control Effort Cost

```math
c_{\mathrm{ctrl}}(h,r)
=
m_{\mathrm{ctrl}}(h,r)
```

---

### 6.8 Combined Cost

The combined cost averages normalized versions of seven costs:

```math
c_{\mathrm{combined}}(h,r)
=
\frac{1}{7}
\left[
\tilde c_{\mathrm{nom}}
+
\tilde c_{\mathrm{collision}}
+
\tilde c_{\mathrm{MDP}}
+
\tilde c_{\mathrm{ASD}}
+
\tilde c_{\mathrm{imbalance}}
+
\tilde c_{\mathrm{PSC}}
+
\tilde c_{\mathrm{ctrl}}
\right]
```

Each component is min-max normalized over the current trajectory-pair library:

```math
\tilde c_k(i,j)
=
\frac{c_k(i,j)-\min_{a,b}c_k(a,b)}
{\max_{a,b}c_k(a,b)-\min_{a,b}c_k(a,b)+\epsilon}
```

Path efficiency is reported as a metric but is not included as a cost in the combined cost.

---

## 7. Larger-Is-Better and Smaller-Is-Better Metrics

Some metrics are better when larger:

- MDP
- ASD
- PSC
- path efficiency

Some metrics are better when smaller:

- nominal cost
- number of collisions
- imbalance
- control effort

This matters because collaboration benefit is always signed so that positive means collaboration helps.

---

## 8. Collaboration Benefit

The primary model is NTC: KL(marginals). Each baseline is compared against it.

For larger-is-better metrics:

```math
\Delta E_{\mathrm{model}}[m]
=
\mathbb{E}_{\gamma_{\mathrm{marg}}^*}[m]
-
\mathbb{E}_{\mathrm{model}}[m]
```

For smaller-is-better metrics:

```math
\Delta E_{\mathrm{model}}[m]
=
\mathbb{E}_{\mathrm{model}}[m]
-
\mathbb{E}_{\gamma_{\mathrm{marg}}^*}[m]
```

Therefore:

- $\Delta E>0$: NTC: KL(marginals) improves performance over the baseline
- $\Delta E=0$: no difference
- $\Delta E<0$: baseline is better

---

## 9. Plots

### Metric Pages

The metric pages plot collaboration benefit.

- x-axis: $s$
- y-axis: $\Delta E$
- one panel per metric
- one curve per baseline comparison

All metric pages use the same sign convention:

- positive means collaboration helps
- negative means the baseline is better

---

### Pointwise vs OT Pages

These plots compare the pointwise deterministic optimizer to the OT modal pair.

The plotted value is:

```math
m(h_{i_{\mathrm{pair}}^*},r_{j_{\mathrm{pair}}^*})
-
m(h_{i_{\gamma}^*},r_{j_{\gamma}^*})
```

This plot is not a collaboration benefit plot. It compares two selected trajectory pairs.

---

### Movies

Each movie frame corresponds to one value of $s$.

Top panels:

- green: top $K$ samples under the relevant distribution
- red: OT mode, $\arg\max_{i,j}\gamma_{\mathrm{marg}}^*(i,j)$
- black: pointwise optimum, $\arg\min_{i,j}J_{\mathrm{pair}}(i,j)$

Bottom panels:

- collaboration benefit curves
- vertical red line shows the current value of $s$

---

## 10. YAML Configuration

The YAML file controls which costs and outputs are generated.

Example:

```yaml
costs_to_run:
  - C_NOMINAL

movie_costs:
  - C_NOMINAL

s_min: 0.5
s_max: 10.0
s_step: 0.5

make_metric_pages: true
make_pointwise_vs_ot_pages: true
make_snapshot_pngs: false
make_movies: true

parallel: false
max_workers:
```

Available costs:

```yaml
C_NOMINAL
C_NUM_COLLISIONS
C_MDP
C_ASD
C_IMBALANCE
C_PSC
C_CONTROL_EFFORT
C_COMBINED
```

Available metrics:

```yaml
NOMINAL_COST
NUM_COLLISIONS
MDP
ASD
IMBALANCE
PSC
PATH_EFF
CONTROL_EFFORT
```
