# NTC vs TC 2 Agent Navigation Simulation

This repository studies a central question:

> **Can collaboration improve performance for a given task at state $s$?**

We compare **Trivial Collaboration (TC)** models with **Non-Trivial Collaboration (NTC)** models using joint distributions.

---

# Simulation Setup

Two agents move toward each other:

- Agent 1: $(-s/2, 0) \rightarrow (s/2, 0)$  
- Agent 2: $(s/2, 0) \rightarrow (-s/2, 0)$  

where:

- $s$ = initial separation **and** path length

Each agent selects from a trajectory library with prior:

$$
p_h(h) \propto \exp(-\phi(h)), \quad p_r(r) \propto \exp(-\phi(r))
$$

---

# Models

## Independent (TC)

$$
\gamma(h,r) = p_h(h)p_r(r)
$$

---

## Response (fixed sample)

$$
h^ = \arg\max_h p_h(h)
$$

$$
q_r^*(r) \propto p_r(r)\exp\left(-\frac{c(h^,r)}{\lambda}\right)
$$

---

## Response (marginalized)

$$
q_r^*(r) \propto p_r(r)\exp\left(-\frac{\mathbb{E}_{p_h}[c(h,r)]}{\lambda}\right)
$$

---

## NTC: KL(joint)

$$
\gamma^*(h,r) \propto p_h(h)p_r(r)\exp\left(-\frac{c(h,r)}{\lambda}\right)
$$

---

## NTC: KL(marginals)

$$
\gamma^* =
\arg\min_\gamma
\left[
\mathbb{E}_\gamma[c(h,r)]
+ \lambda_h \mathrm{KL}(\gamma_h || p_h)
+ \lambda_r \mathrm{KL}(\gamma_r || p_r)
\right]
$$

---

# Costs (examples)

Let $d_t = \|h_t - r_t\|$

### Nominal
$$
c_{\text{nom}} = \sum_t w_t \left[\frac{1}{1 + e^{k(d_t - d_0)}} + e^{-d_t^2/\sigma^2}\right]
$$

### Collision
$$
c_{\text{coll}} = \mathbf{1}[\min_t d_t < d_{\text{thresh}}]
$$

### MDP
$$
c_{\text{MDP}} = -\min_t d_t
$$

### ASD
$$
c_{\text{ASD}} = -\frac{1}{T}\sum_t d_t
$$

---

# Metrics

We evaluate:

- MDP (minimum distance)  
- ASD (average distance)  
- collisions  
- imbalance  
- PSC (passing consistency)  
- efficiency  

Expected value:

$$
\mathbb{E}_\gamma[\text{metric}] = \sum_{h,r} \gamma(h,r)\,\text{metric}(h,r)
$$

---

# Collaboration Benefit

All plots show:

$$
\Delta E
$$

Defined as:

- larger-is-better metrics:
$$
\Delta E = E_{\text{NTC}} - E_{\text{baseline}}
$$

- smaller-is-better metrics:
$$
\Delta E = E_{\text{baseline}} - E_{\text{NTC}}
$$

---

## Interpretation

- $\Delta E > 0$ → collaboration improves performance  
- $\Delta E = 0$ → no benefit  
- $\Delta E < 0$ → baseline better  

---

# Plots

## Metric plots

- x-axis: $s$  
- y-axis: $\Delta E$  
- gray band ≈ negligible benefit  

---

## Pointwise vs OT

$$
\text{metric}(h^_{pair}, r^_{pair})
-
\text{metric}(h^_\gamma, r^_\gamma)
$$

---

## Movies

Top:
- green: OT samples  
- red: OT mode  
- black: pointwise optimum  

Bottom:
- collaboration curves  
- vertical line = current $s$

---

# Config (YAML)

```yaml
s_min: 0.5
s_max: 10
s_step: 0.5