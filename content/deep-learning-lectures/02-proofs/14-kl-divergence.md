---
title: "13. KL Divergence — 정의, 성질, NLL과의 관계"
slug: kl-divergence
order: 14
---

# 13. KL Divergence — 정의, 성질, NLL과의 관계

> **출제 근거**: 4주차 ★8 (KL Divergence), 퀴즈 15 (\"KL divergence 직관 (Gaussian)\")
> **시험 출제 방식**: \"Define KL divergence, prove $\mathrm{KL}(p\|q) \geq 0$, and explain its relation to NLL.\"

---

## 1. 왜 시험에 나오는다

- 4주차 ★8. \"두 분포의 차이\" 를 정량화하는 도구.
- NLL ↔ KL ↔ Cross-Entropy 의 통합 시각 ([04, 05] 토픽 보강).
- Jensen 부등식으로 양수성 증명 — 표준 \"증명\" 문제 형식.

---

## 2. 사전 수학

### 2.1 [고2] Jensen 부등식

$f$ 가 **convex** (볼록) 이면:

$$
f\!\left(\sum_i p_i x_i\right) \leq \sum_i p_i f(x_i), \quad \sum_i p_i = 1, \; p_i \geq 0
$$

$f$ 가 **concave** (오목) 이면 부등호 반대.

> 직관: convex 함수는 \"아래로 볼록\". 가중평균의 함수값 ≤ 함수값의 가중평균.

### 2.2 $-\log$ 의 convexity

$-\log x$ 의 2차 미분 $= 1/x^2 > 0$ → convex. 따라서 Jensen:

$$
-\log\!\left(\sum_i p_i x_i\right) \leq \sum_i p_i (-\log x_i)
$$

또는 $\log$ 가 concave 이므로:

$$
\log\!\left(\sum_i p_i x_i\right) \geq \sum_i p_i \log x_i
$$

### 2.3 Entropy 와 Cross-Entropy

확률분포 $p$ 의 entropy:

$$
H(p) = -\sum_i p_i \log p_i
$$

$p$ 와 $q$ 의 cross-entropy:

$$
H(p, q) = -\sum_i p_i \log q_i
$$

> ⚠️ $H(p,q) \neq H(q,p)$ (비대칭).

---

## 3. 정의

이산:

$$
\boxed{\; \mathrm{KL}(p \| q) \;=\; \sum_i p_i \log \frac{p_i}{q_i} \;}
$$

연속:

$$
\mathrm{KL}(p \| q) \;=\; \int p(x) \log \frac{p(x)}{q(x)}\, dx
$$

**Line-by-Line:**

| 기호 | 의미 |
|------|------|
| $p$ | true 분포 (보통 \"data\") |
| $q$ | model 분포 (\"approximation\") |
| $\log p/q$ | true vs model log-ratio |
| $\sum p \cdot \log(p/q)$ | $p$ 에 대한 가중평균 |

> 💡 \"$q$ 로 $p$ 를 근사할 때 잃는 정보량\" — 정보이론적 해석.

---

## 4. 성질

### 4.1 비대칭

$$
\mathrm{KL}(p\|q) \neq \mathrm{KL}(q\|p)
$$

따라서 \"distance\" 가 아닌 \"divergence\".

### 4.2 비음수성 (Gibbs' inequality)

$$
\boxed{\; \mathrm{KL}(p\|q) \geq 0, \quad \text{등호} \iff p = q \;}
$$

#### 증명 (Jensen)

$$
\mathrm{KL}(p\|q) = -\sum_i p_i \log\frac{q_i}{p_i}
$$

$-\log$ convex → Jensen:

$$
-\sum_i p_i \log\frac{q_i}{p_i} \;\geq\; -\log\!\left(\sum_i p_i \cdot \frac{q_i}{p_i}\right) \;=\; -\log\!\left(\sum_i q_i\right) \;=\; -\log 1 = 0
$$

> 마지막에서 $\sum_i q_i = 1$ ($q$ 가 확률분포).

등호: $-\log$ 가 strictly convex 이므로 등호 ⟺ $q_i/p_i$ 가 모든 $i$ 에서 상수 ⟺ $p = q$. ∎

### 4.3 Cross-Entropy 와의 관계

$$
\mathrm{KL}(p\|q) = \sum_i p_i \log p_i - \sum_i p_i \log q_i = -H(p) + H(p, q)
$$

따라서:

$$
\boxed{\; H(p, q) \;=\; H(p) + \mathrm{KL}(p\|q) \;}
$$

> 💡 \"Cross-entropy = entropy + KL divergence\".
> $p$ 가 고정 (data) 이면 $H(p)$ 는 상수 → **$H(p,q)$ 최소화 = $\mathrm{KL}(p\|q)$ 최소화**.

---

## 5. NLL ↔ KL ↔ MSE 통합 시각

[04 NLL→MSE](04_NLL_MSE_Gaussian_유도.md) 의 NLL 최소화는 다음과 동치:

1. **NLL 최소화**: $\arg\min_\theta -\sum_i \log P(y_i\mid x_i, \theta)$
2. **Cross-entropy 최소화**: empirical $\hat p$ 와 model $q_\theta$ 사이 $H(\hat p, q_\theta)$
3. **KL 최소화**: $\mathrm{KL}(\hat p \| q_\theta)$ ($\hat p$ 의 entropy 는 상수)

> 🎯 한 줄: \"MLE = empirical distribution 과 model distribution 사이의 KL 최소화\".

---

## 6. Gaussian 간 KL (퀴즈 15)

$p = \mathcal{N}(\mu_1, \sigma_1^2)$, $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
\mathrm{KL}(p\|q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

특수: $\sigma_1 = \sigma_2 = \sigma$ 면

$$
\mathrm{KL} = \frac{(\mu_1-\mu_2)^2}{2\sigma^2}
$$

→ 평균 차이 제곱의 정량화 (MSE 와 유사한 구조).

---

## 7. 모범 답안 템플릿

```
[Definition]
KL(p ‖ q) = Σ_i p_i log (p_i / q_i)              (discrete)
         = ∫ p(x) log (p(x)/q(x)) dx              (continuous)

[Property — Non-negativity (Gibbs)]
By Jensen's inequality (since -log is convex):
  KL(p‖q) = -Σ_i p_i log(q_i/p_i)
          ≥ -log Σ_i p_i (q_i/p_i)
          = -log Σ_i q_i = -log 1 = 0.
Equality iff q_i/p_i is constant in i, i.e. p = q.

[Relation to cross-entropy]
H(p,q) = -Σ p log q = -Σ p log p + Σ p log(p/q) = H(p) + KL(p‖q).

[Connection to MLE / NLL]
NLL on dataset D corresponds to cross-entropy between the empirical
distribution p̂ and the model q_θ, up to constants:
  -Σ_i log q_θ(x_i) ≈ n · H(p̂, q_θ) = n · [H(p̂) + KL(p̂‖q_θ)].
Since H(p̂) is independent of θ, MLE is equivalent to minimizing
KL(p̂‖q_θ) — \"make the model distribution as close to the data
distribution as possible.\"
```

---

## 8. 자주 틀리는 함정

1. **\"distance\" 라고 답함**: KL은 비대칭, 삼각부등식 무성립 → divergence.
2. **Jensen 적용 시 부등호 방향 실수**: $-\log$ 는 convex → $\sum p f(x) \geq f(\sum p x)$.
3. **$\sum q_i = 1$ 사용 누락**: 양수성 증명의 마지막 핵심.
4. **NLL 과의 관계 누락**: 시험에서 \"왜 ML 이 KL 최소화와 같은가\" 직접 묻는 변형.

---

## 9. 연결 개념

- ↔ [04 NLL→MSE](04_NLL_MSE_Gaussian_유도.md), [05 Cross-Entropy](05_CrossEntropy_Categorical_유도.md): NLL 의 정보이론적 해석
- → [15 Inductive Bias](15_Inductive_Bias_강도.md): KL 이 prior 강도 측정 가능
