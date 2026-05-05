---
title: 'Quiz 8 — Gaussian KL → MSE'
description: '가우시안 KL divergence와 MSE 동치성 증명'
draft: false
---

## 0. 한 줄 요약

두 1변량 정규분포의 KL Divergence 를 정의 $\mathbb{E}_{x\sim p}[\log p - \log q]$ 로부터 직접 유도하면 $\log\dfrac{\sigma_2}{\sigma_1} + \dfrac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \dfrac{1}{2}$ 이고, **분산이 같다**는 가정 $\sigma_1=\sigma_2=\sigma$ 하에서 이는 $\dfrac{(\mu_1-\mu_2)^2}{2\sigma^2}$ 로 환원되며, 이 값은 $\mu_1=\mu_2$ 일 때만 0 이 되어 KL 의 전역 최솟값에 도달한다 — 이로부터 **MSE = Gaussian-NLL = KL** 의 동치 사슬이 성립함을 본다.

---

## 1. 문제 (정확한 출제 형태)

두 1변량 정규분포
$$
p(x) = \mathcal{N}(x;\,\mu_1,\,\sigma_1^2), \qquad q(x) = \mathcal{N}(x;\,\mu_2,\,\sigma_2^2)
$$
에 대하여,

**(Q1)** KL Divergence 의 정의 $D_{KL}(p\Vert q) = \mathbb{E}_{x\sim p}[\log p(x) - \log q(x)]$ 로부터 직접 유도하라.

**(Q2)** 분산이 같다는 가정 $\sigma_1 = \sigma_2 = \sigma$ 하에서 위 식을 단순화하라.

**(Q3)** 단순화된 식으로부터 $\mu_1 = \mu_2$ 일 때만 $D_{KL} = 0$ 이며, 이것이 KL 의 최솟값임을 보여라.

---

## 2. 출제 의도와 시험 가치

이 문제는 동시에 세 가지 능력을 측정한다.

1. **정의 적용 능력**: 슬라이드의 결과식을 외우는 것이 아니라, 정의 $\mathbb{E}_{x\sim p}[\log p - \log q]$ 에서 출발해 적분 / 기댓값 계산을 실제로 수행할 수 있는가.
2. **단순화 (reduction)**: 일반식 → 특수 케이스로 환원할 때 어떤 항이 살아남고 어떤 항이 사라지는지를 다룰 수 있는가.
3. **분포 일치 ⇔ 거리 0 ⇔ 최소화** 라는 KL 의 본질을 구체적 케이스에서 확인.

또한 결과식 $\dfrac{(\mu_1-\mu_2)^2}{2\sigma^2}$ 은 곧 MSE / 2σ² 이다. 즉 "왜 회귀 문제에서 MSE 손실을 쓰는가?" 라는 질문의 정량적 근거 — Gaussian 가정 하에 KL 최소화 = NLL 최소화 = MSE 최소화 = MLE — 가 이 한 문제 안에 응축되어 있다.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 정규분포 PDF

$$
\mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

로그:
$$
\log \mathcal{N}(x;\mu,\sigma^2) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

### 3.2 KL Divergence 정의

$$
D_{KL}(p\Vert q) = \mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right] = \mathbb{E}_{x\sim p}[\log p(x) - \log q(x)]
$$

### 3.3 기댓값 도구

- 평균 정의: $\mathbb{E}_{x\sim p}[x] = \mu_1$.
- 분산 정의: $\mathbb{E}_{x\sim p}[(x-\mu_1)^2] = \sigma_1^2$.
- 선형성: $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$.
- 상수의 기댓값: $\mathbb{E}[c] = c$.

### 3.4 핵심 보조 등식

$x \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 일 때
$$
\mathbb{E}_{x\sim p}[(x-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2
$$

(분산-평균 분해. 유도는 4.1 Step 3 참조.)

### 3.5 Gibbs 부등식

$D_{KL}(p\Vert q) \ge 0$ 이고 등호는 거의 모든 $x$ 에서 $p(x) = q(x)$ 일 때만 성립.

### 3.6 미분 도구 (검증용)

- 2차함수 $f(t) = at^2 + bt + c$ 의 최솟값: $a > 0$ 이면 $t = -b/(2a)$.
- Strictly convex 판정: $f''(t) > 0$ 이면 유일한 전역 최솟값.

---

## 4. 풀이 (모든 단계, 등호 근거)

### 4.1 (Q1) — KL 정의로부터 직접 유도

#### Step 1. 두 PDF 의 로그

$$
\log p(x) = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x-\mu_1)^2}{2\sigma_1^2}
$$
$$
\log q(x) = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

근거: $\log\frac{1}{\sqrt{2\pi\sigma^2}} = -\frac{1}{2}\log(2\pi\sigma^2)$, $\log\exp(z) = z$.

#### Step 2. 차이 (상수항 정리)

$$
\log p - \log q = \tfrac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

상수항: $\log(2\pi)$ 가 상쇄되고 $-\frac{1}{2}\log\sigma_1^2 + \frac{1}{2}\log\sigma_2^2 = \log\frac{\sigma_2}{\sigma_1}$.

$$
\log p - \log q = \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

#### Step 3. $\mathbb{E}_{x\sim p}$ 적용

선형성으로 항별 분리.

**(i)** 상수항: $\mathbb{E}\!\left[\log\frac{\sigma_2}{\sigma_1}\right] = \log\frac{\sigma_2}{\sigma_1}$.

**(ii)** 첫 번째 제곱항: $\mathbb{E}\!\left[\frac{(x-\mu_1)^2}{2\sigma_1^2}\right] = \frac{\sigma_1^2}{2\sigma_1^2} = \frac{1}{2}$.

근거: $p = \mathcal{N}(\mu_1,\sigma_1^2)$ 이므로 $\mathbb{E}[(x-\mu_1)^2]$ 는 정확히 분산 $\sigma_1^2$.

**(iii)** 두 번째 제곱항: $(x-\mu_2)^2 = \bigl((x-\mu_1) + (\mu_1-\mu_2)\bigr)^2$ 로 전개.

$$
(x-\mu_2)^2 = (x-\mu_1)^2 + 2(x-\mu_1)(\mu_1-\mu_2) + (\mu_1-\mu_2)^2
$$

기댓값:
$$
\mathbb{E}[(x-\mu_2)^2] = \underbrace{\mathbb{E}[(x-\mu_1)^2]}_{\sigma_1^2} + 2(\mu_1-\mu_2)\underbrace{\mathbb{E}[x-\mu_1]}_{0} + (\mu_1-\mu_2)^2
$$
$$
= \sigma_1^2 + (\mu_1-\mu_2)^2
$$

따라서
$$
\mathbb{E}\!\left[\frac{(x-\mu_2)^2}{2\sigma_2^2}\right] = \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2}
$$

#### Step 4. 합산 (일반식)

$$
\boxed{\;D_{KL}\!\left(\mathcal{N}(\mu_1,\sigma_1^2)\,\Vert\,\mathcal{N}(\mu_2,\sigma_2^2)\right) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}\;}
$$

---

### 4.2 (Q2) — 분산 동일 가정 단순화

$\sigma_1 = \sigma_2 = \sigma$ 대입.

- (i) $\log\frac{\sigma}{\sigma} = \log 1 = 0$.
- (ii) $\frac{\sigma^2 + (\mu_1-\mu_2)^2}{2\sigma^2} = \frac{1}{2} + \frac{(\mu_1-\mu_2)^2}{2\sigma^2}$.
- (iii) $-\frac{1}{2}$ 그대로.

합산: $\frac{1}{2}$ 와 $-\frac{1}{2}$ 가 상쇄.

$$
\boxed{\;D_{KL}\!\left(\mathcal{N}(\mu_1,\sigma^2)\,\Vert\,\mathcal{N}(\mu_2,\sigma^2)\right) = \frac{(\mu_1-\mu_2)^2}{2\sigma^2}\;}
$$

특수 케이스 $\sigma = 1$:
$$
D_{KL} = \frac{(\mu_1-\mu_2)^2}{2}
$$

---

### 4.3 (Q3) — $\mu_1 = \mu_2$ 가 KL 의 전역 최솟값 0

#### Step 1. 비음성

$(\mu_1-\mu_2)^2 \ge 0$ 이고 $\sigma^2 > 0$ 이므로
$$
D_{KL} = \frac{(\mu_1-\mu_2)^2}{2\sigma^2} \ge 0
$$
(이는 Gibbs 부등식 $D_{KL} \ge 0$ 의 구체적 확인.)

#### Step 2. 등호 조건

$$
D_{KL} = 0 \iff (\mu_1-\mu_2)^2 = 0 \iff \mu_1 = \mu_2
$$

#### Step 3. 분포 일치와의 동치

$\sigma_1 = \sigma_2$ 가정 하에 $\mu_1 = \mu_2$ 이면 $p = q$. 따라서
$$
\mu_1 = \mu_2 \iff p = q \iff D_{KL}(p\Vert q) = 0
$$

#### Step 4. 미분 검증

$f(\mu_2) = \frac{(\mu_1-\mu_2)^2}{2\sigma^2}$ 의 도함수.
$$
f'(\mu_2) = \frac{\mu_2 - \mu_1}{\sigma^2}, \qquad f''(\mu_2) = \frac{1}{\sigma^2} > 0
$$

$f''>0$ 이므로 strictly convex → $\mu_2 = \mu_1$ 이 유일한 전역 최솟값. 이때 값은 0.

$$
\boxed{\;\arg\min_{\mu_2} D_{KL}\bigl(\mathcal{N}(\mu_1,\sigma^2)\,\Vert\,\mathcal{N}(\mu_2,\sigma^2)\bigr) = \mu_1, \qquad \min D_{KL} = 0\;}
$$

---

### 4.4 MSE 와의 동치성 (출제 의도의 응용)

$y \sim \mathcal{N}(h_\theta(x), \sigma^2)$ (모델이 평균 $h_\theta(x)$ 의 가우시안을 출력) 가정.

데이터 분포 $p(y\mid x) = \mathcal{N}(y; h_\theta^{*}(x), \sigma^2)$ 와의 KL:
$$
D_{KL}(p\Vert q) = \frac{(h_\theta^{*}(x) - h_\theta(x))^2}{2\sigma^2}
$$

이는 MSE 손실 $(y - h_\theta(x))^2$ 의 상수배 (정확히 $\frac{1}{2\sigma^2}$ 배).

따라서:
$$
\text{KL 최소화} \iff \text{NLL 최소화 (MLE)} \iff \text{MSE 최소화}
$$

---

## 5. 검증

### 5.1 차원 / 단위 검증

- 일반식의 $\log\frac{\sigma_2}{\sigma_1}$ 은 무차원 (비율의 로그). $\frac{(\mu_1-\mu_2)^2}{2\sigma_2^2}$ 도 무차원 (분자·분모 모두 길이²). $\frac{\sigma_1^2}{2\sigma_2^2}$ 도 무차원. 일관성 OK.

### 5.2 특수 케이스

- $\sigma_1 = \sigma_2$ 이고 $\mu_1 = \mu_2$ 면 $D_{KL} = 0$. ✓
- $\sigma_1 = \sigma_2 = 1$, $\mu_1 = 0$, $\mu_2 = 1$ 면 $D_{KL} = 1/2$.
- 일반식에서도: $\log 1 + \frac{1 + 1}{2} - \frac{1}{2} = 0 + 1 - \frac{1}{2} = \frac{1}{2}$. ✓

### 5.3 비대칭성

KL 은 비대칭. 일반식에서 $p$ 와 $q$ 를 바꾸면 $\log\frac{\sigma_1}{\sigma_2} + \frac{\sigma_2^2 + (\mu_1-\mu_2)^2}{2\sigma_1^2} - \frac{1}{2}$ 로 다른 값. 등분산 가정 하에서만 우연히 대칭이 됨.

### 5.4 수치 sanity

$\sigma_1 = \sigma_2 = 1$, $\mu_1 = 0$, $\mu_2 = 2$ 면 $D_{KL} = 4/2 = 2$. 분포가 멀어질수록 단조 증가. ✓

---

## 6. 일반화·통찰

### 6.1 D-variate isotropic 일반화

$p = \mathcal{N}(\mu_1, \beta_1 I)$, $q = \mathcal{N}(\mu_2, \beta_2 I)$ (등방, D-차원).

$$
D_{KL}(p\Vert q) = \frac{D}{2}\log\frac{\beta_2}{\beta_1} + \frac{D\beta_1 + \|\mu_1-\mu_2\|^2}{2\beta_2} - \frac{D}{2}
$$

$\beta_1 = \beta_2 = \beta$ 단순화: $\frac{\|\mu_1-\mu_2\|^2}{2\beta}$ — 다변량 등분산 KL = L² 거리 / 2분산.

### 6.2 일반 공분산

$p = \mathcal{N}(\mu_1, \Sigma_1)$, $q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$
D_{KL}(p\Vert q) = \tfrac{1}{2}\!\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - D + \mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1-\mu_2)^\top \Sigma_2^{-1}(\mu_1-\mu_2)\right]
$$

1차원 케이스가 정확히 (Q1) 의 결과.

### 6.3 손실 함수의 통일적 시각

| 가정 분포 | NLL | 결과 손실 |
|-----------|-----|----------|
| Gaussian (등분산) | $\frac{(y - \hat y)^2}{2\sigma^2}$ + const | **MSE** |
| Bernoulli | $-y\log\hat p - (1-y)\log(1-\hat p)$ | **BCE** |
| Categorical | $-\sum_c y_c \log \hat p_c$ | **CE** |
| Laplace | $\frac{|y - \hat y|}{b}$ + const | **MAE** |

손실 = Negative Log-Likelihood = KL (데이터 분포 vs 모델 분포) 까지의 상수.

### 6.4 KL ↔ MSE ↔ MLE ↔ Wasserstein

등분산 Gaussian 에서 KL 은 평균 차이의 L² 거리에 비례. 동시에 두 가우시안 사이의 2-Wasserstein 거리도 평균 차이 + 분산 차이. 등분산이면 둘 다 평균 차이 항만 남아 본질적으로 같은 척도가 된다.

---

## 7. 시험 출제 변형 5가지

1. **(Q1 변형)** $\sigma_1, \sigma_2$ 모두 다른 일반식의 유도를 요구하되 $\mathbb{E}[(x-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2$ 보조 등식의 유도까지 명시 요구.
2. **(Q2 역방향)** 결과식 $\frac{(\mu_1-\mu_2)^2}{2\sigma^2}$ 를 주고 "이 값이 MSE 와 어떻게 연관되는가?" 라는 정성 + 수식 해석 문제.
3. **(D-variate)** D-차원 등방 가우시안으로 일반화한 KL 을 유도하라.
4. **(KL 비대칭성)** $D_{KL}(p\Vert q)$ 와 $D_{KL}(q\Vert p)$ 를 둘 다 계산하고 두 값이 다름을 보인 후 왜 다른지 의미를 설명하라.
5. **(MAP 와의 관계)** Gaussian likelihood + Gaussian prior 하에서 MAP 를 구하면 ridge 회귀와 동일함을 KL 관점에서 설명하라.

---

## 8. 백지 재현 체크리스트

- [ ] 정규분포 PDF 와 그 로그를 정확히 쓴다.
- [ ] KL 정의 $\mathbb{E}_{x\sim p}[\log p - \log q]$ 를 쓴다.
- [ ] 두 로그의 차이에서 $\log(2\pi)$ 가 상쇄되어 $\log\frac{\sigma_2}{\sigma_1}$ 이 남음을 보인다.
- [ ] $\mathbb{E}_{x\sim p}[(x-\mu_1)^2] = \sigma_1^2$ (분산 정의) 를 명시.
- [ ] $(x-\mu_2)^2 = ((x-\mu_1)+(\mu_1-\mu_2))^2$ 로 전개하고 $\mathbb{E}[x-\mu_1] = 0$ 으로 중간항 제거.
- [ ] 일반식 $\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$ 을 쓴다.
- [ ] $\sigma_1 = \sigma_2$ 대입 시 $\frac{1}{2}$ 와 $-\frac{1}{2}$ 가 상쇄됨을 본다.
- [ ] $\frac{(\mu_1-\mu_2)^2}{2\sigma^2}$ 가 비음수임을 적시.
- [ ] 등호 조건 $\mu_1 = \mu_2$ 와 분포 일치 ($p=q$) 의 동치를 명시.
- [ ] $f''(\mu_2) = 1/\sigma^2 > 0$ (strictly convex) 으로 유일 전역 최솟값임을 검증.
- [ ] (보너스) MSE 와의 동치성 한 줄 언급.

---

## 9. 핵심 공식 카드

```
KL 정의:
  D_KL(p || q) = E_{x~p}[ log p(x) - log q(x) ]

가우시안 일반식:
  D_KL( N(μ1, σ1²) || N(μ2, σ2²) )
    = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 1/2

등분산 단순화:
  D_KL( N(μ1, σ²) || N(μ2, σ²) ) = (μ1-μ2)² / (2σ²)

핵심 등식:
  E_{x~p}[(x-μ2)²] = σ1² + (μ1-μ2)²    (with x ~ N(μ1, σ1²))

MSE 동치:
  KL = MSE / (2σ²)    (Gaussian likelihood, fixed σ)
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 5/6/7 (Bernoulli MLE/MAP)**: NLL 최소화 = KL 최소화의 또 다른 사례. Q8 은 Gaussian 버전, Q5–7 은 Bernoulli 버전.
- **Quiz 9 (Triangular Prior MAP)**: 같은 NLL 프레임 안에서 prior 가 다를 때의 MAP 행동 변화. Q8 의 likelihood 항이 Gaussian 이라는 점만 다르다.
- **Quiz 10 (Backprop)**: 출력층의 손실 그래디언트가 Cross-Entropy + softmax 에서 $p - e_y$ 로 닫히는데, 이는 Gaussian + identity 의 $h_\theta - y$ (MSE 미분) 와 같은 GLM canonical link 결과. Q8 의 MSE 동치는 그 회귀 버전.
- **Quiz 3 (Softmax Jacobian)**: Q8 은 분포 거리, Q3 는 분포 출력의 미분. 두 결과 모두 Q10 의 backprop 합성에 들어간다.
- **공통 메시지**: 손실 = NLL = KL (상수 차이). "왜 MSE / CE 를 쓰는가?" 의 단일 답.
