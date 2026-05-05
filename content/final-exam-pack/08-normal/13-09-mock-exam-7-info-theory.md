---
title: "딥러닝 이론 모의고사 #7 — 정보이론 + 통합"
slug: 09-mock-exam-7-info-theory
order: 13
---

# 딥러닝 이론 모의고사 #7 — 정보이론 + 통합

> 배점 100점 / 8문제

---

## 문제 1. [12점] 엔트로피

**(a)** [4점] 엔트로피 $H(p) = -\sum_x p(x)\log p(x)$의 정의를 쓰고, "불확실성의 측도"라는 의미를 설명하시오.

**(b)** [4점] $p = (0.5, 0.5)$와 $p = (0.9, 0.1)$의 엔트로피를 각각 계산하고, 어느 쪽이 더 큰지 설명하시오 (log base 2 사용).

**(c)** [4점] 엔트로피가 최대인 분포는? 최소인 분포는? 그 이유를 설명하시오.

---

## 문제 2. [12점] Cross-Entropy

**(a)** [4점] $CE(p,q) = -\sum_x p(x)\log q(x)$의 정의를 쓰고, "p의 관점에서 q로 코딩하는 비용"이라는 해석을 설명하시오.

**(b)** [4점] CE가 딥러닝에서 Loss 함수로 사용되는 이유를 NLL과의 관계로 설명하시오.

**(c)** [4점] $CE(p,q) \geq H(p)$를 보이시오 (KL ≥ 0 이용).

---

## 문제 3. [15점] KL Divergence

**(a)** [4점] $KL(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$의 정의와 "두 분포의 차이"라는 의미를 설명하시오.

**(b)** [6점] Jensen 부등식을 사용하여 $KL(p\|q) \geq 0$를 증명하시오. $-\log$가 볼록인 이유를 포함하시오.

**(c)** [5점] $KL(p\|q) \neq KL(q\|p)$를 구체적 수치 예시로 보이시오. 이것이 "KL은 거리(metric)가 아니다"라는 의미인 이유를 설명하시오.

---

## 문제 4. [12점] CE = KL + H

**(a)** [6점] $KL(p\|q) = CE(p,q) - H(p)$를 유도하시오.

**(b)** [6점] 이로부터 "$p$가 고정일 때 CE 최소화 = KL 최소화"를 보이시오. 이것이 분류 문제에서 왜 중요한지 (p = 데이터 분포, q = 모델 분포) 설명하시오.

---

## 문제 5. [12점] NLL = CE = KL 등가 체인

분류 문제에서 데이터 분포 $p_{\text{data}}$, 모델 분포 $q_\theta$일 때:

**(a)** [6점] $\text{NLL}(\theta) = -\frac{1}{n}\sum_{i=1}^n \log q_\theta(y_i|x_i)$가 $CE(p_{\text{data}}, q_\theta)$의 몬테카를로 근사임을 보이시오.

**(b)** [6점] 따라서 NLL 최소화 = CE 최소화 = KL 최소화임을 설명하시오. "모델 학습의 목표는 모델 분포를 데이터 분포에 가깝게 만드는 것"이라는 통합적 해석을 서술하시오.

---

## 문제 6. [12점] 가우시안 간 KL

**(a)** [6점] $KL(\mathcal{N}(\mu_1,\sigma_1^2) \| \mathcal{N}(\mu_2,\sigma_2^2))$를 1차원에서 유도하시오.

힌트: $KL = \int p\log(p/q)$를 직접 계산. $p = \mathcal{N}(\mu_1,\sigma_1^2)$, $q = \mathcal{N}(\mu_2,\sigma_2^2)$.

**(b)** [6점] $\sigma_1 = \sigma_2 = \sigma$일 때 $KL \propto \|\mu_1 - \mu_2\|^2$가 됨을 보이시오. 이것이 "가우시안 가정 하에서 KL ≈ MSE"라는 의미를 설명하시오.

---

## 문제 7. [13점] 최대 엔트로피와 가우시안

**(a)** [7점] 평균 $\mu$, 분산 $\sigma^2$가 주어졌을 때 엔트로피를 최대화하는 분포가 가우시안임을 라그랑주 승수법으로 유도하시오.

제약: $\int p(x)dx = 1$, $\int xp(x)dx = \mu$, $\int(x-\mu)^2p(x)dx = \sigma^2$

**(b)** [6점] 이 결과가 "가우시안 가정은 가장 보수적(불확실성 최대)인 선택"이라는 의미와, CLT에 의한 정당화와의 관계를 설명하시오.

---

## 문제 8. [12점] 정보이론과 딥러닝 통합

다음의 등가 관계들을 각각 1-2줄로 설명하시오.

**(a)** CE 최소화 = KL 최소화 (왜?)
**(b)** NLL 최소화 = CE 최소화 (왜?)
**(c)** Gaussian NLL = MSE (왜?)
**(d)** Bernoulli NLL = BCE (왜?)
**(e)** MAP = NLL + Prior (왜?)
**(f)** Gaussian Prior → L2 Reg (왜?)

---
---

# 모범답안

## 답 1.
### (b)
$H(0.5, 0.5) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1$ bit
$H(0.9, 0.1) = -0.9\log_2 0.9 - 0.1\log_2 0.1 ≈ 0.469$ bit
$(0.5,0.5)$이 더 큼. 균일할수록 불확실성이 높으므로.

### (c)
최대: Uniform 분포 ($H = \log K$, K개 값). 모든 결과가 동등하게 가능 → 최대 불확실성.
최소: Delta 분포 ($H = 0$, 하나의 값에 확률 1). 결과가 확실 → 불확실성 없음.

## 답 3.
### (b)
$KL(p\|q) = \mathbb{E}_p[-\log(q/p)] = -\mathbb{E}_p[\log(q(X)/p(X))]$

$-\log$는 볼록 (왜: $(-\log)'' = 1/x^2 > 0$).
Jensen ($f$ 볼록 → $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$):
$$\geq -\log\mathbb{E}_p[q(X)/p(X)] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log\sum_x q(x) = -\log 1 = 0 \quad \square$$

### (c)
$p=(0.9,0.1)$, $q=(0.5,0.5)$
$KL(p\|q) = 0.9\ln(0.9/0.5) + 0.1\ln(0.1/0.5) ≈ 0.9(0.588) + 0.1(-1.609) ≈ 0.368$
$KL(q\|p) = 0.5\ln(0.5/0.9) + 0.5\ln(0.5/0.1) ≈ 0.5(-0.588) + 0.5(1.609) ≈ 0.510$
$0.368 \neq 0.510$ → 비대칭. 거리(metric)의 조건인 대칭성 $d(x,y)=d(y,x)$ 불만족.

## 답 6.
### (a)
$\log p = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x-\mu_1)^2}{2\sigma_1^2}$
$\log q = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x-\mu_2)^2}{2\sigma_2^2}$

$KL = \mathbb{E}_p[\log p - \log q]$
$= \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$

### (b)
$\sigma_1=\sigma_2=\sigma$: $KL = \frac{(\mu_1-\mu_2)^2}{2\sigma^2}$ $\propto \|\mu_1-\mu_2\|^2$
→ 분산이 같은 가우시안끼리의 KL = MSE를 $\sigma^2$로 스케일링한 것. $\square$

## 답 7.
### (a)
$\mathcal{L} = -\int p\log p + \lambda_0(\int p-1) + \lambda_1(\int xp-\mu) + \lambda_2(\int(x-\mu)^2 p-\sigma^2)$

변분: $\delta\mathcal{L}/\delta p = 0$:
$-\log p(x) - 1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 = 0$
$\log p(x) = -1+\lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2$
$p(x) = \exp[-1+\lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2]$

$\lambda_2 < 0$이어야 정규화 가능 → $\lambda_2 = -1/(2\sigma^2)$로 놓으면
$p(x) \propto \exp[-(x-\mu)^2/(2\sigma^2)]$ = **가우시안** $\square$

## 답 8.
(a) $H(p)$가 상수이므로 $\min CE = \min(KL + H) = \min KL$
(b) NLL = $-\frac{1}{n}\sum\log q(y_i|x_i) \approx \mathbb{E}_p[-\log q] = CE$
(c) Gaussian의 $-\log p$에서 지수항만 남으면 $(y-h(x))^2$
(d) Bernoulli의 $-\log p$에서 $-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$
(e) MAP = argmax [log Likelihood + log Prior] → argmin [NLL - log Prior]
(f) $-\log\mathcal{N}(0,\sigma_p^2) \propto \|\theta\|^2$ → L2 항
