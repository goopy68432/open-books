---
title: "07. NLL 마스터 — 음의 로그우도 통합 정리"
slug: nll-master
order: 7
---

# 07. NLL 마스터 — 음의 로그우도 통합 정리

> 시험 답안에 매번 등장하는 NLL을 한 번에 정리. 이 문서 하나로 NLL 관련 모든 질문 대비.

---

## 1. NLL이란?

### 정의

$$\text{NLL}(\theta) = -\log L(\theta) = -\sum_{i=1}^n \log p(y_i | \theta)$$

(i.i.d 가정 하에)

**Negative Log-Likelihood**의 줄임말. 한국어로 **음의 로그우도**.

---

## 2. NLL 만드는 4단계

```
   Likelihood:    L(θ) = ∏ p(yᵢ|θ)        ← 곱 (i.i.d)
        │
        ▼ (log)
   Log-Likelihood: ℓ(θ) = ∑ log p(yᵢ|θ)   ← 합
        │
        ▼ (× -1)
   NLL:            -ℓ(θ) = -∑ log p(yᵢ|θ) ← 음수
```

### 각 단계의 "왜?"

| 단계 | 왜 그렇게 하나? |
|-----|-------------|
| 곱 (likelihood) | i.i.d 가정 → 결합확률 = 곱 |
| log | ① 곱→합 ② argmax 보존 ③ 수치안정 |
| × (-1) | 머신러닝의 표준은 손실 **최소화** |

---

## 3. **왜 음수를 붙이는가?** (시험 단골)

### 이유 1: 손실 함수의 표준
머신러닝/통계의 관례:
- 우도(L, ℓ)는 **최대화**
- 손실 함수는 **최소화**

NLL = -ℓ → 최대화를 최소화로 변환:
$$\arg\max_\theta \ell(\theta) = \arg\min_\theta [-\ell(\theta)] = \arg\min_\theta \text{NLL}(\theta)$$

### 이유 2: 음의 로그 → 양수 손실
$\log L < 0$ (확률이 < 1이라 로그는 음수)이라 ℓ도 음수.

$-\ell$로 음수 부호를 뒤집어 **양수 손실**(positive loss)로 만듦. 직관적.

### 이유 3: 그래디언트 하강과의 호환
신경망은 손실을 **줄이는** 방향(그래디언트 하강)으로 학습.
- $\theta \leftarrow \theta - \eta \nabla \text{NLL}(\theta)$

만약 $\ell$을 그대로 쓰면 그래디언트 **상승**해야 하므로 부호 헷갈림.

### 시험 답안 표준 문장
> "NLL = -log L로 정의하는 이유는, 머신러닝 표준이 손실 최소화이기 때문이다. NLL 최소화는 log L 최대화와 동치이며, 그래디언트 하강 알고리즘과 호환된다."

---

## 4. 분포별 NLL 비교표

| 분포 | NLL 공식 | 머신러닝 손실 |
|------|--------|------------|
| Bern(θ) | $-\sum [y_i \log\theta + (1-y_i)\log(1-\theta)]$ | **Binary Cross Entropy** |
| 이항(n,θ) | (위와 유사) | BCE |
| N(μ, σ²) | $\frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum (y_i - f_i)^2$ | **MSE** (상수 제외) |
| Categorical | $-\sum_i \sum_k y_{ik} \log p_{ik}$ | **Cross Entropy** |
| Poisson(λ) | $n\lambda - (\sum y_i)\log\lambda + \log\prod y_i!$ | Poisson loss |
| Exp(λ) | $-n\log\lambda + \lambda\sum x_i$ | (회귀에 사용) |

**핵심:** 손실 함수는 **분포 가정의 자연스러운 결과**. 임의로 정한 게 아님.

---

## 5. **베르누이 NLL = Binary Cross Entropy** (기출 4번 핵심)

### 유도

베르누이 우도:
$$L(\theta) = \prod \theta^{y_i}(1-\theta)^{1-y_i}$$

NLL:
$$\text{NLL} = -\sum_i [y_i \log\theta + (1-y_i)\log(1-\theta)]$$

이게 정확히 **BCE 손실**:
$$\text{BCE} = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

(여기서 $p_i$는 모델 예측 확률, θ 일반화)

### 의미

> "신경망이 '데이터가 어떤 확률분포 (베르누이)에서 i.i.d로 나왔다'고 가정하면, MLE 풀이가 자동으로 BCE 손실 최소화가 된다."

---

## 6. **가우스 NLL = MSE** (회귀 손실의 정체)

### 모델
$y_i = f(x_i; \theta) + \epsilon_i$, $\epsilon_i \sim N(0, \sigma^2)$ i.i.d.

### NLL
$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum (y_i - f(x_i; \theta))^2$$

θ 무관 항(첫 항) 제거 → **MSE 손실**:
$$\arg\min_\theta \text{NLL} = \arg\min_\theta \sum (y_i - f_i)^2$$

### 의미

> "MSE는 가우스 잡음 가정의 산물. 다른 잡음 (Laplace 등)이면 다른 손실 (MAE)."

---

## 7. NLL 미분 = MLE 풀이

### 일반 공식

$\hat{\theta}_{\text{MLE}} = \arg\min \text{NLL}(\theta)$

미분 = 0 (페르마):
$$\frac{d\text{NLL}}{d\theta} = -\frac{d\ell}{d\theta} = 0 \iff \frac{d\ell}{d\theta} = 0$$

→ NLL 최소화 ⇔ ℓ 최대화 ⇔ L 최대화. **세 형태 모두 같은 θ**.

### 베르누이 적용

$$\frac{d\text{NLL}}{d\theta} = -\frac{k}{\theta} + \frac{n-k}{1-\theta} = 0$$

$$\Rightarrow \theta(n-k) = (1-\theta)k \Rightarrow \theta = k/n$$

(부호 다르지만 결과 동일)

---

## 8. NLL과 MAP의 관계

MAP은 posterior 최대화. 음의 로그를 취하면:
$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta [-\log p(\theta|D)]$$
$$= \arg\min_\theta [-\log p(D|\theta) - \log p(\theta) + \log p(D)]$$

$\log p(D)$는 θ 무관 → 제거:
$$= \arg\min_\theta [\text{NLL} + (-\log p(\theta))]$$

### 해석

MAP 목적함수 = **NLL + 정규화 항** (-log prior)

| Prior | -log prior | 정규화 |
|-------|----------|-------|
| Gaussian $N(0, \tau^2 I)$ | $\frac{\|w\|^2}{2\tau^2}$ + 상수 | **L2** |
| Laplace | $\sum \|w_i\|/b$ | **L1** |
| 균일 | 상수 | 없음 (=MLE) |

→ 정규화는 **prior의 베이지안 해석**.

---

## 9. NLL의 그래디언트 (신경망 학습)

### Softmax + CE (기출 8번 응용)

$\text{NLL} = -\sum y_i \log p_i$, $p = \text{softmax}(z)$일 때:

$$\frac{\partial \text{NLL}}{\partial z_j} = p_j - y_j$$

**놀라운 단순성!** 그래디언트 = "예측 - 정답"

### 시그모이드 + BCE

이진분류 $p = \sigma(z)$, $\text{NLL} = -[y\log p + (1-y)\log(1-p)]$:

$$\frac{\partial \text{NLL}}{\partial z} = p - y$$

같은 형태 — 신경망이 학습할 때 **단순한 오차 신호**.

---

## 10. 자주 나오는 NLL 함정

### 함정 1: NLL > 0 vs ℓ < 0
- $\ell = \log L$은 **음수** (L < 1이라)
- $\text{NLL} = -\ell$은 **양수**
- 헷갈리지 말 것

### 함정 2: 부호 실수
미분 시 NLL의 부호:
$$\frac{d\text{NLL}}{d\theta} = -\frac{d\ell}{d\theta}$$

베르누이: $-(\frac{k}{\theta} - \frac{n-k}{1-\theta}) = -\frac{k}{\theta} + \frac{n-k}{1-\theta}$

= 0 풀이는 같지만, 식 적을 때 부호 주의.

### 함정 3: 상수 항 무시
NLL 식에 종종 상수 ($\log\sqrt{2\pi}\sigma$ 등) 포함. argmin에서는 무시 OK이지만, **NLL 값 자체를 묻는다면 포함**.

### 함정 4: $-\log L$ vs $\log(1/L)$
같음! ($-\log L = \log L^{-1} = \log(1/L)$)

---

## 11. 시험 답안 NLL 표준 5문장

답안에 다음 5문장 중 1~3개 포함:

1. **NLL 정의:** "NLL은 음의 로그우도 $-\log L(\theta)$로 정의된다."
2. **음수 이유:** "음수를 붙이는 이유는 머신러닝의 표준이 손실 최소화이기 때문이다."
3. **MLE 동치:** "NLL 최소화는 log L 최대화와 동치이며, 같은 $\hat{\theta}$를 산출한다."
4. **분포 → 손실:** "베르누이 NLL은 Binary Cross Entropy, 가우스 NLL은 MSE 손실로 환원된다."
5. **MAP 연결:** "MAP은 NLL + (-log prior) 최소화로, 정규화는 prior의 베이지안 해석이다."

---

## 12. NLL 한 줄 요약

> "NLL은 i.i.d 가정의 곱 → 로그로 합 → 음수로 손실. MLE 풀이의 최소화 형태이며, 분포에 따라 BCE/MSE/CE 등 익숙한 손실로 자연스럽게 변환된다."

---

## 13. 기출 4번 NLL 부분만 다시 빨리 보기

기출 4번 문제: "우도, NLL, MLE와의 관계"

### 빠른 답안 (3분)

**(1) 우도** (i.i.d → 곱):
$$L(\theta) = \theta^k(1-\theta)^{n-k}$$

**(2) NLL** (log + 음수):
$$\boxed{\text{NLL}(\theta) = -k\log\theta - (n-k)\log(1-\theta)}$$

**(3) MLE와의 관계:**
> NLL 최소화는 log L 최대화와 동치이므로 MLE의 풀이 형태로 동일하다. 미분 = 0 풀이로 $\hat{\theta}_{\text{MLE}} = k/n$이며, 이는 NLL의 최솟값이기도 하다.

---

## 다음

[`08-info-theory-bonus.md`](#) (선택, 미작성) — KL, Entropy, Information Theory 관점 (시험 범위 밖이지만 NLL 깊은 이해)

또는 다른 챕터로:
- [`../04-mle-bernoulli/`](../04-mle-bernoulli/) — NLL 8단계 체인 직접 풀이
- [`../99-strategy/02-derivation-method.md`](../99-strategy/02-derivation-method.md) — 7단계 답안 템플릿
