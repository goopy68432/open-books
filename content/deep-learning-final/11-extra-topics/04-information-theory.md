---
title: "04. 정보 이론 (Information Theory)"
slug: information-theory
order: 4
---

# 04. 정보 이론 (Information Theory)

> Entropy, Cross Entropy, KL — Cross Entropy 손실의 정체.

---

## 1. Entropy (엔트로피)

이산:
$$H(p) = -\sum_i p_i \log p_i = E_p[-\log p(X)]$$

연속:
$$H(p) = -\int p(x) \log p(x)\,dx$$

### 직관
"분포의 불확실성." 균일분포는 가장 큰 엔트로피, 디랙 델타는 0.

### 단위
- log base 2: bits
- log base e (자연로그): nats
- 시험에서는 **자연로그** 가정.

### 성질
- $H(p) \geq 0$
- $H(p) \leq \log n$ (n 종류, 균일에서 최대)

---

## 2. Cross Entropy

$$H(p, q) = -\sum_i p_i \log q_i = E_p[-\log q(X)]$$

**의미:** "진짜 분포가 p인데, 모델이 q라고 예측. 평균 정보량 (nats)"

**비대칭:** $H(p, q) \neq H(q, p)$ 일반적.

### 머신러닝 손실로서

데이터 $y$ (one-hot, p = y), 모델 예측 $q = \hat{p}$:
$$H(y, \hat{p}) = -\sum_i y_i \log \hat{p}_i$$

이게 **분류 문제 CE 손실**.

---

## 3. KL Divergence (상대 엔트로피)

$$\text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} = E_p\left[\log \frac{p(X)}{q(X)}\right]$$

### 성질
- $\text{KL}(p\|q) \geq 0$ (Jensen, 증명은 [`../10-ten-proofs/07-kl-nonneg.md`](../10-ten-proofs/07-kl-nonneg.md))
- 등호 ⇔ $p = q$
- 비대칭: $\text{KL}(p\|q) \neq \text{KL}(q\|p)$
- 거리 아님 (삼각부등식 X)

---

## 4. **핵심 분해 공식** (시험 단골)

$$\boxed{H(p, q) = H(p) + \text{KL}(p \| q)}$$

### 증명

$$H(p, q) = -\sum p_i \log q_i$$
$$= -\sum p_i \log \frac{q_i}{p_i} - \sum p_i \log p_i$$
$$= \sum p_i \log \frac{p_i}{q_i} + H(p)$$
$$= \text{KL}(p \| q) + H(p). \quad \blacksquare$$

### 의미

> "Cross Entropy = 진짜 분포의 엔트로피 + KL 거리"

머신러닝에서 p (정답)는 고정 → $H(p)$ 상수 → CE 최소화 ⇔ KL 최소화.

---

## 5. Mutual Information (상호 정보)

$$I(X; Y) = \text{KL}(p(x,y) \| p(x)p(y)) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

**의미:** "X를 알면 Y에 대한 불확실성이 얼마나 줄어드나"

### 성질
- $I(X;Y) \geq 0$
- 등호 ⇔ X, Y 독립
- 대칭: $I(X;Y) = I(Y;X)$

### 응용
- VAE, InfoGAN
- Information Bottleneck
- 표현 학습

---

## 6. 응용 — Bernoulli NLL = BCE = CE 일치

이진 분류 ($y \in \{0,1\}$, 모델 예측 $\hat{p}$):

NLL:
$$-[y \log \hat{p} + (1-y)\log(1-\hat{p})]$$

이걸 CE 형태로:
- 진짜 분포 $p = (y, 1-y)$ (one-hot)
- 모델 $q = (\hat{p}, 1-\hat{p})$

$$H(p, q) = -y\log\hat{p} - (1-y)\log(1-\hat{p})$$

**즉, 베르누이 NLL = BCE = Cross Entropy**. 셋이 같은 식.

---

## 7. 카테고리 분포 (다중 클래스)

$y \in \{1, \ldots, K\}$, one-hot, 모델 $\mathbf{p} = \text{softmax}(z)$:

$$H(\mathbf{y}, \mathbf{p}) = -\sum_{k=1}^K y_k \log p_k$$

신경망 분류 손실 = **Categorical Cross Entropy**.

---

## 8. 시험 답안 — 분해 공식 증명

### [문제] $H(p, q) = H(p) + \text{KL}(p\|q)$를 증명하라.

### [풀이]

정의로 시작:
$$H(p, q) = -\sum_i p_i \log q_i.$$

$\log q_i = \log q_i - \log p_i + \log p_i = -\log(p_i/q_i) + \log p_i$로 변환:
$$H(p, q) = \sum_i p_i \log(p_i/q_i) - \sum_i p_i \log p_i = \text{KL}(p\|q) + H(p). \quad \blacksquare$$

**의미:** "CE = 진짜 엔트로피 (어쩔 수 없는 부분) + KL 거리 (모델이 좁힐 수 있는 부분)."

---

## 9. 한 줄 요약

> "엔트로피 = 불확실성, CE = H + KL, KL ≥ 0 (Jensen). 모델 학습은 KL 최소화."
